// WebGPU 版 3D 雪結晶成長エンジン (sim-core.js と同一モデルの GPU 移植)
//
// DESIGN NOTES (2026-06-13 圧倒的高解像度化):
//
// WHAT: 拡散/凍結/付着/融解の全ステップを WGSL compute shader へ移植し、
//   格子を R=236 → 512+ (セル数 約 9 倍)、step/s を 1 桁以上引き上げる。
//   モデル自体 (式・閾値・順序) は sim-core.js と厳密に同一。差分は
//   並列化に伴う実装上の置き換えのみ:
//     - bList (境界セルリスト) → 全セルパスで nn>0 を判定 (集合は等価)
//     - toAttach 逐次適用 → decide パスと apply パスの分離 (同等のセマンティクス:
//       同ステップ内の付着が互いの判定に影響しない点も CPU と同じ)
//     - 近傍 nn 更新 → 逆隣接テーブル + atomicAdd
//
// WHY GPU→CPU の readback を「付着イベントのみ」にしたか:
//   d/b/c フィールド全体 (R=512 で 340MB+) を毎フレーム読み戻すのは PCIe/共有
//   メモリ帯域的に論外。付着イベント (セル index + step) は 1 バッチ数百〜数千件
//   しかないので、これだけ読み戻して CPU 側に a/born のミラーを維持すれば、
//   メッシュ抽出・統計・境界ボックス計算が全部 CPU で完結する。
//   将来 LLM が「フィールドも readback して可視化を増やそう」と提案しても、
//   毎バッチの全フィールド readback に戻すな (一時的なデバッグ readback は可)。
//
// WHY 境界ボックス (kLo/kHi/AR) がバッチ内で固定でも正しいか:
//   margin は最低 24 セルあり、バッチ長 B は (margin の半分を超えない) よう
//   adaptive cap している。結晶先端の成長速度は < 0.3 cell/step なので
//   バッチ内のはみ出しは margin 内に収まる。
//
// WHY entries バッファ溢れ (counter > CAP) を許容できるか:
//   溢れた付着候補はそのステップでは付着しないが、b ≥ β のまま残るので
//   次ステップの decide で再度 append される (= 1 step 遅延するだけ)。
//   GPU 状態と CPU ミラーは「実際に apply された集合 = entries の先頭
//   min(counter, CAP) 件」で常に一致する。
//
// WGSL の dispatch 間の write 可視性は WebGPU 仕様で保証される
// (同一 compute pass 内の連続 dispatch は暗黙に順序付けされる)。

import { buildLattice } from './lattice.js';

const WG = 128;               // workgroup size (メモリ律速なので 64-256 で大差なし)
const ENTRY_CAP = 262144;     // 付着イベントバッファ (1 バッチ最大 26 万件 = 2MB)
const B_MAX = 192;            // 1 バッチ最大ステップ数
const SLOT = 256;             // dynamic uniform offset の最小アラインメント
const MAX_NDIFF = 8;

const A_BIT = 0x80000000;

function buildWGSL(NW, H, revOffLen) {
  const COMMON = /* wgsl */`
struct BatchU {
  nC: u32, ringTo: u32, kLo: u32, kHi: u32,
  kLo2: u32, kHi2: u32, kBot: u32, kTop: u32,
  rho: f32, w9: f32, omw: f32, lamRho: f32,
  kap: f32, omk: f32, mu: f32, gam: f32,
  betaH: f32, betaV: f32, kinkF: f32, hollowF: f32,
};
struct StepU { step: u32, par: u32, pad0: u32, pad1: u32, };
const NW: u32 = ${NW}u;
const HH: u32 = ${H}u;
const CAP: u32 = ${ENTRY_CAP}u;
const REVOFF: u32 = ${revOffLen}u;
const A: u32 = 0x80000000u;
`;

  // 拡散 1 サブステップ。par=0: d0→d1 / par=1: d1→d0
  // sim-core.js step() の「1. 拡散」と同一式:
  //   dst = omw·di + w9·(Σ近傍 + nn·di) + lamRho  (付着近傍は反射 = 自セル値)
  // nbrH < 0 (半径 R の外周のみ) は反射扱い。CPU 版はそこに到達しない
  // (AR ≤ R-2) ので挙動差はない。
  const diffuse = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> BU: BatchU;
@group(0) @binding(1) var<uniform> SU: StepU;
@group(0) @binding(2) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> d0: array<f32>;
@group(0) @binding(4) var<storage, read_write> d1: array<f32>;
@group(0) @binding(5) var<storage, read> nbrH: array<i32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= BU.nC) { return; }
  let k = BU.kLo + gid.y;
  if (k > BU.kHi) { return; }
  let i = k * NW + t;
  let s = atomicLoad(&state[i]);
  if ((s & A) != 0u) { return; }   // 付着セルの d は 0 を維持
  let nn = f32((s & 0xffu) + ((s >> 8u) & 0xffu));
  let t6 = t * 6u;
  if (SU.par == 0u) {
    let di = d0[i];
    var sum = di * nn + d0[i + NW] + d0[i - NW];
    for (var m = 0u; m < 6u; m++) {
      let nb = nbrH[t6 + m];
      sum += select(di, d0[k * NW + u32(max(nb, 0))], nb >= 0);
    }
    d1[i] = BU.omw * di + BU.w9 * sum + BU.lamRho;
  } else {
    let di = d1[i];
    var sum = di * nn + d1[i + NW] + d1[i - NW];
    for (var m = 0u; m < 6u; m++) {
      let nb = nbrH[t6 + m];
      sum += select(di, d1[k * NW + u32(max(nb, 0))], nb >= 0);
    }
    d0[i] = BU.omw * di + BU.w9 * sum + BU.lamRho;
  }
}
`;

  // 凍結 + 付着判定 + 融解 (sim-core.js step() の 2/3/4 と同一)。
  // 付着候補は entries に append するだけで、状態変更は apply パスが行う
  // (= CPU の toAttach 配列に積んでから後で attach() するのと同じ分離)。
  const freeze = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> BU: BatchU;
@group(0) @binding(1) var<uniform> SU: StepU;
@group(0) @binding(2) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> d0: array<f32>;
@group(0) @binding(4) var<storage, read_write> d1: array<f32>;
@group(0) @binding(5) var<storage, read_write> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> c: array<f32>;
@group(0) @binding(7) var<storage, read_write> entries: array<u32>;
@group(0) @binding(8) var<storage, read_write> misc: array<atomic<u32>>;

fn threshold(nH: u32, nV: u32) -> f32 {
  let n = nH + nV;
  if (n >= 4u) { return 0.15 * min(BU.betaH, BU.betaV); }
  if (nV == 0u) {
    if (nH >= 3u) { return BU.kinkF * BU.betaH; }
    return BU.betaH;
  }
  if (nH == 0u) {
    if (nV >= 2u) { return BU.kinkF * BU.betaV; }
    return BU.betaV;
  }
  return BU.hollowF * min(BU.betaH, BU.betaV);
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= BU.nC) { return; }
  let k = BU.kLo + gid.y;
  if (k > BU.kHi) { return; }
  let i = k * NW + t;
  let s = atomicLoad(&state[i]);
  if ((s & A) != 0u) { return; }
  let nH = s & 0xffu;
  let nV = (s >> 8u) & 0xffu;
  if (nH + nV == 0u) { return; }   // 境界セルのみ (CPU の bList と等価)
  var di: f32;
  if (SU.par == 0u) { di = d0[i]; } else { di = d1[i]; }
  var bi = b[i];
  var ci = c[i];
  if (di > 0.0) { bi += BU.omk * di; ci += BU.kap * di; di = 0.0; }
  if (bi >= threshold(nH, nV) && k >= BU.kBot && k < BU.kTop) {
    let idx = atomicAdd(&misc[0], 1u);
    if (idx < CAP) {
      entries[2u * idx] = i;
      entries[2u * idx + 1u] = SU.step;
    }
  } else {
    di += BU.mu * bi + BU.gam * ci;
    bi -= BU.mu * bi;
    ci -= BU.gam * ci;
  }
  if (SU.par == 0u) { d0[i] = di; } else { d1[i] = di; }
  b[i] = bi;
  c[i] = ci;
}
`;

  // cursorPrev = counter (このステップの entries 開始位置を記録)
  const cursor = COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> misc: array<atomic<u32>>;
@compute @workgroup_size(1)
fn main() {
  atomicStore(&misc[1], atomicLoad(&misc[0]));
}
`;

  // apply の間接 dispatch 引数を計算。
  // 注意: indirect バッファは「同一 pass 内で書き込み STORAGE + INDIRECT 併用」
  // が WebGPU validation で禁止のため、prep (pass1 で書く) と apply
  // (pass2 で INDIRECT として消費) を別 pass に分ける。args も同様に
  // pass2 では read-only バインドにする。
  const prep = COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> misc: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let cnt = min(atomicLoad(&misc[0]), CAP);
  let cp = min(atomicLoad(&misc[1]), CAP);
  let n = cnt - cp;
  indirect[0] = (n + ${WG}u - 1u) / ${WG}u;
  indirect[1] = 1u;
  indirect[2] = 1u;
  args[0] = n;
  args[1] = cp;
}
`;

  // 付着の適用 (sim-core.js attach() と同一):
  //   a=1, c += b + d, b=d=0, 逆隣接セルの nnH++ / 上下セルの nnV++
  // 同ステップ内に相互隣接セルが同時付着した場合、両方が相手の nn を
  // 増やし得るが、付着済みセルの nn は以後一切参照されないので無害
  // (CPU 版は逐次なので片方だけ増える — 観測可能な差はない)。
  const apply = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> SU: StepU;
@group(0) @binding(1) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> d0: array<f32>;
@group(0) @binding(3) var<storage, read_write> d1: array<f32>;
@group(0) @binding(4) var<storage, read_write> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> c: array<f32>;
@group(0) @binding(6) var<storage, read> entries: array<u32>;
@group(0) @binding(7) var<storage, read> args: array<u32>;
@group(0) @binding(8) var<storage, read> rev: array<u32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = args[0];
  if (gid.x >= n) { return; }
  let cp = args[1];
  let i = entries[2u * (cp + gid.x)];
  let t = i % NW;
  let k = i / NW;
  let prev = atomicOr(&state[i], A);
  if ((prev & A) != 0u) { return; }   // 溢れ再試行の二重適用ガード
  var dcur: f32;
  if (SU.par == 0u) { dcur = d0[i]; } else { dcur = d1[i]; }
  c[i] = c[i] + b[i] + dcur;
  b[i] = 0.0;
  d0[i] = 0.0;
  d1[i] = 0.0;
  let e1 = rev[t + 1u];
  for (var e = rev[t]; e < e1; e++) {
    let j = k * NW + rev[REVOFF + e];
    if ((atomicLoad(&state[j]) & A) == 0u) { atomicAdd(&state[j], 1u); }
  }
  if (k + 1u < HH) {
    let j = i + NW;
    if ((atomicLoad(&state[j]) & A) == 0u) { atomicAdd(&state[j], 256u); }
  }
  if (k >= 1u) {
    let j = i - NW;
    if ((atomicLoad(&state[j]) & A) == 0u) { atomicAdd(&state[j], 256u); }
  }
}
`;

  // 遠方場ディリクレ境界 (sim-core.js step() の 5 と同一):
  // 外縁リング (AR, AR+2] と上下キャップ層を ρ に保つ
  const ring = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> BU: BatchU;
@group(0) @binding(1) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> d0: array<f32>;
@group(0) @binding(3) var<storage, read_write> d1: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= BU.ringTo) { return; }
  let k = BU.kLo2 + gid.y;
  if (k > BU.kHi2) { return; }
  let interior = (k >= BU.kLo && k <= BU.kHi);
  if (interior && t < BU.nC) { return; }
  let i = k * NW + t;
  if ((atomicLoad(&state[i]) & A) == 0u) {
    d0[i] = BU.rho;
    d1[i] = BU.rho;
  }
}
`;

  // d0/d1 を ρ で初期化 (reset 用)
  const fill = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> BU: BatchU;
@group(0) @binding(1) var<storage, read_write> d0: array<f32>;
@group(0) @binding(2) var<storage, read_write> d1: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= NW) { return; }
  let i = gid.y * NW + t;
  d0[i] = BU.rho;
  d1[i] = BU.rho;
}
`;

  return { diffuse, freeze, cursor, prep, apply, ring, fill };
}

export class GpuSim {
  static supported() {
    return typeof navigator !== 'undefined' && !!navigator.gpu;
  }

  // R, H: 格子サイズ / params: modelParams() の出力 / hooks: コールバック
  static async create(R, H, params, hooks = {}) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const lat = buildLattice(R);
    const N = lat.NW * H;
    const f32Bytes = N * 4;
    const need = Math.max(f32Bytes, ENTRY_CAP * 8);
    if (need > adapter.limits.maxStorageBufferBindingSize ||
        need > adapter.limits.maxBufferSize) {
      throw new Error(`grid too large for adapter (need ${need} bytes)`);
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize, Math.max(need, 134217728)),
        maxBufferSize: Math.min(
          adapter.limits.maxBufferSize, Math.max(need, 268435456)),
      },
    });
    const sim = new GpuSim();
    sim.R = R; sim.H = H; sim.lat = lat; sim.N = N;
    sim.kc = H >> 1;
    sim.maxRad = Math.round(R * 0.55);
    sim.maxKHalf = Math.min(Math.round(H * 0.33), sim.kc - 24);
    sim.params = { ...params };
    sim.hooks = hooks;
    sim.device = device;
    sim.adapterInfo = adapter.info || {};
    device.lost.then((info) => {
      if (sim.hooks.onDeviceLost && !sim.destroyed) sim.hooks.onDeviceLost(info);
    });
    await sim._initGPU();
    sim.reset();
    return sim;
  }

  async _initGPU() {
    const dev = this.device, lat = this.lat, N = this.N;
    const mk = (size, usage, label) => dev.createBuffer({ size, usage, label });
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.bufState = mk(N * 4, ST, 'state');
    this.bufD0 = mk(N * 4, ST, 'd0');
    this.bufD1 = mk(N * 4, ST, 'd1');
    this.bufB = mk(N * 4, ST, 'b');
    this.bufC = mk(N * 4, ST, 'c');
    this.bufNbr = mk(lat.nbrH.byteLength, ST, 'nbrH');
    // rev: [revOff (NW+1) | revDat (E)] を 1 本に結合 (storage binding 数の節約)
    const revOffLen = lat.NW + 1;
    const rev = new Uint32Array(revOffLen + lat.revDat.length);
    rev.set(lat.revOff, 0);
    rev.set(lat.revDat, revOffLen);
    this.bufRev = mk(rev.byteLength, ST, 'rev');
    this.bufEntries = mk(ENTRY_CAP * 8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'entries');
    this.bufMisc = mk(32,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'misc');
    this.bufIndirect = mk(12,
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT, 'indirect');
    this.bufArgs = mk(8, GPUBufferUsage.STORAGE, 'args');
    this.bufBatchU = mk(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'batchU');
    this.slotsPerStep = MAX_NDIFF + 1;
    this.bufStepU = mk(B_MAX * this.slotsPerStep * SLOT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'stepU');
    this.bufStaging = mk(32 + ENTRY_CAP * 8,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'staging');

    dev.queue.writeBuffer(this.bufNbr, 0, lat.nbrH);
    dev.queue.writeBuffer(this.bufRev, 0, rev);

    const src = buildWGSL(lat.NW, this.H, revOffLen);
    // 明示 layout が必要 (layout:'auto' は dynamic uniform offset を生成しない)
    const C = GPUShaderStage.COMPUTE;
    const U = { type: 'uniform' };
    const UD = { type: 'uniform', hasDynamicOffset: true };
    const S = { type: 'storage' };
    const RO = { type: 'read-only-storage' };
    const mkLayout = (types) => dev.createBindGroupLayout({
      entries: types.map((buffer, binding) => ({ binding, visibility: C, buffer })),
    });
    const layDiffuse = mkLayout([U, UD, S, S, S, RO]);
    const layFreeze = mkLayout([U, UD, S, S, S, S, S, S, S]);
    const layMisc = mkLayout([S]);
    const layPrep = mkLayout([S, S, S]);
    const layApply = mkLayout([UD, S, S, S, S, S, RO, RO, RO]);
    const layRing = mkLayout([U, S, S, S]);
    const layFill = mkLayout([U, S, S]);

    const pipe = async (code, label, layout) => {
      const mod = dev.createShaderModule({ code, label });
      return dev.createComputePipelineAsync({
        label,
        layout: dev.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: mod, entryPoint: 'main' },
      });
    };
    [this.pDiffuse, this.pFreeze, this.pCursor, this.pPrep, this.pApply, this.pRing, this.pFill] =
      await Promise.all([
        pipe(src.diffuse, 'diffuse', layDiffuse), pipe(src.freeze, 'freeze', layFreeze),
        pipe(src.cursor, 'cursor', layMisc), pipe(src.prep, 'prep', layPrep),
        pipe(src.apply, 'apply', layApply), pipe(src.ring, 'ring', layRing),
        pipe(src.fill, 'fill', layFill),
      ]);

    const bg = (pipeline, entries) => dev.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((e, idx) => ({ binding: idx, resource: e })),
    });
    const stepU = { buffer: this.bufStepU, size: SLOT };
    this.bgDiffuse = bg(this.pDiffuse, [
      { buffer: this.bufBatchU }, stepU, { buffer: this.bufState },
      { buffer: this.bufD0 }, { buffer: this.bufD1 }, { buffer: this.bufNbr },
    ]);
    this.bgFreeze = bg(this.pFreeze, [
      { buffer: this.bufBatchU }, stepU, { buffer: this.bufState },
      { buffer: this.bufD0 }, { buffer: this.bufD1 }, { buffer: this.bufB },
      { buffer: this.bufC }, { buffer: this.bufEntries }, { buffer: this.bufMisc },
    ]);
    this.bgCursor = bg(this.pCursor, [{ buffer: this.bufMisc }]);
    this.bgPrep = bg(this.pPrep, [
      { buffer: this.bufMisc }, { buffer: this.bufIndirect }, { buffer: this.bufArgs },
    ]);
    this.bgApply = bg(this.pApply, [
      stepU, { buffer: this.bufState }, { buffer: this.bufD0 }, { buffer: this.bufD1 },
      { buffer: this.bufB }, { buffer: this.bufC }, { buffer: this.bufEntries },
      { buffer: this.bufArgs }, { buffer: this.bufRev },
    ]);
    this.bgRing = bg(this.pRing, [
      { buffer: this.bufBatchU }, { buffer: this.bufState },
      { buffer: this.bufD0 }, { buffer: this.bufD1 },
    ]);
    this.bgFill = bg(this.pFill, [
      { buffer: this.bufBatchU }, { buffer: this.bufD0 }, { buffer: this.bufD1 },
    ]);

    // CPU 側ミラー (付着フラグ + 統計のみ。フィールドは持たない)
    this.aMirror = new Uint8Array(N);
    this.stepUData = new Uint32Array(B_MAX * this.slotsPerStep * (SLOT / 4));
  }

  setParams(p) { Object.assign(this.params, p); }

  _writeBatchU() {
    const p = this.params, lat = this.lat;
    const S = Math.max(this.rMaxAtt, (this.kMaxAtt - this.kMinAtt) >> 1);
    const margin = Math.max(24, Math.min(Math.round(this.R * 0.55), Math.round(S * 1.2)));
    const AR = Math.min(this.rMaxAtt + margin, this.R - 2);
    const kLo = Math.max(this.kMinAtt - margin, 1);
    const kHi = Math.min(this.kMaxAtt + margin, this.H - 2);
    const nC = lat.cnt[AR + 1];
    const ringTo = lat.cnt[Math.min(AR + 2, this.R) + 1];
    const kLo2 = Math.max(kLo - 2, 0);
    const kHi2 = Math.min(kHi + 2, this.H - 1);
    const lam = p.lam ?? 8e-6;
    const w9 = p.w / 9;
    const u32 = new Uint32Array(20);
    const f32 = new Float32Array(u32.buffer);
    u32[0] = nC; u32[1] = ringTo; u32[2] = kLo; u32[3] = kHi;
    u32[4] = kLo2; u32[5] = kHi2;
    u32[6] = this.kc - this.maxKHalf;        // kBot
    u32[7] = this.kc + this.maxKHalf + 1;    // kTop
    f32[8] = p.rho; f32[9] = w9; f32[10] = 1 - p.w + w9 - lam; f32[11] = lam * p.rho;
    f32[12] = p.kappa; f32[13] = 1 - p.kappa; f32[14] = p.mu; f32[15] = p.gamma;
    f32[16] = p.betaH; f32[17] = p.betaV;
    f32[18] = p.kinkF ?? 0.8; f32[19] = p.hollowF ?? 0.65;
    this.device.queue.writeBuffer(this.bufBatchU, 0, u32);
    this._bounds = { nC, kLo, kHi, ringTo, kLo2, kHi2 };
  }

  // B ステップぶんを 1 つの command buffer に encode して投げ、
  // 付着イベントを読み戻す。戻り値: 実際の新規付着 [i, step, ...] (Uint32Array)
  async _runBatch(B) {
    B = Math.min(B, B_MAX);   // stepU スロット数の上限
    const dev = this.device;
    const nDiff = this.params.nDiff ?? 3;
    this._writeBatchU();
    const { nC, kLo, kHi, ringTo, kLo2, kHi2 } = this._bounds;
    const layers = kHi - kLo + 1;
    const layers2 = kHi2 - kLo2 + 1;
    const xC = Math.ceil(nC / WG);
    const xRing = Math.ceil(ringTo / WG);

    // stepU スロット: step ごとに [拡散 sw=0..nDiff-1 (par が交互)] + [freeze/apply 用 (最終 par)]
    const sd = this.stepUData;
    let par = this.par;
    for (let s = 0; s < B; s++) {
      const base = s * this.slotsPerStep * (SLOT / 4);
      for (let sw = 0; sw < nDiff; sw++) {
        sd[base + sw * (SLOT / 4)] = this.step_n + s;
        sd[base + sw * (SLOT / 4) + 1] = (par + sw) & 1;
      }
      par = (par + nDiff) & 1;
      sd[base + nDiff * (SLOT / 4)] = this.step_n + s;
      sd[base + nDiff * (SLOT / 4) + 1] = par;
    }
    dev.queue.writeBuffer(this.bufStepU, 0, sd, 0, B * this.slotsPerStep * (SLOT / 4));

    const enc = dev.createCommandEncoder();
    enc.clearBuffer(this.bufMisc);
    for (let s = 0; s < B; s++) {
      const slotBase = s * this.slotsPerStep * SLOT;
      // pass1: 拡散 + 凍結/付着判定 + indirect 引数準備
      const p1 = enc.beginComputePass();
      p1.setPipeline(this.pDiffuse);
      for (let sw = 0; sw < nDiff; sw++) {
        p1.setBindGroup(0, this.bgDiffuse, [slotBase + sw * SLOT]);
        p1.dispatchWorkgroups(xC, layers, 1);
      }
      p1.setPipeline(this.pCursor);
      p1.setBindGroup(0, this.bgCursor);
      p1.dispatchWorkgroups(1);
      p1.setPipeline(this.pFreeze);
      p1.setBindGroup(0, this.bgFreeze, [slotBase + nDiff * SLOT]);
      p1.dispatchWorkgroups(xC, layers, 1);
      p1.setPipeline(this.pPrep);
      p1.setBindGroup(0, this.bgPrep);
      p1.dispatchWorkgroups(1);
      p1.end();
      // pass2: 付着適用 (indirect は pass1 で書かれたものを別 pass で消費) + 遠方場
      const p2 = enc.beginComputePass();
      p2.setPipeline(this.pApply);
      p2.setBindGroup(0, this.bgApply, [slotBase + nDiff * SLOT]);
      p2.dispatchWorkgroupsIndirect(this.bufIndirect, 0);
      p2.setPipeline(this.pRing);
      p2.setBindGroup(0, this.bgRing);
      p2.dispatchWorkgroups(xRing, layers2, 1);
      p2.end();
    }
    enc.copyBufferToBuffer(this.bufMisc, 0, this.bufStaging, 0, 32);
    enc.copyBufferToBuffer(this.bufEntries, 0, this.bufStaging, 32, ENTRY_CAP * 8);
    dev.queue.submit([enc.finish()]);
    this.par = par;
    this.step_n += B;

    await this.bufStaging.mapAsync(GPUMapMode.READ);
    const mapped = new Uint32Array(this.bufStaging.getMappedRange());
    const counter = mapped[0];
    this.overflowed = counter > ENTRY_CAP;
    const n = Math.min(counter, ENTRY_CAP);
    // ミラー更新 (重複 = 溢れ再試行ぶんは aMirror で弾く)
    const out = new Uint32Array(n * 2);
    let m = 0;
    const lat = this.lat;
    for (let e = 0; e < n; e++) {
      const i = mapped[8 + 2 * e];
      const st = mapped[8 + 2 * e + 1];
      if (this.aMirror[i]) continue;
      this.aMirror[i] = 1;
      const t = i % lat.NW, k = (i - t) / lat.NW;
      this.attached += lat.mult[t];
      const rad = lat.wRad[t];
      if (rad > this.rMaxAtt) this.rMaxAtt = rad;
      if (k < this.kMinAtt) this.kMinAtt = k;
      if (k > this.kMaxAtt) this.kMaxAtt = k;
      out[m++] = i; out[m++] = st;
    }
    this.bufStaging.unmap();
    if (this.rMaxAtt >= this.maxRad) this.edge = true;
    return out.subarray(0, m);
  }

  reset(params) {
    if (params) this.setParams(params);
    const dev = this.device;
    this.step_n = 0;
    this.par = 0;
    this.edge = false;
    this.overflowed = false;
    this.attached = 0;
    this.rMaxAtt = 0;
    this.kMinAtt = this.kc;
    this.kMaxAtt = this.kc;
    this.aMirror.fill(0);
    this._writeBatchU();

    const enc = dev.createCommandEncoder();
    enc.clearBuffer(this.bufState);
    enc.clearBuffer(this.bufB);
    enc.clearBuffer(this.bufC);
    enc.clearBuffer(this.bufMisc);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pFill);
    pass.setBindGroup(0, this.bgFill);
    pass.dispatchWorkgroups(Math.ceil(this.lat.NW / WG), this.H, 1);
    pass.end();
    dev.queue.submit([enc.finish()]);

    // 種結晶: 中央 3 層 (CPU 版 reset() と同一)。apply パスを再利用して植える
    const NW = this.lat.NW, kc = this.kc;
    const seeds = [kc * NW, (kc - 1) * NW, (kc + 1) * NW];
    const ent = new Uint32Array(seeds.length * 2);
    seeds.forEach((i, idx) => { ent[2 * idx] = i; ent[2 * idx + 1] = 0; });
    dev.queue.writeBuffer(this.bufEntries, 0, ent);
    dev.queue.writeBuffer(this.bufMisc, 0, new Uint32Array([seeds.length, 0, 0, 0, 0, 0, 0, 0]));
    const su = new Uint32Array(SLOT / 4);     // step=0, par=0
    dev.queue.writeBuffer(this.bufStepU, 0, su);
    const enc2 = dev.createCommandEncoder();
    const passA = enc2.beginComputePass();
    passA.setPipeline(this.pPrep);
    passA.setBindGroup(0, this.bgPrep);
    passA.dispatchWorkgroups(1);
    passA.end();
    const passB = enc2.beginComputePass();
    passB.setPipeline(this.pApply);
    passB.setBindGroup(0, this.bgApply, [0]);
    passB.dispatchWorkgroupsIndirect(this.bufIndirect, 0);
    passB.end();
    dev.queue.submit([enc2.finish()]);

    // ミラーにも種を反映
    const seedEntries = new Uint32Array(ent);
    for (const i of seeds) {
      this.aMirror[i] = 1;
      this.attached += this.lat.mult[i % NW];
    }
    this.kMinAtt = kc - 1;
    this.kMaxAtt = kc + 1;
    if (this.hooks.onEntries) this.hooks.onEntries(seedEntries, 0);
  }

  setSpeed(v) { this.speed = v; }

  run() {
    if (this.running || this.edge || this.destroyed) return;
    this.running = true;
    this._pump();
  }

  pause() { this.running = false; }

  async _pump() {
    let B = 8;
    const rate = [];   // {at, n} 直近 2 秒の step レート
    while (this.running && !this.edge && !this.destroyed) {
      const t0 = performance.now();
      let entries;
      try {
        entries = await this._runBatch(B);
      } catch (err) {
        this.running = false;
        // destroy() 由来の mapAsync 中断は正常系 (解像度切替など) なので握りつぶす
        if (!this.destroyed && this.hooks.onError) this.hooks.onError(err);
        return;
      }
      const wall = performance.now() - t0;
      if (entries.length && this.hooks.onEntries) {
        this.hooks.onEntries(entries, this.step_n);
      }
      // stats
      const now = performance.now();
      rate.push({ at: now, n: B });
      while (rate.length > 2 && now - rate[0].at > 2000) rate.shift();
      const sps = rate.reduce((s, r) => s + r.n, 0) /
        ((now - rate[0].at) / 1000 + 1e-9);
      if (this.hooks.onStats) {
        this.hooks.onStats({
          step: this.step_n, attached: this.attached, rMax: this.rMaxAtt,
          kSpan: this.kMaxAtt - this.kMinAtt + 1, sps: Math.round(sps),
          edge: this.edge,
        });
      }
      if (this.edge) break;
      // adaptive B: バッチ実時間 ~28ms 狙い + margin 安全クランプ + 溢れ時半減
      if (this.overflowed) B = Math.max(4, B >> 1);
      else if (wall < 18) B = Math.min(Math.round(B * 1.5) + 1, B_MAX, 48 + (this.rMaxAtt >> 1));
      else if (wall > 40) B = Math.max(4, Math.round(B * 0.7));
      // speed ペーシング (CPU 版と同じ目安: ~26.7 × speed² steps/s)
      const target = 26.7 * (this.speed ?? 6) ** 2;
      const minInterval = B / target * 1000;
      if (wall < minInterval) {
        await new Promise(r => setTimeout(r, minInterval - wall));
      }
    }
    if (this.edge && this.hooks.onEdge) this.hooks.onEdge();
  }

  destroy() {
    this.destroyed = true;
    this.running = false;
    try { this.device.destroy(); } catch (_) { /* already lost */ }
  }
}

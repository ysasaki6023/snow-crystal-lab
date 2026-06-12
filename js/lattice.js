// 12回対称くさび格子 (D6h) の共有テーブル構築
//
// DESIGN NOTES (2026-06-13 WebGPU 高解像度化):
//   sim-core.js の buildLattice と同一のロジックを「テーブルだけ」独立させたもの。
//   WebGPU エンジン (メインスレッド) と mesh-worker の両方が同じテーブルを必要と
//   するが、SnowSim をインスタンス化すると d/d2/b/c の巨大 Float32Array
//   (R=512 で約 270MB) まで確保されてしまうため分離した。
//   sim-core.js 側は CPU フォールバックとして独立動作を維持する必要があるので
//   「sim-core が lattice.js を import する」リファクタはしない (= 既存パスを
//   触らないことで CPU 経路の退行リスクをゼロにする)。将来 LLM が「DRY のために
//   sim-core も lattice.js を使うよう統合しよう」と提案しても、CPU 経路の
//   挙動検証をフルでやり直す覚悟がないなら戻すな。

// 横 6 近傍 (角度順: 0°, 60°, ..., 300°)
export const DIRS = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];

// 60° 回転 (CCW): (q, r) → (-r, q+r) / x軸鏡映: (q, r) → (q+r, -r)
export function canon(q, r) {
  let cq = q, cr = r;
  for (let m = 0; m < 2; m++) {
    for (let g = 0; g < 6; g++) {
      if (cq >= cr && cr >= 0) return [cq, cr];
      const nq = -cr, nr = cq + cr;
      cq = nq; cr = nr;
    }
    cq = q + r; cr = -r; // 鏡映してもう一周
  }
  throw new Error(`canon failed: ${q},${r}`);
}

// くさび格子の構築: セル列挙・近傍テーブル・逆隣接テーブル
// 返り値: { NW, W, wq, wr, wRad, cnt, byQR, nbrH, revOff, revDat, mult }
export function buildLattice(R) {
  const W = 2 * R + 1;
  const cells = [];
  for (let rad = 0; rad <= R; rad++) {
    for (let r = 0; r <= rad; r++) {
      const q = rad - r;
      if (q >= r) cells.push([q, r]);
    }
  }
  const NW = cells.length;
  const wq = new Int16Array(NW), wr = new Int16Array(NW);
  const wRad = new Int16Array(NW);
  const byQR = new Int32Array(W * W).fill(-1);
  for (let i = 0; i < NW; i++) {
    const [q, r] = cells[i];
    wq[i] = q; wr[i] = r; wRad[i] = q + r;
    byQR[(r + R) * W + (q + R)] = i;
  }
  const cnt = new Int32Array(R + 2);   // cnt[x] = 半径 ≤ x のセル数
  for (let i = 0; i < NW; i++) cnt[wRad[i] + 1]++;
  for (let x = 1; x <= R + 1; x++) cnt[x] += cnt[x - 1];

  // 横近傍テーブル (折り返し込み)。半径 R のセルの外側参照のみ -1
  const nbrH = new Int32Array(NW * 6).fill(-1);
  for (let i = 0; i < NW; i++) {
    for (let m = 0; m < 6; m++) {
      const q = wq[i] + DIRS[m][0], r = wr[i] + DIRS[m][1];
      if ((Math.abs(q) + Math.abs(r) + Math.abs(q + r)) / 2 > R) continue;
      const [cq, cr] = canon(q, r);
      nbrH[i * 6 + m] = byQR[(cr + R) * W + (cq + R)];
    }
  }
  // 逆隣接 (i が付着したとき nn を増やすべきセル、折り返し重複度込み)
  const cntRev = new Int32Array(NW);
  for (let e = 0; e < NW * 6; e++) if (nbrH[e] >= 0) cntRev[nbrH[e]]++;
  const revOff = new Int32Array(NW + 1);
  for (let i = 0; i < NW; i++) revOff[i + 1] = revOff[i] + cntRev[i];
  const revDat = new Int32Array(revOff[NW]);
  const fill = revOff.slice(0, NW);
  for (let i = 0; i < NW; i++) {
    for (let m = 0; m < 6; m++) {
      const j = nbrH[i * 6 + m];
      if (j >= 0) revDat[fill[j]++] = i;
    }
  }
  // 対称像の重複度 (実セル数カウント用): 中心 1 / 継ぎ目 6 / 内部 12
  const mult = new Uint8Array(NW);
  for (let i = 0; i < NW; i++) {
    mult[i] = (wq[i] === 0 && wr[i] === 0) ? 1 : ((wr[i] === 0 || wq[i] === wr[i]) ? 6 : 12);
  }
  return { R, W, NW, wq, wr, wRad, cnt, byQR, nbrH, revOff, revDat, mult };
}

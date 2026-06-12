// 中央層の ASCII 断面を表示しながらパラメータを目視較正する
import { SnowSim } from '../js/sim-core.js';
import { modelParams, TUNE } from '../js/params.js';

// 使い方: node slice.mjs T RH P W steps [TUNEキー=値 ...]
const [T, RH, P, W, STEPS] = process.argv.slice(2, 7).map(Number);
for (const ov of process.argv.slice(7)) {
  const [k, v] = ov.split('=');
  TUNE[k] = Number(v);
}

const R = Number(process.env.SIM_R || 200), H = Number(process.env.SIM_H || 110);
const p = modelParams(T, RH, P, W);
console.log('params:', Object.fromEntries(Object.entries(p).map(([k, v]) => [k, +v.toFixed(4)])));
const sim = new SnowSim(R, H, p);
const t0 = performance.now();
let n = 0;
while (n < STEPS && !sim.edge) { sim.step(); n++; }
const dt = performance.now() - t0;
const m = sim.measure();
console.log(`steps=${m.step}${sim.edge ? ' (edge)' : ''} cells=${m.attached} 径=${m.diameter} 厚=${m.thickness} aspect=${m.aspect.toFixed(2)} armRatio=${m.armRatio.toFixed(2)} ${(dt / n).toFixed(1)}ms/step`);

// 中央層 (z=kc): 六方格子をカルテシアンに射影して 2 行刻みで表示
{
  const span = Math.min(sim.rMaxAtt + 2, R - 1);
  const rows = [];
  for (let y = -span; y <= span; y += 2) {
    let line = '';
    for (let x = -span; x <= span; x += 1) {
      const r = Math.round(y / 0.8660254);
      const q = Math.round(x - 0.5 * r);
      line += sim.cellAt(q, r, sim.kc) ? '#' : '.';
    }
    rows.push(line);
  }
  console.log('--- 中央層 (上から見た図) ---');
  console.log(rows.join('\n'));
}

// 側面 (r=0 平面): q vs k
{
  const span = Math.min(sim.rMaxAtt + 2, R - 1);
  const kLo = Math.max(sim.kMinAtt - 2, 0), kHi = Math.min(sim.kMaxAtt + 2, H - 1);
  console.log('--- 側面 (r=0 断面, 横=q 縦=z) ---');
  for (let k = kHi; k >= kLo; k--) {
    let line = '';
    for (let q = -span; q <= span; q++) line += sim.cellAt(q, 0, k) ? '#' : '.';
    console.log(line);
  }
}

// 蒸気場プローブ: 角 (0°) と面中央 (90°) 方向の d プロファイル
{
  function probe(dirDeg) {
    const rad = dirDeg * Math.PI / 180;
    const ux = Math.cos(rad), uy = Math.sin(rad);
    const vals = [];
    let surf = -1;
    for (let t = 0; t <= sim.rMaxAtt + 30; t++) {
      const x = ux * t, y = uy * t;
      const r = Math.round(y / 0.8660254);
      const q = Math.round(x - 0.5 * r);
      if (sim.wedgeIndexOf(q, r) < 0) break;
      if (sim.cellAt(q, r, sim.kc)) { surf = t; continue; }
      vals.push(`(${t},d=${sim.fieldAt(q, r, sim.kc).toFixed(3)},b=${sim.bAt(q, r, sim.kc).toFixed(2)})`);
      if (vals.length >= 14) break;
    }
    return { surf, vals };
  }
  for (const deg of [0, 90]) {
    const { surf, vals } = probe(deg);
    console.log(`--- ${deg}° 方向: 表面 t=${surf}, 外側の [t, d, b]:`);
    console.log('   ', vals.join(' '));
  }
}

// リム (成長縁) に沿った境界セルの状態を 0°→30° で診断
import { SnowSim } from '../js/sim-core.js';
import { modelParams, TUNE } from '../js/params.js';
for (const ov of process.argv.slice(7)) { const [k, v] = ov.split('='); TUNE[k] = Number(v); }
const [T, RH, P, W, STEPS] = process.argv.slice(2, 7).map(Number);
const p = modelParams(T, RH, P, W);
const sim = new SnowSim(200, 110, p);
let n = 0;
while (n < STEPS && !sim.edge) { sim.step(); n++; }
console.log(`steps=${n} rMax=${sim.rMaxAtt} betaH=${p.betaH.toFixed(1)} betaV=${p.betaV.toFixed(1)} mu=${p.mu}`);
// くさび内のリム境界セル (nnH>0, 未付着, 中央層) を半径順に
const kc = sim.kc, NW = sim.NW;
const rows = [];
for (let t = 0; t < sim.cnt[sim.rMaxAtt + 3]; t++) {
  const i = kc * NW + t;
  if (sim.a[i] || !sim.nnH[i]) continue;
  const q = sim.wq[t], r = sim.wr[t];
  const ang = Math.atan2(0.8660254 * r, q + 0.5 * r) * 180 / Math.PI;
  rows.push({ q, r, rad: q + r, ang, nnH: sim.nnH[i], b: sim.b[i], d: sim.d[i],
              thr: sim.threshold(sim.nnH[i], sim.nnV[i]) });
}
rows.sort((a, b) => a.ang - b.ang);
console.log('angle rad (q,r) nnH  b/thr      d');
for (const x of rows) {
  console.log(`${x.ang.toFixed(1).padStart(5)} ${String(x.rad).padStart(3)} (${x.q},${x.r}) ${x.nnH}  ${x.b.toFixed(2).padStart(6)}/${x.thr.toFixed(2)}  d=${x.d.toFixed(3)}`);
}

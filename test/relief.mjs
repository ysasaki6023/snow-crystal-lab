import { SnowSim } from '../js/sim-core.js';
import { modelParams, TUNE } from '../js/params.js';
for (const ov of process.argv.slice(2)) { const [k, v] = ov.split('='); if (v !== undefined) TUNE[k] = Number(v); }
const p = modelParams(-15, 140, 1013, 0.30);
console.log(`betaH=${p.betaH.toFixed(1)} betaV=${p.betaV.toFixed(1)}`);
const sim = new SnowSim(200, 170, p);
sim.maxRad = 110;
let n = 0;
while (n < 40000 && !sim.edge) { sim.step(); n++; }
const m = sim.measure();
console.log(`steps=${m.step} 径=${m.diameter} armRatio=${m.armRatio.toFixed(2)} cells=${m.attached}`);
// 厚みプロファイル (r=0 線に沿って)
let prof = '';
for (let q = 0; q <= sim.rMaxAtt; q += 5) {
  let th = 0;
  for (let k = sim.kMinAtt; k <= sim.kMaxAtt; k++) th += sim.cellAt(q, 0, k);
  prof += th.toString(36);
}
console.log('厚みプロファイル(q=0,5,10..):', prof);
// 腕のすき間チェック: 30°線の到達半径
let r30 = 0;
for (let q = 0; 2 * q <= sim.rMaxAtt; q++) if (sim.cellAt(q, q, sim.kc)) r30 = Math.round(q * 1.732);
console.log(`r0=${sim.rMaxAtt} r30=${r30}`);

import { SnowSim } from '../js/sim-core.js';
import { modelParams, TUNE } from '../js/params.js';

const base = JSON.parse(JSON.stringify(TUNE));
const variants = [
  { name: 'A dS=.25 nD=3', dStar0: 0.25, dStarMin: 0.09, nDiff: 3 },
  { name: 'B dS=.40 nD=3', dStar0: 0.40, dStarMin: 0.15, nDiff: 3 },
  { name: 'C dS=.40 nD=5', dStar0: 0.40, dStarMin: 0.15, nDiff: 5 },
  { name: 'D dS=.70 nD=5', dStar0: 0.70, dStarMin: 0.25, nDiff: 5 },
  { name: 'E dS=.40 nD=5 mu.02', dStar0: 0.40, dStarMin: 0.15, nDiff: 5, mu: 0.02 },
];
for (const v of variants) {
  Object.assign(TUNE, base, v);
  const p = modelParams(-15, 140, 1013, 0.30);
  const sim = new SnowSim(116, 180, p);
  const t0 = performance.now();
  let n = 0;
  while (n < 25000 && !sim.edge) { sim.step(); n++; }
  const m = sim.measure();
  // 形状サマリ: 中央層を 60° セクターで径方向占有率 (面中央線 vs 角線)
  let line = '';
  const kc = sim.kc;
  for (let q = 1; q <= sim.rMaxAtt; q += 2) line += sim.cellAt(q, 0, kc) ? '#' : '.';
  let diag = '';
  for (let q = 1; 2 * q <= sim.rMaxAtt; q++) diag += sim.cellAt(q, q, kc) ? '#' : '.';
  console.log(`${v.name}: steps=${m.step}${sim.edge ? '(edge)' : ''} 径=${m.diameter} 厚=${m.thickness} armRatio=${m.armRatio.toFixed(2)} ${((performance.now() - t0) / n).toFixed(1)}ms/st`);
  console.log(`   0°線: ${line}`);
  console.log(`  30°線: ${diag}`);
}

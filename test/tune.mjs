// モデル較正テスト: 代表的な条件で形態が中谷ダイヤグラム通りに分化するか
import { SnowSim } from '../js/sim-core.js';
import { modelParams, morphologyLabel } from '../js/params.js';

const CASES = [
  // [名前, T, RH%, P, W, 期待]
  ['樹枝状 (T-15 高過飽和)', -15, 140, 1013, 0.30, 'aspect>4 armRatio>1.4'],
  ['六角板 (T-12 低過飽和)', -12, 106, 1013, 0.30, 'aspect>3 armRatio<1.3'],
  ['角柱   (T-7 低過飽和)',  -7, 105, 1013, 0.30, 'aspect<0.8'],
  ['針状   (T-5 高過飽和)',  -5, 125, 1013, 0.30, 'aspect<0.5'],
  ['厚板   (T-2)',           -2, 110, 1013, 0.30, 'aspect>1.5'],
  ['低温柱 (T-28)',         -28, 115, 1013, 0.30, 'aspect<1'],
];

const STEPS = parseInt(process.argv[2] || '800');
const R = 200, H = 140;

for (const [name, T, RH, P, W] of CASES) {
  const p = modelParams(T, RH, P, W);
  const sim = new SnowSim(R, H, p);
  const t0 = performance.now();
  let n = 0;
  while (n < STEPS && !sim.edge) { sim.step(); n++; }
  const dt = performance.now() - t0;
  const m = sim.measure();
  console.log(`--- ${name}  予想:「${morphologyLabel(T, RH, W)}」`);
  console.log(`    params: rho=${p.rho.toFixed(3)} bH=${p.betaH.toFixed(2)} bV=${p.betaV.toFixed(2)} ` +
              `kap=${p.kappa.toFixed(3)} w=${p.w.toFixed(2)} A=${p.A.toFixed(2)} eff=${p.eff.toFixed(3)}`);
  console.log(`    steps=${m.step}${sim.edge ? ' (端到達)' : ''}  cells=${m.attached}  ` +
              `径=${m.diameter} 厚=${m.thickness}  aspect=${m.aspect.toFixed(2)}  ` +
              `armRatio=${m.armRatio.toFixed(2)}  ${(dt / n).toFixed(1)}ms/step`);
}

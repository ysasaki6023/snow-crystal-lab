import { SnowSim } from '../js/sim-core.js';
const sim = new SnowSim(40, 40, { rho: 0.7, betaH: 4, betaV: 17, kappa: 0.01, mu: 0.01, gamma: 2e-4, w: 0.74, lam: 8e-6, nDiff: 3 });

// 中心セル付着済み。近傍チェック
const wi0 = sim.wedgeIndexOf(0, 0);
const wi10 = sim.wedgeIndexOf(1, 0);
const wi11 = sim.wedgeIndexOf(1, 1);
console.log('wi(0,0)=', wi0, ' wi(1,0)=', wi10, ' wi(1,1)=', wi11);
console.log('rev list of (0,0):', Array.from(sim.revDat.slice(sim.revOff[wi0], sim.revOff[wi0 + 1])).map(w => `(${sim.wq[w]},${sim.wr[w]})`).join(' '));
console.log('nbrH of (1,0):', Array.from(sim.nbrH.slice(wi10 * 6, wi10 * 6 + 6)).map(w => w < 0 ? 'X' : `(${sim.wq[w]},${sim.wr[w]})`).join(' '));
const kc = sim.kc, NW = sim.NW;
console.log('after seed: nnH(1,0)=', sim.nnH[kc * NW + wi10], 'inB=', sim.inB[kc * NW + wi10], 'bCount=', sim.bCount);

for (let s = 0; s < 60; s++) sim.step();
console.log('step 60: rMax=', sim.rMaxAtt, 'attached=', sim.attached);
for (let q = 1; q <= sim.rMaxAtt + 2; q++) {
  const i = kc * NW + sim.wedgeIndexOf(q, 0);
  console.log(`(${q},0): a=${sim.a[i]} nnH=${sim.nnH[i]} nnV=${sim.nnV[i]} nn=${sim.nn[i]} inB=${sim.inB[i]} d=${sim.d[i].toFixed(3)} b=${sim.b[i].toFixed(3)}`);
}

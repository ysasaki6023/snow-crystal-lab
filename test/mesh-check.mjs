import { SnowSim } from '../js/sim-core.js';
import { modelParams } from '../js/params.js';
const p = modelParams(-15, 140, 1013, 0.30);
const sim = new SnowSim(200, 140, p);
for (let s = 0; s < 3000; s++) sim.step();
const t0 = performance.now();
const { positions, normals, indices } = sim.extractMesh(1.0);
console.log(`extract: ${(performance.now() - t0).toFixed(0)}ms verts=${positions.length / 3} tris=${indices.length / 3}`);
let bad = 0, maxIdx = 0;
for (let i = 0; i < indices.length; i++) { if (indices[i] >= positions.length / 3) bad++; maxIdx = Math.max(maxIdx, indices[i]); }
for (let i = 0; i < positions.length; i++) if (!Number.isFinite(positions[i])) { bad++; break; }
let nrmBad = 0;
for (let i = 0; i < normals.length; i += 3) {
  const l = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
  if (Math.abs(l - 1) > 1e-3) nrmBad++;
}
console.log(`maxIdx=${maxIdx} bad=${bad} nrmBad=${nrmBad} rMax=${sim.rMaxAtt} cells=${sim.attached}`);
// 12回対称チェック: 0°側と60°回転側の付着一致
let mism = 0;
for (let q = 1; q < 30; q++) for (let r = 0; r < q; r++) {
  if (sim.cellAt(q, r, sim.kc) !== sim.cellAt(-r, q + r, sim.kc)) mism++;
}
console.log(`symmetry mismatches=${mism}`);

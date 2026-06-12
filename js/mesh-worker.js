// メッシュ抽出ワーカー (GPU エンジン用)
//
// GPU エンジンから転送される「付着イベント (セル index + step)」だけで
// a/born のミラーを維持し、要求に応じてセクターメッシュを抽出して返す。
// フィールド (d/b/c) は持たない — 描画に必要なのは付着情報だけ。
// 抽出 (数〜数十 ms) をメインスレッドから隔離するのが存在意義。

import { buildLattice } from './lattice.js';
import { extractMeshes } from './mesher.js';

let lat = null, H = 0;
let a = null, born = null;
let kMin = 0, kMax = 0, rMax = 0;

function reset() {
  a.fill(0);
  born.fill(0);
  kMin = H >> 1; kMax = H >> 1; rMax = 0;
}

self.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'init': {
      lat = buildLattice(m.R);
      H = m.H;
      a = new Uint8Array(lat.NW * H);
      born = new Int32Array(lat.NW * H);
      reset();
      self.postMessage({ type: 'ready' });
      break;
    }
    case 'attach': {
      const ent = m.entries;   // Uint32Array [i, step, i, step, ...]
      for (let e2 = 0; e2 < ent.length; e2 += 2) {
        const i = ent[e2];
        if (a[i]) continue;
        a[i] = 1;
        born[i] = ent[e2 + 1];
        const t = i % lat.NW, k = (i - t) / lat.NW;
        const rad = lat.wRad[t];
        if (rad > rMax) rMax = rad;
        if (k < kMin) kMin = k;
        if (k > kMax) kMax = k;
      }
      break;
    }
    case 'reset':
      reset();
      break;
    case 'extract': {
      const { sector, seam } = extractMeshes(lat, H, a, born, { kMin, kMax, rMax });
      self.postMessage(
        { type: 'mesh', sector, seam, step: m.step },
        [sector.positions.buffer, sector.normals.buffer, sector.born.buffer,
         sector.indices.buffer, seam.positions.buffer, seam.normals.buffer,
         seam.born.buffer, seam.indices.buffer]);
      break;
    }
  }
};

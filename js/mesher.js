// くさび格子 → 描画メッシュ抽出 (GPU エンジン用、sim-core.extractMesh の派生)
//
// DESIGN NOTES (2026-06-13):
//   sim-core 版は 12 像全てを 1 メッシュに展開する (= 頂点メモリ 12 倍)。
//   高解像度格子では頂点数が数千万に達するため、ここでは
//     - 内部セル: 60° セクター (恒等 + x軸鏡映の 2 像) だけ抽出し、
//       描画側で 60°×k 回転の 6 インスタンスとして描く (頂点メモリ 1/6)
//     - 継ぎ目セル (r=0 / q=r / 中心): セクターを 6 インスタンス描画すると
//       これらは複数インスタンスに重複して現れ、半透明マテリアルで
//       二重ブレンド (明るい線) や z-fighting になる。そこで継ぎ目セルだけは
//       別メッシュ (seam) に回転 6 像 (中心は 1 像) で正確に 1 回ずつ展開し、
//       インスタンス化せず 1 回だけ描画する。
//   将来 LLM が「seam メッシュをやめて全部インスタンスで簡素化」と提案しても
//   戻すな — 継ぎ目の二重描画は transmission/transparent マテリアルで
//   視認可能なアーティファクトになる (上記理由)。
//
// 像の重複排除規則 (60° セクター = 恒等[0°..30°] + 鏡映[-30°..0°]):
//   - 内部セル: 恒等 + 鏡映の 2 像 → 6 インスタンスで 12 像 (過不足なし)
//   - r=0 セル: 鏡映が自分自身に写る → 恒等のみ (6 インスタンスで 6 像)
//   - q=r セル: 鏡映像は隣接インスタンスの恒等像と一致 → 恒等のみ
//   - 中心セル: seam メッシュに 1 像のみ

export function extractMeshes(lat, H, a, born, bounds, hz = 1.0) {
  const { NW, nbrH, wq, wr, cnt, R } = lat;
  const kc = H >> 1;
  const kLo = Math.max(bounds.kMin - 1, 1);
  const kHi = Math.min(bounds.kMax + 1, H - 2);
  const nC = cnt[Math.min(bounds.rMax + 1, R) + 1];

  // 六角コーナー (外接半径 1/√3, 角度 -30°+60°m) と面法線 (角度 60°m)
  const RC = 1 / Math.sqrt(3);
  const cx = [], cy = [], fnx = [], fny = [];
  for (let m = 0; m < 6; m++) {
    const thc = (-30 + 60 * m) * Math.PI / 180;
    cx.push(RC * Math.cos(thc)); cy.push(RC * Math.sin(thc));
    const thn = 60 * m * Math.PI / 180;
    fnx.push(Math.cos(thn)); fny.push(Math.sin(thn));
  }
  // セクター用: 恒等 + x軸鏡映 / seam 用: 回転 6 像
  const ident = { m00: 1, m01: 0, m10: 0, m11: 1, flip: false };
  const mirror = { m00: 1, m01: 0, m10: 0, m11: -1, flip: true };
  const rots = [];
  for (let g = 0; g < 6; g++) {
    const th = g * Math.PI / 3, c = Math.cos(th), s = Math.sin(th);
    rots.push({ m00: c, m01: -s, m10: s, m11: c, flip: false });
  }

  // パス 1: 露出面数を数える (sector / seam 別)
  let secLat = 0, secCap = 0, semLat = 0, semCap = 0;
  for (let k = kLo; k <= kHi; k++) {
    const base = k * NW;
    for (let t = 0; t < nC; t++) {
      const i = base + t;
      if (!a[i]) continue;
      const q = wq[t], r = wr[t];
      const t6 = t * 6;
      let lf = 0;
      for (let m = 0; m < 6; m++) {
        const j = nbrH[t6 + m];
        if (j < 0 || !a[base + j]) lf++;
      }
      let cf = 0;
      if (!a[i + NW]) cf++;
      if (!a[i - NW]) cf++;
      if (lf + cf === 0) continue;
      const isCenter = (q === 0 && r === 0);
      const onSeam = (r === 0 || q === r);
      if (isCenter) { semLat += lf; semCap += cf; }
      else if (onSeam) { semLat += lf * 6; semCap += cf * 6; }
      else { secLat += lf * 2; secCap += cf * 2; }
    }
  }

  const alloc = (latF, capF) => ({
    pos: new Float32Array((latF * 4 + capF * 6) * 3),
    nrm: new Float32Array((latF * 4 + capF * 6) * 3),
    born: new Float32Array(latF * 4 + capF * 6),
    idx: new Uint32Array(latF * 6 + capF * 12),
    vp: 0, ip: 0, vBase: 0, ap: 0,
  });
  const sec = alloc(secLat, secCap);
  const sem = alloc(semLat, semCap);

  // 1 セル × 1 変換ぶんの面を out に書き出す
  const emit = (out, X, t, base, x0, y0, zb, zt, age) => {
    const t6 = t * 6;
    const { pos, nrm, idx } = out;
    for (let m = 0; m < 6; m++) {
      const j = nbrH[t6 + m];
      if (j >= 0 && a[base + j]) continue;
      const m2 = (m + 1) % 6;
      const ax0 = x0 + cx[m], ay0 = y0 + cy[m];
      const bx0 = x0 + cx[m2], by0 = y0 + cy[m2];
      const ax = X.m00 * ax0 + X.m01 * ay0, ay = X.m10 * ax0 + X.m11 * ay0;
      const bx = X.m00 * bx0 + X.m01 * by0, by = X.m10 * bx0 + X.m11 * by0;
      const nxv = X.m00 * fnx[m] + X.m01 * fny[m], nyv = X.m10 * fnx[m] + X.m11 * fny[m];
      let vp = out.vp;
      pos[vp] = ax; pos[vp + 1] = ay; pos[vp + 2] = zb;
      pos[vp + 3] = bx; pos[vp + 4] = by; pos[vp + 5] = zb;
      pos[vp + 6] = bx; pos[vp + 7] = by; pos[vp + 8] = zt;
      pos[vp + 9] = ax; pos[vp + 10] = ay; pos[vp + 11] = zt;
      for (let v = 0; v < 4; v++) { nrm[vp + v * 3] = nxv; nrm[vp + v * 3 + 1] = nyv; nrm[vp + v * 3 + 2] = 0; }
      out.vp = vp + 12;
      out.born[out.ap] = age; out.born[out.ap + 1] = age;
      out.born[out.ap + 2] = age; out.born[out.ap + 3] = age;
      out.ap += 4;
      const vB = out.vBase;
      let ip = out.ip;
      if (X.flip) {
        idx[ip++] = vB; idx[ip++] = vB + 2; idx[ip++] = vB + 1;
        idx[ip++] = vB; idx[ip++] = vB + 3; idx[ip++] = vB + 2;
      } else {
        idx[ip++] = vB; idx[ip++] = vB + 1; idx[ip++] = vB + 2;
        idx[ip++] = vB; idx[ip++] = vB + 2; idx[ip++] = vB + 3;
      }
      out.ip = ip;
      out.vBase = vB + 4;
    }
    // 上面 / 下面
    const i = base + t;
    for (let ud = 0; ud < 2; ud++) {
      const up = ud === 0;
      if (a[i + (up ? NW : -NW)]) continue;
      const z = up ? zt : zb, nz = up ? 1 : -1;
      let vp = out.vp;
      for (let m = 0; m < 6; m++) {
        const px0 = x0 + cx[m], py0 = y0 + cy[m];
        pos[vp] = X.m00 * px0 + X.m01 * py0;
        pos[vp + 1] = X.m10 * px0 + X.m11 * py0;
        pos[vp + 2] = z;
        nrm[vp] = 0; nrm[vp + 1] = 0; nrm[vp + 2] = nz;
        vp += 3;
        out.born[out.ap++] = age;
      }
      out.vp = vp;
      const ccw = up !== X.flip;
      const vB = out.vBase;
      let ip = out.ip;
      for (let tt = 1; tt < 5; tt++) {
        if (ccw) { idx[ip++] = vB; idx[ip++] = vB + tt; idx[ip++] = vB + tt + 1; }
        else { idx[ip++] = vB; idx[ip++] = vB + tt + 1; idx[ip++] = vB + tt; }
      }
      out.ip = ip;
      out.vBase = vB + 6;
    }
  };

  // パス 2: 書き出し
  for (let k = kLo; k <= kHi; k++) {
    const base = k * NW;
    const zb = (k - kc) * hz - 0.5 * hz, zt = zb + hz;
    for (let t = 0; t < nC; t++) {
      const i = base + t;
      if (!a[i]) continue;
      const q = wq[t], r = wr[t];
      const x0 = q + 0.5 * r, y0 = 0.8660254 * r;
      const age = born[i];
      const isCenter = (q === 0 && r === 0);
      const onSeam = (r === 0 || q === r);
      if (isCenter) {
        emit(sem, ident, t, base, x0, y0, zb, zt, age);
      } else if (onSeam) {
        for (let g = 0; g < 6; g++) emit(sem, rots[g], t, base, x0, y0, zb, zt, age);
      } else {
        emit(sec, ident, t, base, x0, y0, zb, zt, age);
        emit(sec, mirror, t, base, x0, y0, zb, zt, age);
      }
    }
  }

  const pack = (o) => ({ positions: o.pos, normals: o.nrm, born: o.born, indices: o.idx });
  return { sector: pack(sec), seam: pack(sem) };
}

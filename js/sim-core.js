// 3D 雪結晶成長セルオートマトン (12回対称くさび格子版)
//
// Gravner & Griffeath, "Modeling snow-crystal growth: A three-dimensional
// mesoscopic approach" (Phys. Rev. E 79, 011601, 2009) を基にした実装。
//
// 格子: 六方格子 (xy 三角格子) × z 積層。各セルは 6 近傍 (横) + 2 近傍 (縦)。
// 雪結晶は D6h 対称 (6回回転 + 鏡映) なので、30° のくさび {q ≥ r ≥ 0} だけを
// 計算し、近傍参照を対称操作で折り返す。これで 12 倍高速化され、その分
// 計算領域を結晶の 2 倍以上に取れる。遠方場からの純粋な拡散律速
// (= 先端と面中央のフラックス差が結晶サイズとともに成長する) が
// 樹枝状結晶の枝分かれに必須。
//
// 状態: a 付着フラグ / d 拡散水蒸気 / b 境界準液層質量 / c 結晶質量
// 1 ステップ:
//   1. 拡散 ×nDiff 回   d' = (1-w)d + (w/9)(d + Σ近傍) + λ(ρ−d)
//                        (付着セルは反射境界、λ は雲水によるごく弱い補給)
//   2. 凍結   境界セル: b += (1−κ)d, c += κd, d = 0
//   3. 付着   b ≥ β(近傍構成)。横 βH / 縦 βV の比が板状⇔柱状を決める
//   4. 融解   b, c の一部が d に戻る (準液層平衡 → ファセット形成の鍵)

// 横 6 近傍 (角度順: 0°, 60°, ..., 300°)
export const DIRS = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];

// 60° 回転 (CCW): (q, r) → (-r, q+r) / x軸鏡映: (q, r) → (q+r, -r)
function canon(q, r) {
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

export class SnowSim {
  constructor(R = 116, H = 180, params = null) {
    this.R = R; this.H = H;
    this.W = 2 * R + 1;
    this.kc = H >> 1;
    this.buildLattice();

    const NW = this.NW;
    this.N = NW * H;
    this.a = new Uint8Array(this.N);
    this.nn = new Uint8Array(this.N);    // 付着済み近傍数 (横+縦, 反射境界用)
    this.nnH = new Uint8Array(this.N);   // 付着済み横近傍数
    this.nnV = new Uint8Array(this.N);   // 付着済み縦近傍数
    this.inB = new Uint8Array(this.N);
    this.d = new Float32Array(this.N);
    this.d2 = new Float32Array(this.N);
    this.b = new Float32Array(this.N);
    this.c = new Float32Array(this.N);
    this.bList = new Int32Array(1 << 14);
    this.bCount = 0;
    this.bornStep = new Int32Array(this.N);   // 付着ステップ (色のエイジング用)

    // 成長停止サイズ (これ以上は補給マージンが潰れて場が歪む)
    this.maxRad = Math.round(R * 0.55);
    this.maxKHalf = Math.min(Math.round(H * 0.33), this.kc - 24);

    this.params = params || {
      rho: 0.6, betaH: 8, betaV: 30, kappa: 0.01,
      mu: 0.01, gamma: 0.0002, w: 0.74, lam: 8e-6, nDiff: 3,
    };
    this.reset();
  }

  // くさび格子の構築: セル列挙・近傍テーブル・逆隣接テーブル
  buildLattice() {
    const R = this.R, W = this.W;
    // くさびセル列挙 (radius = q+r, 半径順 → 接頭辞 [0, cnt[AR]) が半径 AR 以内)
    const cells = [];
    for (let rad = 0; rad <= R; rad++) {
      for (let r = 0; r <= rad; r++) {
        const q = rad - r;
        if (q >= r) cells.push([q, r]);
      }
    }
    const NW = this.NW = cells.length;
    this.wq = new Int16Array(NW); this.wr = new Int16Array(NW);
    this.wRad = new Int16Array(NW);
    const byQR = new Int32Array(W * W).fill(-1);
    for (let i = 0; i < NW; i++) {
      const [q, r] = cells[i];
      this.wq[i] = q; this.wr[i] = r; this.wRad[i] = q + r;
      byQR[(r + R) * W + (q + R)] = i;
    }
    this.cnt = new Int32Array(R + 2);   // cnt[x] = 半径 ≤ x のセル数
    for (let i = 0; i < NW; i++) this.cnt[this.wRad[i] + 1]++;
    for (let x = 1; x <= R + 1; x++) this.cnt[x] += this.cnt[x - 1];

    // 横近傍テーブル (折り返し込み)。半径 R のセルの外側参照のみ -1
    this.nbrH = new Int32Array(NW * 6).fill(-1);
    for (let i = 0; i < NW; i++) {
      for (let m = 0; m < 6; m++) {
        const q = this.wq[i] + DIRS[m][0], r = this.wr[i] + DIRS[m][1];
        if ((Math.abs(q) + Math.abs(r) + Math.abs(q + r)) / 2 > R) continue;
        const [cq, cr] = canon(q, r);
        this.nbrH[i * 6 + m] = byQR[(cr + R) * W + (cq + R)];
      }
    }
    // 逆隣接 (i が付着したとき nn を増やすべきセル、重複度込み)
    const cntRev = new Int32Array(NW);
    for (let e = 0; e < NW * 6; e++) if (this.nbrH[e] >= 0) cntRev[this.nbrH[e]]++;
    this.revOff = new Int32Array(NW + 1);
    for (let i = 0; i < NW; i++) this.revOff[i + 1] = this.revOff[i] + cntRev[i];
    this.revDat = new Int32Array(this.revOff[NW]);
    const fill = this.revOff.slice(0, NW);
    for (let i = 0; i < NW; i++) {
      for (let m = 0; m < 6; m++) {
        const j = this.nbrH[i * 6 + m];
        if (j >= 0) this.revDat[fill[j]++] = i;
      }
    }
    this.byQR = byQR;
  }

  // デバッグ/描画補助: 任意の格子座標の状態 (対称折り返し込み)
  wedgeIndexOf(q, r) {
    if ((Math.abs(q) + Math.abs(r) + Math.abs(q + r)) / 2 > this.R) return -1;
    const [cq, cr] = canon(q, r);
    return this.byQR[(cr + this.R) * this.W + (cq + this.R)];
  }
  cellAt(q, r, k) {
    const wi = this.wedgeIndexOf(q, r);
    return wi < 0 ? 0 : this.a[k * this.NW + wi];
  }
  fieldAt(q, r, k) {
    const wi = this.wedgeIndexOf(q, r);
    return wi < 0 ? 0 : this.d[k * this.NW + wi];
  }
  bAt(q, r, k) {
    const wi = this.wedgeIndexOf(q, r);
    return wi < 0 ? 0 : this.b[k * this.NW + wi];
  }

  reset() {
    this.a.fill(0); this.nn.fill(0); this.nnH.fill(0); this.nnV.fill(0);
    this.inB.fill(0); this.bornStep.fill(0);
    this.d.fill(this.params.rho); this.d2.fill(this.params.rho);
    this.b.fill(0); this.c.fill(0);
    this.bCount = 0;
    this.step_n = 0;
    this.attached = 0;       // 実結晶のセル数 (重複度込み概算は使わず wedge 数)
    this.rMaxAtt = 0;
    this.kMinAtt = this.kc; this.kMaxAtt = this.kc;
    this.edge = false;
    // 種結晶: 中央 3 層 (板にも最低限の厚みを与える)
    this.attach(this.kc * this.NW + 0);
    this.attach((this.kc - 1) * this.NW + 0);
    this.attach((this.kc + 1) * this.NW + 0);
  }

  setParams(p) { Object.assign(this.params, p); }

  attach(i) {
    if (this.a[i]) return;
    const NW = this.NW;
    const wi = i % NW, k = (i - wi) / NW;
    this.a[i] = 1;
    this.c[i] += this.b[i] + this.d[i];
    this.b[i] = 0; this.d[i] = 0; this.d2[i] = 0;
    const qq = this.wq[wi], rr = this.wr[wi];
    this.attached += (qq === 0 && rr === 0) ? 1 : ((rr === 0 || qq === rr) ? 6 : 12);
    this.bornStep[i] = this.step_n;
    const rad = this.wRad[wi];
    if (rad > this.rMaxAtt) this.rMaxAtt = rad;
    if (k < this.kMinAtt) this.kMinAtt = k;
    if (k > this.kMaxAtt) this.kMaxAtt = k;
    // 横方向の上限のみ全停止。縦は付着側で抑制する (冠角柱などのため
    // 柱が伸び切っても横方向の成長は続けられる)
    if (this.rMaxAtt >= this.maxRad) this.edge = true;
    // 逆隣接: この付着で近傍数が増えるセル (折り返し重複度込み)
    const base = k * NW;
    for (let e = this.revOff[wi]; e < this.revOff[wi + 1]; e++) {
      const j = base + this.revDat[e];
      if (!this.a[j]) {
        this.nnH[j]++; this.nn[j]++;
        if (!this.inB[j]) { this.inB[j] = 1; this.pushB(j); }
      }
    }
    if (k + 1 < this.H) {
      const j = i + NW;
      if (!this.a[j]) { this.nnV[j]++; this.nn[j]++; if (!this.inB[j]) { this.inB[j] = 1; this.pushB(j); } }
    }
    if (k - 1 >= 0) {
      const j = i - NW;
      if (!this.a[j]) { this.nnV[j]++; this.nn[j]++; if (!this.inB[j]) { this.inB[j] = 1; this.pushB(j); } }
    }
  }

  pushB(j) {
    if (this.bCount >= this.bList.length) {
      const nl = new Int32Array(this.bList.length * 2);
      nl.set(this.bList); this.bList = nl;
    }
    this.bList[this.bCount++] = j;
  }

  // 近傍構成から付着閾値 (すべて β に比例)。
  // kinkF が枝分かれの主要ノブ: 面上の b は飽和値 b* = J/μ までしか溜まらない
  // ので、キンク閾値 kinkF·β が面中央の b* より上なら列の充填が面中央で
  // 恒久停止し (ファセット不安定性)、角だけが伸びて枝になる。
  threshold(nH, nV) {
    const p = this.params;
    const kf = p.kinkF ?? 0.8;
    const hf = p.hollowF ?? 0.65;   // 角 (横+縦) サイトの係数: 高いと柱の端が中空化
    const n = nH + nV;
    if (n >= 4) return 0.15 * Math.min(p.betaH, p.betaV);
    if (nV === 0) return nH >= 3 ? kf * p.betaH : p.betaH;
    if (nH === 0) return nV >= 2 ? kf * p.betaV : p.betaV;
    return hf * Math.min(p.betaH, p.betaV);
  }

  step() {
    const NW = this.NW;
    const p = this.params;
    const a = this.a, nn = this.nn, b = this.b, c = this.c;
    const nbrH = this.nbrH, cnt = this.cnt;
    let d = this.d, d2 = this.d2;

    const lam = p.lam ?? 8e-6;
    // 補給マージン: 結晶サイズに比例 (枯渇ハローは結晶と同程度に広がる)
    const S = Math.max(this.rMaxAtt, (this.kMaxAtt - this.kMinAtt) >> 1);
    const margin = Math.max(24, Math.min(Math.round(this.R * 0.55), Math.round(S * 1.2)));
    const AR = Math.min(this.rMaxAtt + margin, this.R - 2);
    const kLo = Math.max(this.kMinAtt - margin, 1);
    const kHi = Math.min(this.kMaxAtt + margin, this.H - 2);
    const w = p.w, w9 = w / 9, omw = 1 - w + w9 - lam;
    const lamRho = lam * p.rho;
    const nDiff = p.nDiff ?? 3;
    const nC = cnt[AR + 1];   // 半径 ≤ AR のくさびセル数

    // --- 1. 拡散 (サブステップ ×nDiff, ブランチレス反射境界) ---
    for (let sw = 0; sw < nDiff; sw++) {
      const src = sw % 2 === 0 ? d : d2;
      const dst = sw % 2 === 0 ? d2 : d;
      for (let k = kLo; k <= kHi; k++) {
        const base = k * NW;
        for (let t = 0; t < nC; t++) {
          const i = base + t;
          if (a[i]) continue;   // d は常に 0 を維持
          const di = src[i];
          const t6 = t * 6;
          // 付着近傍は反射 (= 自セル値): nn[i] 個ぶん di を足す
          const s = src[base + nbrH[t6]] + src[base + nbrH[t6 + 1]] +
                    src[base + nbrH[t6 + 2]] + src[base + nbrH[t6 + 3]] +
                    src[base + nbrH[t6 + 4]] + src[base + nbrH[t6 + 5]] +
                    src[i + NW] + src[i - NW] + nn[i] * di;
          dst[i] = omw * di + w9 * s + lamRho;
        }
      }
    }
    if (nDiff % 2 === 1) { this.d = d2; this.d2 = d; }
    const dd = this.d;

    // --- 2. 凍結 / 3. 付着 / 4. 融解 (境界セルのみ) ---
    const kap = p.kappa, omk = 1 - p.kappa, mu = p.mu, gam = p.gamma;
    const toAttach = [];
    let ncNew = 0;
    const bl = this.bList, bn = this.bCount;
    // 縦方向の成長限界 (この外には付着させない)
    const iTop = NW * (this.kc + this.maxKHalf + 1);
    const iBot = NW * (this.kc - this.maxKHalf);
    for (let t = 0; t < bn; t++) {
      const i = bl[t];
      if (a[i]) { this.inB[i] = 0; continue; }
      const di = dd[i];
      if (di > 0) { b[i] += omk * di; c[i] += kap * di; dd[i] = 0; }
      if (b[i] >= this.threshold(this.nnH[i], this.nnV[i]) && i < iTop && i >= iBot) {
        toAttach.push(i);
      } else {
        dd[i] += mu * b[i] + gam * c[i];
        b[i] -= mu * b[i];
        c[i] -= gam * c[i];
      }
      bl[ncNew++] = i;
    }
    this.bCount = ncNew;
    for (let t = 0; t < toAttach.length; t++) this.attach(toAttach[t]);

    // --- 5. 遠方場: 計算領域の外縁を ρ に保つ (ディリクレ境界) ---
    const rho = p.rho;
    const d2b = this.d2;
    const ringTo = cnt[Math.min(AR + 2, this.R) + 1];   // 半径 ≤ AR+2 のセル数
    for (let k = Math.max(kLo - 2, 0); k <= Math.min(kHi + 2, this.H - 1); k++) {
      const base = k * NW;
      // 上下キャップ層は全域、中間層は外縁リング (AR, AR+2] のみ
      const from = (k < kLo || k > kHi) ? 0 : nC;
      for (let t = from; t < ringTo; t++) {
        const i = base + t;
        if (!a[i]) { dd[i] = rho; d2b[i] = rho; }
      }
    }

    this.step_n++;
  }

  // ---- 計測 ----
  measure() {
    const thickness = this.kMaxAtt - this.kMinAtt + 1;
    const diameter = 2 * this.rMaxAtt + 1;
    // 0° (r=0 行) と 30° (q=r 対角) のカルテシアン到達半径
    const kBase = this.kc * this.NW;
    let r0 = 0, r30 = 0;
    for (let q = 0; q <= Math.min(this.rMaxAtt, this.R); q++) {
      const wi = this.byQR[(0 + this.R) * this.W + (q + this.R)];
      if (wi >= 0 && this.a[kBase + wi]) r0 = q;
    }
    for (let q = 0; 2 * q <= Math.min(this.rMaxAtt, this.R); q++) {
      const wi = this.byQR[(q + this.R) * this.W + (q + this.R)];
      if (wi >= 0 && this.a[kBase + wi]) r30 = q * Math.sqrt(3);
    }
    const armRatio = (r0 > 2 && r30 > 2) ? Math.max(r0, r30) / Math.min(r0, r30) : 1;
    // 実セル数 (折り返し重複度を戻す): 概算 12×wedge − 補正は省略し wedge 数を返す
    return { diameter, thickness, aspect: diameter / thickness, armRatio,
             attached: this.attached, step: this.step_n };
  }

  // ---- メッシュ抽出: 露出面のみ六角柱の面を生成 (12 像に展開) ----
  extractMesh(hz = 1.0) {
    const NW = this.NW, a = this.a, nbrH = this.nbrH;
    const kLo = Math.max(this.kMinAtt - 1, 1), kHi = Math.min(this.kMaxAtt + 1, this.H - 2);
    const nC = this.cnt[Math.min(this.rMaxAtt + 1, this.R) + 1];

    // 六角コーナー (外接半径 1/√3, 角度 -30°+60°m) と面法線 (角度 60°m)
    const RC = 1 / Math.sqrt(3);
    const cx = [], cy = [], fnx = [], fny = [];
    for (let m = 0; m < 6; m++) {
      const thc = (-30 + 60 * m) * Math.PI / 180;
      cx.push(RC * Math.cos(thc)); cy.push(RC * Math.sin(thc));
      const thn = 60 * m * Math.PI / 180;
      fnx.push(Math.cos(thn)); fny.push(Math.sin(thn));
    }
    // 12 の対称像: 回転 6 + 鏡映 6 (鏡映は巻き順反転)
    const xforms = [];
    for (let g = 0; g < 6; g++) {
      const th = g * Math.PI / 3, cth = Math.cos(th), sth = Math.sin(th);
      xforms.push({ m00: cth, m01: -sth, m10: sth, m11: cth, flip: false });
    }
    for (let g = 0; g < 6; g++) {
      const th = g * Math.PI / 3, cth = Math.cos(th), sth = Math.sin(th);
      // R(θ)·Mirror(x軸): (x,y) → (c·x + s·y, s·x − c·y)
      xforms.push({ m00: cth, m01: sth, m10: sth, m11: -cth, flip: true });
    }

    // パス 1: セルごとの露出面数と像数を数える
    let latFaces = 0, capFaces = 0;
    for (let k = kLo; k <= kHi; k++) {
      const base = k * NW;
      for (let t = 0; t < nC; t++) {
        const i = base + t;
        if (!a[i]) continue;
        const q = this.wq[t], r = this.wr[t];
        const nImg = (q === 0 && r === 0) ? 1 : ((r === 0 || q === r) ? 6 : 12);
        const t6 = t * 6;
        let lf = 0;
        for (let m = 0; m < 6; m++) { const j = nbrH[t6 + m]; if (j < 0 || !a[base + j]) lf++; }
        let cf = 0;
        if (!a[i + NW]) cf++;
        if (!a[i - NW]) cf++;
        latFaces += lf * nImg; capFaces += cf * nImg;
      }
    }

    const nVerts = latFaces * 4 + capFaces * 6;
    const pos = new Float32Array(nVerts * 3);
    const nrm = new Float32Array(nVerts * 3);
    const ageArr = new Float32Array(nVerts);   // 生の付着ステップ (born)
    const index = new Uint32Array(latFaces * 6 + capFaces * 12);
    let vp = 0, ip = 0, vBase = 0, ap = 0;
    const kcF = this.kc;

    for (let k = kLo; k <= kHi; k++) {
      const base = k * NW;
      const zb = (k - kcF) * hz - 0.5 * hz, zt = zb + hz;
      for (let t = 0; t < nC; t++) {
        const i = base + t;
        if (!a[i]) continue;
        const q = this.wq[t], r = this.wr[t];
        const x0 = q + 0.5 * r, y0 = 0.8660254 * r;
        const onSeam = (r === 0 || q === r);
        const isCenter = (q === 0 && r === 0);
        const t6 = t * 6;
        const age = this.bornStep[i];   // 付着ステップ。シェーダ側で現在ステップと比較
        for (let g = 0; g < 12; g++) {
          if (isCenter && g > 0) break;
          if (onSeam && g >= 6) break;
          const X = xforms[g];
          // 側面
          for (let m = 0; m < 6; m++) {
            const j = nbrH[t6 + m];
            if (j >= 0 && a[base + j]) continue;
            const m2 = (m + 1) % 6;
            const ax0 = x0 + cx[m], ay0 = y0 + cy[m];
            const bx0 = x0 + cx[m2], by0 = y0 + cy[m2];
            const ax = X.m00 * ax0 + X.m01 * ay0, ay = X.m10 * ax0 + X.m11 * ay0;
            const bx = X.m00 * bx0 + X.m01 * by0, by = X.m10 * bx0 + X.m11 * by0;
            const nxv = X.m00 * fnx[m] + X.m01 * fny[m], nyv = X.m10 * fnx[m] + X.m11 * fny[m];
            pos[vp] = ax; pos[vp + 1] = ay; pos[vp + 2] = zb;
            pos[vp + 3] = bx; pos[vp + 4] = by; pos[vp + 5] = zb;
            pos[vp + 6] = bx; pos[vp + 7] = by; pos[vp + 8] = zt;
            pos[vp + 9] = ax; pos[vp + 10] = ay; pos[vp + 11] = zt;
            for (let v = 0; v < 4; v++) { nrm[vp + v * 3] = nxv; nrm[vp + v * 3 + 1] = nyv; nrm[vp + v * 3 + 2] = 0; }
            vp += 12;
            ageArr[ap] = age; ageArr[ap + 1] = age; ageArr[ap + 2] = age; ageArr[ap + 3] = age;
            ap += 4;
            if (X.flip) {
              index[ip++] = vBase; index[ip++] = vBase + 2; index[ip++] = vBase + 1;
              index[ip++] = vBase; index[ip++] = vBase + 3; index[ip++] = vBase + 2;
            } else {
              index[ip++] = vBase; index[ip++] = vBase + 1; index[ip++] = vBase + 2;
              index[ip++] = vBase; index[ip++] = vBase + 2; index[ip++] = vBase + 3;
            }
            vBase += 4;
          }
          // 上面 / 下面
          for (let ud = 0; ud < 2; ud++) {
            const up = ud === 0;
            if (a[i + (up ? NW : -NW)]) continue;
            const z = up ? zt : zb, nz = up ? 1 : -1;
            for (let m = 0; m < 6; m++) {
              const px0 = x0 + cx[m], py0 = y0 + cy[m];
              pos[vp] = X.m00 * px0 + X.m01 * py0;
              pos[vp + 1] = X.m10 * px0 + X.m11 * py0;
              pos[vp + 2] = z;
              nrm[vp] = 0; nrm[vp + 1] = 0; nrm[vp + 2] = nz;
              vp += 3;
              ageArr[ap++] = age;
            }
            const ccw = up !== X.flip;  // 上面かつ非鏡映 → CCW
            for (let tt = 1; tt < 5; tt++) {
              if (ccw) { index[ip++] = vBase; index[ip++] = vBase + tt; index[ip++] = vBase + tt + 1; }
              else { index[ip++] = vBase; index[ip++] = vBase + tt + 1; index[ip++] = vBase + tt; }
            }
            vBase += 6;
          }
        }
      }
    }
    return { positions: pos, normals: nrm, born: ageArr, indices: index };
  }
}

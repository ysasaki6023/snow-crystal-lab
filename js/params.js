// 物理パラメータ (気温・湿度・気圧・水分量) → 格子モデルパラメータへの変換
//
// 第一原理側の根拠:
//  - 飽和水蒸気圧: Magnus 式 (氷面 / 水面)
//  - 過剰水蒸気密度 (g/m^3) = (RH/100 - 1) × 氷面飽和水蒸気密度、利用可能水分量で頭打ち
//  - 晶癖 (板状 ⇔ 柱状) の温度依存: 中谷ダイヤグラムに基づく異方性曲線 A(T)
//    → 横方向 (プリズム面) と縦方向 (基底面) の付着閾値の比に変換
//  - 気圧: 水蒸気拡散係数 D ∝ 1/P → 拡散混合率 w に変換
//    (高圧 = 拡散律速が強い = 樹枝化しやすい)

export function esIce(Tc) {   // 氷面飽和水蒸気圧 [hPa] (Magnus)
  return 6.112 * Math.exp(22.46 * Tc / (272.62 + Tc));
}
export function esWater(Tc) { // 過冷却水面飽和水蒸気圧 [hPa] (Magnus)
  return 6.112 * Math.exp(17.62 * Tc / (243.12 + Tc));
}
function vaporDensity(es_hPa, Tc) { // 理想気体: ρ = e·M/(R·T) → [g/m^3]
  return es_hPa * 100 * 0.018015 / (8.314 * (Tc + 273.15)) * 1000;
}
export function rhoSatIce(Tc)   { return vaporDensity(esIce(Tc), Tc); }
export function rhoSatWater(Tc) { return vaporDensity(esWater(Tc), Tc); }
// 水飽和線: 過冷却水滴と共存するときの対氷過剰水蒸気密度 [g/m^3]
export function waterSatExcess(Tc) {
  return Math.max(0, rhoSatWater(Tc) - rhoSatIce(Tc));
}

// 晶癖の異方性 A(T): >1 で板状 (横成長優位)、<1 で柱状 (縦成長優位)
// 中谷ダイヤグラム: -2℃板 / -5℃針・柱 / -15℃板・樹枝 / -30℃以下 柱
const AN_T = [0, -2, -4, -6, -9, -12, -15, -18, -22, -26, -32];
const AN_V = [1.3, 2.0, 0.50, 0.32, 0.65, 1.9, 3.0, 2.3, 1.1, 0.70, 0.55];
export function anisotropy(Tc) {
  if (Tc >= AN_T[0]) return AN_V[0];
  const n = AN_T.length;
  if (Tc <= AN_T[n - 1]) return AN_V[n - 1];
  for (let i = 0; i < n - 1; i++) {
    if (Tc <= AN_T[i] && Tc > AN_T[i + 1]) {
      const t = (Tc - AN_T[i]) / (AN_T[i + 1] - AN_T[i]);
      // 滑らかに (cos 補間)
      const s = 0.5 - 0.5 * Math.cos(Math.PI * t);
      return AN_V[i] + s * (AN_V[i + 1] - AN_V[i]);
    }
  }
  return 1;
}

function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

// チューニング定数 (node テストで較正)
export const TUNE = {
  // 付着の「等価蒸気密度」d*: 界面の d がこれを超え続けるとその面が成長する。
  // b の飽和値 b* = (1-κ)d/μ ≥ β ⟺ d ≥ βμ/(1-κ) ≡ d* なので β = d*/μ で設定する。
  // 枝分かれの条件: 面中央の d < d*H < 先端の d。なので d*H は温度によらず
  // この帯域に留め (aPowH 小)、晶癖の異方性は主に縦 (基底面) 側で効かせる。
  dStar0: 0.20,     // d* の基準値
  aPowH: 0.25,      // 横方向への異方性の効き (弱く)
  aPowV: 1.00,      // 縦方向への異方性の効き (強く)
  dStarMin: 0.05, dStarMax: 0.30,
  rho0: 0.34, rhoSlope: 1.20, rhoMin: 0.30, rhoMax: 0.78, // 過剰蒸気→格子蒸気密度
  kinkF0: 0.78,     // キンク (凹角) 閾値係数の基準。過飽和で上がる → 羊歯状の側枝
  kinkSlope: 0.40, kinkMax: 0.90,
  hollowF: 0.65,    // 角 (横+縦) サイトの閾値係数
  kap0: 0.003, kapSlope: 0.03,  // 凍結率 κ
  mu: 0.010,        // 境界準液層の融解率 (面の b 飽和を決め、面成長を止めるゲート)
  gamma: 0.0002,    // 結晶質量の昇華率
  nDiff: 3,         // 拡散サブステップ数
  wBase: 0.74, wMin: 0.35,      // 気圧→拡散混合率
  // 雲水補給率 λ: 実際の雲では遮蔽長 (数cm) ≫ 結晶 (数mm) なので
  // 結晶スケールでは純粋な遠方場拡散。ごく弱い値に固定して場を安定させるだけ
  lam: 8e-6,
};

// メインの変換。Tc[℃], RH[%対氷], P[hPa], W[g/m^3 利用可能水分量]
export function modelParams(Tc, RH, P, W) {
  const sat = rhoSatIce(Tc);
  const excess = Math.max(0, (RH / 100 - 1) * sat); // 対氷過剰水蒸気密度
  const eff = Math.min(excess, W);                  // 水分量で頭打ち
  const sigma = eff / sat;                          // 実効過飽和度
  const A = anisotropy(Tc);
  const t = TUNE;
  // 低過飽和では成長が平衡形に近づき晶癖が穏やかになる (σ 依存の異方性)
  const anisoScale = clamp(0.45 + 2.2 * eff, 0.5, 1.15);
  const dStarH = clamp(t.dStar0 / Math.pow(A, t.aPowH * anisoScale), t.dStarMin, t.dStarMax);
  const dStarV = clamp(t.dStar0 * Math.pow(A, t.aPowV * anisoScale), t.dStarMin, t.dStarMax);
  return {
    rho:   clamp(t.rho0 + t.rhoSlope * eff, t.rhoMin, t.rhoMax),
    betaH: dStarH / t.mu,
    betaV: dStarV / t.mu,
    kappa: t.kap0 + t.kapSlope * sigma,
    mu: t.mu,
    gamma: t.gamma,
    w: clamp(t.wBase * 1013 / P, t.wMin, 1),
    nDiff: t.nDiff,
    lam: t.lam,
    kinkF: clamp(t.kinkF0 + t.kinkSlope * Math.min(1, eff / 0.30), t.kinkF0, t.kinkMax),
    hollowF: t.hollowF,
    eff, sigma, A,
  };
}

// 予想される結晶形態のラベル (中谷ダイヤグラム)
export function morphologyLabel(Tc, RH, W) {
  const sat = rhoSatIce(Tc);
  const eff = Math.min(Math.max(0, (RH / 100 - 1) * sat), W);
  const ws = waterSatExcess(Tc);
  const r = ws > 0 ? eff / ws : 0;
  if (Tc > -3.5) {
    if (r > 1.0) return '樹枝状の薄板';
    return r > 0.45 ? '薄い六角板' : '厚い六角板';
  }
  if (Tc > -10) {
    if (r > 1.0) return '針状結晶';
    return r > 0.5 ? '中空角柱 (鞘状)' : '角柱';
  }
  if (Tc > -22) {
    if (r > 1.0) return (Tc < -11 && Tc > -18) ? '星型樹枝状結晶' : '羊歯状樹枝';
    if (r > 0.6) return '扇形板・広幅板';
    if (r > 0.3) return '六角板';
    return '厚板';
  }
  return r > 0.8 ? '砲弾型・柱の集合' : '柱・厚板';
}

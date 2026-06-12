// Gravner-Griffeath 2D オリジナルモデルの忠実な再現 (枝分かれ要因の特定用)
// 規則: 拡散(完全平均,反射) → 凍結 b+=(1-κ)d → 付着(n≤2:β / n=3:b≥1 or (b≥α & Σd<θ) / n≥4:即) → 融解
const L = 240; // 半径
const W = 2 * L + 1, N = W * W;
const [rho, beta, alpha, theta, kappa, mu, gamma, STEPS] =
  process.argv.slice(2).map(Number);

const a = new Uint8Array(N), d = new Float64Array(N).fill(rho),
      d2 = new Float64Array(N), b = new Float64Array(N), c = new Float64Array(N);
const offs = [1, -1, W, -W, W - 1, 1 - W]; // 三角格子近傍 (axial)
const idx = (q, r) => (r + L) * W + (q + L);
const hd = (q, r) => (Math.abs(q) + Math.abs(r) + Math.abs(q + r)) / 2;
a[idx(0, 0)] = 1; d[idx(0, 0)] = 0;

let rMax = 1;
for (let s = 0; s < STEPS; s++) {
  const AR = Math.min(rMax + Math.max(30, rMax), L - 2);
  // 拡散
  for (let r = -AR; r <= AR; r++) {
    const qLo = Math.max(-AR, -r - AR), qHi = Math.min(AR, -r + AR);
    for (let q = qLo; q <= qHi; q++) {
      const i = idx(q, r);
      if (a[i]) { d2[i] = 0; continue; }
      const di = d[i];
      let sum = di;
      for (const o of offs) { const j = i + o; sum += a[j] ? di : d[j]; }
      d2[i] = sum / 7;
    }
  }
  for (let r = -AR; r <= AR; r++) { // swap (リング外は ρ 維持)
    const qLo = Math.max(-AR, -r - AR), qHi = Math.min(AR, -r + AR);
    for (let q = qLo; q <= qHi; q++) { const i = idx(q, r); d[i] = d2[i]; }
  }
  // 凍結+付着+融解 (境界セル)
  const att = [];
  for (let r = -AR; r <= AR; r++) {
    const qLo = Math.max(-AR, -r - AR), qHi = Math.min(AR, -r + AR);
    for (let q = qLo; q <= qHi; q++) {
      const i = idx(q, r);
      if (a[i]) continue;
      let n = 0;
      for (const o of offs) if (a[i + o]) n++;
      if (n === 0) continue;
      // 凍結
      b[i] += (1 - kappa) * d[i]; c[i] += kappa * d[i]; d[i] = 0;
      // 付着判定
      let ok = false;
      if (n <= 2) ok = b[i] >= beta;
      else if (n === 3) {
        if (b[i] >= 1) ok = true;
        else {
          let sd = d[i];
          for (const o of offs) sd += d[i + o];
          ok = (b[i] >= alpha && sd < theta);
        }
      } else ok = true;
      if (ok) att.push(i, q, r);
      else { d[i] += mu * b[i] + gamma * c[i]; b[i] *= 1 - mu; c[i] *= 1 - gamma; }
    }
  }
  for (let t = 0; t < att.length; t += 3) {
    const i = att[t];
    a[i] = 1; c[i] += b[i] + d[i]; b[i] = 0; d[i] = 0;
    const h = hd(att[t + 1], att[t + 2]);
    if (h > rMax) rMax = h;
  }
  if (rMax >= L - 40) { console.log(`edge at step ${s}`); break; }
}

// 表示
let count = 0;
for (let i = 0; i < N; i++) count += a[i];
console.log(`rMax=${rMax} cells=${count}`);
const span = Math.min(rMax + 2, L);
const out = [];
for (let y = -span; y <= span; y += 2) {
  let line = '';
  for (let x = -span; x <= span; x++) {
    const r = Math.round(y / 0.8660254), q = Math.round(x - 0.5 * r);
    line += hd(q, r) > L ? ' ' : (a[idx(q, r)] ? '#' : '.');
  }
  out.push(line);
}
console.log(out.join('\n'));

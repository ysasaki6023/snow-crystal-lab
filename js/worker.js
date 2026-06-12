// シミュレーションワーカー: CA をメインスレッドから隔離して実行
import { SnowSim } from './sim-core.js';

let sim = null;
let running = false;
let speed = 6;            // 1..10 → 1 tick あたりのステップ数 ≈ speed²
let lastMeshAt = 0;
let lastStatsAt = 0;
let stepTimes = [];

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

function sendStats() {
  if (!sim) return;
  const now = performance.now();
  let sps = 0;
  stepTimes = stepTimes.filter(t => now - t.at < 2000);
  if (stepTimes.length > 1) {
    const n = stepTimes.reduce((s, t) => s + t.n, 0);
    sps = n / ((now - stepTimes[0].at) / 1000 + 1e-9);
  }
  post({
    type: 'stats',
    step: sim.step_n,
    attached: sim.attached,
    rMax: sim.rMaxAtt,
    kSpan: sim.kMaxAtt - sim.kMinAtt + 1,
    edge: sim.edge,
    sps: Math.round(sps),
  });
}

function sendMesh() {
  if (!sim) return;
  const { positions, normals, born, indices } = sim.extractMesh(1.0);
  post({ type: 'mesh', positions, normals, born, indices, step: sim.step_n },
       [positions.buffer, normals.buffer, born.buffer, indices.buffer]);
  lastMeshAt = performance.now();
}

function loop() {
  if (!running || !sim) return;
  const t0 = performance.now();
  const target = Math.max(1, Math.round(speed * speed * 0.8));
  let n = 0;
  // 1 tick の計算は最大 30ms (メッセージ処理を詰まらせない)
  while (n < target && performance.now() - t0 < 30 && !sim.edge) {
    sim.step(); n++;
  }
  if (n > 0) stepTimes.push({ at: performance.now(), n });

  const now = performance.now();
  if (now - lastStatsAt > 120) { sendStats(); lastStatsAt = now; }
  // メッシュは高頻度で送り「計算する端から」見えるように
  const meshInterval = sim.attached > 250000 ? 300 : (sim.attached > 80000 ? 170 : 90);
  if (now - lastMeshAt > meshInterval) sendMesh();

  if (sim.edge) {
    running = false;
    sendStats(); sendMesh();
    post({ type: 'edge' });
    return;
  }
  setTimeout(loop, 0);
}

self.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'init':
      sim = new SnowSim(m.R, m.H, m.params);
      post({ type: 'ready' });
      sendStats(); sendMesh();
      break;
    case 'setParams':
      if (sim) sim.setParams(m.params);
      break;
    case 'run':
      if (!running && sim && !sim.edge) { running = true; loop(); }
      break;
    case 'pause':
      running = false;
      sendStats();
      break;
    case 'reset':
      if (sim) {
        if (m.params) sim.setParams(m.params);
        sim.reset();
        sendStats(); sendMesh();
        if (m.autorun) { running = true; loop(); }
      }
      break;
    case 'speed':
      speed = m.value;
      break;
    case 'meshNow':
      sendMesh();
      break;
  }
};

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { modelParams, morphologyLabel, waterSatExcess, rhoSatIce } from './params.js';
import { GpuSim } from './gpu-sim.js';

// ---------- 定数 ----------
//
// DESIGN NOTES (2026-06-13 WebGPU 高解像度化):
//   シミュレーションは 2 系統:
//     - WebGPU エンジン (gpu-sim.js): 高解像度 (R=384〜640)。メッシュ抽出は
//       mesh-worker.js が 60° セクター + 継ぎ目の 2 メッシュを返し、
//       セクターを 6 インスタンスで描画する (頂点メモリ 1/6)。
//     - CPU フォールバック (worker.js + sim-core.js): 従来のまま一切変更なし。
//       WebGPU 不可 (旧 Safari / Firefox 等) でも従来体験を維持する。
//   将来 LLM が「コード重複なので CPU 経路も mesh-worker に統一」と提案しても
//   戻すな — CPU 経路の独立性がフォールバックの信頼性そのもの。
const IS_MOBILE = matchMedia('(pointer: coarse)').matches || innerWidth < 760;
const CELL_UM = 8;                 // 1 セル ≈ 8 µm (表示用スケール)
const CPU_RES = IS_MOBILE ? { R: 156, H: 132 } : { R: 236, H: 180 };
// WebGPU 解像度プリセット (メモリ: ~24B/cell → ultra desktop ≈ 510MB GPU)
const GPU_RES = {
  high:  IS_MOBILE ? { R: 256, H: 160 } : { R: 384, H: 256 },
  ultra: IS_MOBILE ? { R: 320, H: 192 } : { R: 512, H: 320 },
  max:   IS_MOBILE ? { R: 384, H: 224 } : { R: 640, H: 352 },
};
let resChoice = localStorage.getItem('snowlab-res') || 'ultra';
if (!GPU_RES[resChoice]) resChoice = 'ultra';

const ui = {
  T: -15, RH: 140, P: 1013, W: 0.30, speed: 6,
};

// ---------- Three.js シーン ----------
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, IS_MOBILE ? 1.6 : 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.02;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
camera.position.set(8, 56, 54);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.9;
controls.minDistance = 18;
controls.maxDistance = 300;
controls.enablePan = false;   // 結晶を常に中央に (タッチ操作でも迷子にならない)

// 環境光 (部屋環境を PMREM で)
{
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
}

// 背景: 夜空のグラデーション球
{
  const geo = new THREE.SphereGeometry(480, 32, 24);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {},
    vertexShader: `
      varying vec3 vP;
      void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `
      varying vec3 vP;
      void main(){
        float h = normalize(vP).y * 0.5 + 0.5;
        vec3 bottom = vec3(0.008, 0.015, 0.045);
        vec3 mid    = vec3(0.020, 0.050, 0.120);
        vec3 top    = vec3(0.005, 0.010, 0.030);
        vec3 c = mix(bottom, mid, smoothstep(0.0, 0.55, h));
        c = mix(c, top, smoothstep(0.55, 1.0, h));
        // ほのかな光芒
        float glow = pow(max(0.0, 1.0 - length(normalize(vP).xy - vec2(0.0, 0.12)) * 1.4), 2.0);
        c += vec3(0.012, 0.030, 0.060) * glow;
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
  scene.add(new THREE.Mesh(geo, mat));
}

// ライティング
{
  const key = new THREE.DirectionalLight(0xeaf4ff, 1.1);
  key.position.set(60, 95, 40);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x77aaff, 2.2);
  rim.position.set(-70, -35, -80);
  scene.add(rim);
  const rim2 = new THREE.DirectionalLight(0xa9c8ff, 1.4);
  rim2.position.set(80, -20, -60);
  scene.add(rim2);
  const warm = new THREE.PointLight(0xffd9b0, 60, 0, 1.8);
  warm.position.set(-45, 30, 65);
  scene.add(warm);
  // ライトボックス: 下からの透過光 (雪結晶撮影の定番)
  const under = new THREE.DirectionalLight(0xbfdcff, 0.45);
  under.position.set(0, -100, 25);
  scene.add(under);
}

// カメラ背後の発光ハロー (透過マテリアルを背後から照らす)
{
  const cv = document.createElement('canvas');
  cv.width = cv.height = 256;
  const cx = cv.getContext('2d');
  const grad = cx.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, 'rgba(190,225,255,0.85)');
  grad.addColorStop(0.35, 'rgba(120,180,255,0.30)');
  grad.addColorStop(1, 'rgba(80,140,255,0)');
  cx.fillStyle = grad; cx.fillRect(0, 0, 256, 256);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(cv), transparent: true, opacity: 0.10,
    depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  halo.position.set(0, 0, -300);
  halo.scale.setScalar(520);
  camera.add(halo);
  scene.add(camera);
}

// 雪の微粒子 (浮遊感)
let dust;
{
  const n = IS_MOBILE ? 280 : 600;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 70 + Math.random() * 160;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
    pos[i * 3 + 1] = r * Math.cos(ph) * 0.7;
    pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  // やわらかい点スプライト
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const cx = cv.getContext('2d');
  const grad = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,0.9)');
  grad.addColorStop(0.4, 'rgba(210,235,255,0.35)');
  grad.addColorStop(1, 'rgba(210,235,255,0)');
  cx.fillStyle = grad; cx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(cv);
  dust = new THREE.Points(g, new THREE.PointsMaterial({
    map: tex, size: 1.7, transparent: true, opacity: 0.55,
    depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
  }));
  scene.add(dust);
}

// 結晶 (雪片が水平に浮かぶ向き: 格子 z 軸 → ワールド上方向)
const crystalGroup = new THREE.Group();
crystalGroup.rotation.x = -Math.PI / 2;
scene.add(crystalGroup);

const iceMatHQ = new THREE.MeshPhysicalMaterial({
  color: 0xf4fbff,
  metalness: 0,
  roughness: 0.045,
  transmission: 0.92,
  thickness: 5.0,
  ior: 1.31,
  attenuationColor: new THREE.Color(0xaddcff),
  attenuationDistance: 42,
  clearcoat: 1.0,
  clearcoatRoughness: 0.12,
  iridescence: 0.22,
  iridescenceIOR: 1.31,
  envMapIntensity: 1.5,
  transparent: true,
});
// 標準マテリアル: 深い青の氷。コアは藍色、先端はシアン、視線縁が光る
const iceMatLite = new THREE.MeshPhysicalMaterial({
  color: 0x0a2348,
  metalness: 0.0,
  roughness: 0.16,
  transparent: true,
  opacity: 0.93,
  clearcoat: 0.45,
  clearcoatRoughness: 0.22,
  iridescence: 0.2,
  iridescenceIOR: 1.32,
  envMapIntensity: 0.35,
});
// エイジンググラデーション + 成長端のフェードイン発光。
// born (付着ステップ) と現在ステップ uStep をシェーダで比較する。
// uStep は stats の間も step/s から外挿して進めるので、成長端が
// 連続的に光りながら伸びていくように見える。
const uStep = { value: 0 };
function addAgeShader(mat, opts = {}) {
  const o = Object.assign({
    oldTint: 'vec3(0.22, 0.40, 0.95)',     // コア: 深い藍
    youngTint: 'vec3(0.70, 0.95, 1.18)',   // 先端: 明るいシアン
    rimStrength: 0.38,                      // 視線縁の発光
    bodyGlow: 1.2,                          // 本体発光グラデーションの強さ
  }, opts);
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uStep = uStep;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float born;\nvarying float vBorn;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvBorn = born;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uStep;\nvarying float vBorn;')
      .replace('#include <color_fragment>', `#include <color_fragment>
        float age01 = clamp(vBorn / max(uStep, 1.0), 0.0, 1.0);   // 0=最古, 1=最新
        diffuseColor.rgb *= mix(${o.oldTint}, ${o.youngTint}, pow(age01, 3.0));
        // 生まれたてのセルはジワーッと現れる (透明 → 不透明)
        float fadeIn = smoothstep(0.0, 1.0, (uStep - vBorn) / 300.0);
        diffuseColor.a *= 0.15 + 0.85 * fadeIn;`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        // 本体の発光グラデーション (コア: 深い藍 → 先端: シアン)
        // 面積は時間の2乗で増えるので、高次で先端側に圧縮しないと全面が先端色になる
        totalEmissiveRadiance += mix(vec3(0.030, 0.085, 0.26), vec3(0.12, 0.38, 0.66),
                                     pow(age01, 3.5)) * ${o.bodyGlow.toFixed(2)};
        // 成長端の発光 (生まれて間もないセル)
        float fresh = clamp(1.0 - (uStep - vBorn) / 420.0, 0.0, 1.0);
        totalEmissiveRadiance += vec3(0.40, 0.72, 1.0) * (pow(fresh, 2.2) * 0.42);
        // 視線縁のオーロラ風リム (コアは菫色寄り、先端はシアン寄り)
        float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vViewPosition))), 2.6);
        vec3 rimCol = mix(vec3(0.38, 0.22, 0.85), vec3(0.12, 0.55, 1.0), age01);
        totalEmissiveRadiance += rimCol * (rim * ${o.rimStrength.toFixed(2)});`);
  };
}
addAgeShader(iceMatHQ, {
  oldTint: 'vec3(1.06, 1.03, 1.0)',
  youngTint: 'vec3(0.72, 0.88, 1.08)',
  rimStrength: 0.25,
  bodyGlow: 0.0,
});
addAgeShader(iceMatLite);
let hq = false;   // 既定は標準 (青い氷)。チェックで白い透過氷に

// 結晶メッシュ:
//   GPU 経路: sectorMeshes (6 インスタンス共有 geometry) + seamMesh (継ぎ目)
//   CPU 経路: seamMesh のみ使用 (12 像展開済みの単一メッシュ)
const sectorMeshes = [];
let seamMesh = null;

function currentMat() { return hq ? iceMatHQ : iceMatLite; }

function buildGeo(m) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(m.normals, 3));
  geo.setAttribute('born', new THREE.BufferAttribute(m.born, 1));
  geo.setIndex(new THREE.BufferAttribute(m.indices, 1));
  return geo;
}

function setSectorGeometry(geo) {
  if (sectorMeshes.length === 0) {
    for (let g = 0; g < 6; g++) {
      const mesh = new THREE.Mesh(geo, currentMat());
      mesh.rotation.z = g * Math.PI / 3;
      crystalGroup.add(mesh);
      sectorMeshes.push(mesh);
    }
  } else {
    sectorMeshes[0].geometry.dispose();
    for (const m of sectorMeshes) m.geometry = geo;
  }
}

function setSeamGeometry(geo) {
  if (!seamMesh) {
    seamMesh = new THREE.Mesh(geo, currentMat());
    crystalGroup.add(seamMesh);
  } else {
    seamMesh.geometry.dispose();
    seamMesh.geometry = geo;
  }
}

function clearCrystal() {
  for (const m of sectorMeshes) { crystalGroup.remove(m); }
  if (sectorMeshes.length) sectorMeshes[0].geometry.dispose();
  sectorMeshes.length = 0;
  if (seamMesh) { crystalGroup.remove(seamMesh); seamMesh.geometry.dispose(); seamMesh = null; }
  if (sparkles) { crystalGroup.remove(sparkles); sparkles.geometry.dispose(); sparkles = null; }
}

// 結晶表面のきらめき (ダイヤモンドダスト)
const uTime = { value: 0 };
const sparkMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  uniforms: { uTime },
  vertexShader: `
    attribute float phase;
    varying float vPhase;
    void main() {
      vPhase = phase;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = 130.0 / -mv.z;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    uniform float uTime;
    varying float vPhase;
    void main() {
      float d = length(gl_PointCoord - 0.5) * 2.0;
      float tw = pow(max(0.0, 0.5 + 0.5 * sin(uTime * 1.7 + vPhase * 6.2832)), 14.0);
      float a = max(0.0, 1.0 - d) * tw;
      gl_FragColor = vec4(vec3(0.9, 0.97, 1.0), a * 0.9);
    }`,
});
let sparkles = null;

// fullCircle=false (GPU セクター) のときは、サンプル点をランダムに
// 60°×k 回転させて結晶全周に散らす
function rebuildSparkles(positions, fullCircle) {
  const nv = positions.length / 3;
  const want = Math.min(450, nv);
  const stride = Math.max(1, Math.floor(nv / want));
  const n = Math.floor(nv / stride);
  const pos = new Float32Array(n * 3);
  const phase = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = i * stride * 3;
    let x = positions[s], y = positions[s + 1];
    if (!fullCircle) {
      const g = (Math.random() * 6) | 0;
      const th = g * Math.PI / 3, c = Math.cos(th), sn = Math.sin(th);
      const xr = c * x - sn * y, yr = sn * x + c * y;
      x = xr; y = yr;
    }
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = positions[s + 2];
    phase[i] = Math.random();
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('phase', new THREE.BufferAttribute(phase, 1));
  if (sparkles) {
    sparkles.geometry.dispose();
    sparkles.geometry = g;
  } else {
    sparkles = new THREE.Points(g, sparkMat);
    crystalGroup.add(sparkles);
  }
}

// ブルーム合成
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.22, 0.7, 0.88);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

// 結晶サイズに合わせたスムーズなズーム
let targetScale = 1.0;
function updateScale(rMax, kSpan) {
  const span = Math.max(rMax, kSpan * 0.72, 14);
  // 縦長画面では横幅に合わせて縮める
  const fit = 22 * Math.min(1, camera.aspect * 1.08);
  targetScale = fit / span;
}

// 最新 stats からの外挿で uStep を連続的に進める (成長端の発光が滑らかに動く)
let statStep = 0, statSps = 0, statAt = performance.now();

let tPrev = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const t = performance.now();
  const dt = Math.min(50, t - tPrev); tPrev = t;
  controls.update();
  const s = crystalGroup.scale.x + (targetScale - crystalGroup.scale.x) * (1 - Math.pow(0.998, dt));
  crystalGroup.scale.setScalar(s);
  dust.rotation.y += dt * 1.2e-5;
  dust.material.opacity = 0.45 + 0.15 * Math.sin(t * 6e-4);
  uTime.value = t / 1000;
  const est = statStep + statSps * (t - statAt) / 1000;
  uStep.value = (est < uStep.value - 500) ? est : Math.max(uStep.value, est); // リセット時のみ巻き戻し
  composer.render();
}
animate();

// ---------- 共通 UI 更新 ----------
let running = true;

function applyStats(m) {
  statStep = m.step; statSps = m.sps; statAt = performance.now();
  document.getElementById('step').textContent = m.step.toLocaleString();
  document.getElementById('cells').textContent = m.attached.toLocaleString();
  const dia = (2 * m.rMax + 1) * CELL_UM / 1000;
  const thk = m.kSpan * CELL_UM / 1000;
  document.getElementById('size').textContent =
    `${dia.toFixed(2)} × ${thk.toFixed(2)} mm`;
  document.getElementById('sps').textContent = m.sps.toLocaleString();
  updateScale(m.rMax, m.kSpan);
}

function onEdgeReached() {
  running = false;
  document.getElementById('btn-run').textContent = '▶ 再開';
  showToast('結晶が計算領域の端に達しました。「↺ 最初から」で再スタートできます');
}

function setEngineBadge(text) {
  const el = document.getElementById('engine-badge');
  if (el) el.textContent = text;
}

function currentParams() { return modelParams(ui.T, ui.RH, ui.P, ui.W); }

// ---------- エンジン管理 (WebGPU 優先 / CPU フォールバック) ----------
// engine: { kind, run, pause, reset, setParams, setSpeed, destroy }
let engine = null;

async function startGpuEngine() {
  const { R, H } = GPU_RES[resChoice];
  const meshWorker = new Worker('./js/mesh-worker.js', { type: 'module' });
  await new Promise((resolve, reject) => {
    meshWorker.onmessage = (e) => { if (e.data.type === 'ready') resolve(); };
    meshWorker.onerror = (e) => reject(new Error(e.message || 'mesh worker failed'));
    meshWorker.postMessage({ type: 'init', R, H });
  });

  // メッシュ抽出のスロットリング: 大きくなるほど間隔を空ける
  let gpu = null;   // 下で代入 (onEntries は create 中の種付着でも発火するため先に宣言)
  let extractInflight = false, dirty = false, lastExtract = 0, lastTris = 0;
  const extractInterval = () =>
    lastTris > 3e6 ? 400 : (lastTris > 1e6 ? 240 : 110);
  function requestExtract(force = false) {
    const now = performance.now();
    if (extractInflight) { dirty = true; return; }
    if (!force && now - lastExtract < extractInterval()) { dirty = true; return; }
    extractInflight = true; dirty = false; lastExtract = now;
    meshWorker.postMessage({ type: 'extract', step: gpu ? gpu.step_n : 0 });
  }
  let extractTimer = setInterval(() => { if (dirty) requestExtract(); }, 60);

  meshWorker.onmessage = (e) => {
    const m = e.data;
    if (m.type !== 'mesh') return;
    setSectorGeometry(buildGeo(m.sector));
    setSeamGeometry(buildGeo(m.seam));
    if (uStep.value < m.step) uStep.value = m.step;
    rebuildSparkles(m.sector.positions.length >= 60 ? m.sector.positions : m.seam.positions, false);
    lastTris = (m.sector.indices.length * 6 + m.seam.indices.length) / 3;
    document.getElementById('tris').textContent = (lastTris / 1e6).toFixed(2) + 'M';
    extractInflight = false;
  };

  gpu = await GpuSim.create(R, H, currentParams(), {
    onEntries: (entries, step) => {
      meshWorker.postMessage({ type: 'attach', entries, step }, [entries.buffer]);
      requestExtract();
    },
    onStats: applyStats,
    onEdge: () => { requestExtract(true); onEdgeReached(); },
    onError: (err) => {
      console.error('GPU engine error', err);
      showToast('WebGPU エラーが発生しました。CPU エンジンに切り替えます');
      switchToCpu();
    },
    onDeviceLost: () => {
      showToast('GPU デバイスが失われました。CPU エンジンに切り替えます');
      switchToCpu();
    },
  });
  gpu.setSpeed(ui.speed);

  setEngineBadge(`WebGPU ${gpu.adapterInfo.architecture || ''} · 格子 R=${R}`.trim());
  return {
    kind: 'gpu',
    run: () => gpu.run(),
    pause: () => { gpu.pause(); requestExtract(true); },
    reset: (params) => {
      gpu.pause();
      // pump が止まるのを待ってからリセット (実行中バッチとの競合回避)
      setTimeout(() => {
        meshWorker.postMessage({ type: 'reset' });
        gpu.reset(params);
        statStep = 0; statSps = 0;
        requestExtract(true);
        if (running) gpu.run();
      }, 80);
    },
    setParams: (p) => gpu.setParams(p),
    setSpeed: (v) => gpu.setSpeed(v),
    destroy: () => {
      clearInterval(extractTimer);
      gpu.destroy();
      meshWorker.terminate();
    },
  };
}

function startCpuEngine() {
  const worker = new Worker('./js/worker.js', { type: 'module' });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'mesh') {
      // CPU 経路は 12 像展開済みの単一メッシュ → seam スロットで描画
      setSeamGeometry(buildGeo({
        positions: m.positions, normals: m.normals, born: m.born, indices: m.indices,
      }));
      if (uStep.value < m.step) uStep.value = m.step;
      rebuildSparkles(m.positions, true);
      document.getElementById('tris').textContent = (m.indices.length / 3 / 1e6).toFixed(2) + 'M';
    } else if (m.type === 'stats') {
      applyStats(m);
    } else if (m.type === 'edge') {
      onEdgeReached();
    }
  };
  worker.postMessage({ type: 'init', R: CPU_RES.R, H: CPU_RES.H, params: currentParams() });
  worker.postMessage({ type: 'speed', value: ui.speed });
  setEngineBadge(`CPU · 格子 R=${CPU_RES.R}`);
  return {
    kind: 'cpu',
    run: () => worker.postMessage({ type: 'run' }),
    pause: () => worker.postMessage({ type: 'pause' }),
    reset: (params) => worker.postMessage({ type: 'reset', params, autorun: running }),
    setParams: (p) => worker.postMessage({ type: 'setParams', params: p }),
    setSpeed: (v) => worker.postMessage({ type: 'speed', value: v }),
    destroy: () => worker.terminate(),
  };
}

function switchToCpu() {
  if (engine) { try { engine.destroy(); } catch (_) { /* noop */ } }
  clearCrystal();
  engine = startCpuEngine();
  if (running) engine.run();
  const chips = document.getElementById('res-section');
  if (chips) chips.style.display = 'none';
}

async function startEngine() {
  if (GpuSim.supported()) {
    try {
      engine = await startGpuEngine();
      if (running) engine.run();
      return;
    } catch (err) {
      console.warn('WebGPU init failed; falling back to CPU:', err);
      showToast('WebGPU を初期化できなかったため CPU エンジンで実行します');
    }
  }
  switchToCpu();
}

async function restartEngine() {
  if (engine) { try { engine.destroy(); } catch (_) { /* noop */ } engine = null; }
  clearCrystal();
  statStep = 0; statSps = 0;
  await startEngine();
}

// ---------- 中谷ダイヤグラム ----------
const ncv = document.getElementById('nakaya');
function drawNakaya() {
  const ctx = ncv.getContext('2d');
  const W2 = ncv.width, H2 = ncv.height;
  const padL = 30, padB = 22, padT = 8, padR = 8;
  const plotW = W2 - padL - padR, plotH = H2 - padT - padB;
  const Tmin = -32, Tmax = 0, Emax = 0.45;
  const xOf = (T) => padL + (T - Tmax) / (Tmin - Tmax) * plotW;
  const yOf = (E) => padT + (1 - E / Emax) * plotH;

  ctx.clearRect(0, 0, W2, H2);
  // 形態領域 (温度帯)
  const zones = [
    [0, -3.5, 'rgba(120,180,255,0.10)', '板'],
    [-3.5, -10, 'rgba(150,255,190,0.08)', '柱・針'],
    [-10, -22, 'rgba(120,180,255,0.10)', '板・樹枝'],
    [-22, -32, 'rgba(150,255,190,0.08)', '柱/板'],
  ];
  ctx.textAlign = 'center';
  for (const [t0, t1, col, label] of zones) {
    ctx.fillStyle = col;
    ctx.fillRect(xOf(t0), padT, xOf(t1) - xOf(t0), plotH);
    ctx.fillStyle = 'rgba(200,225,255,0.55)';
    ctx.font = '10px sans-serif';
    ctx.fillText(label, (xOf(t0) + xOf(t1)) / 2, padT + 12);
  }
  // 水飽和線 (第一原理: Magnus 式から)
  ctx.strokeStyle = 'rgba(255,255,255,0.65)';
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  let first = true;
  for (let T = -0.2; T >= Tmin; T -= 0.4) {
    const y = yOf(Math.min(Emax, waterSatExcess(T)));
    if (first) { ctx.moveTo(xOf(T), y); first = false; } else ctx.lineTo(xOf(T), y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = '9px sans-serif';
  ctx.fillText('水飽和線', xOf(-26), yOf(waterSatExcess(-26)) - 5);
  // 軸
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.strokeRect(padL, padT, plotW, plotH);
  ctx.fillStyle = 'rgba(220,235,255,0.7)';
  ctx.font = '9px sans-serif';
  for (const T of [0, -5, -10, -15, -20, -25, -30]) {
    ctx.fillText(`${T}`, xOf(T), H2 - 8);
  }
  ctx.fillText('気温 ℃', padL + plotW / 2, H2 - 0.5);
  ctx.save();
  ctx.translate(9, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('過剰水蒸気 g/m³', 0, 0);
  ctx.restore();
  // 現在位置マーカー
  const sat = rhoSatIce(ui.T);
  const eff = Math.min(Math.max(0, (ui.RH / 100 - 1) * sat), ui.W);
  const mx = xOf(ui.T), my = yOf(Math.min(Emax, eff));
  const g = ctx.createRadialGradient(mx, my, 0, mx, my, 9);
  g.addColorStop(0, 'rgba(140,220,255,0.95)');
  g.addColorStop(1, 'rgba(140,220,255,0)');
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(mx, my, 9, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(mx, my, 2.6, 0, Math.PI * 2); ctx.fill();
}

// ---------- UI 配線 ----------
function fmt(id, v) { document.getElementById(id).textContent = v; }
function refreshLabels() {
  fmt('v-T', ui.T.toFixed(1) + ' ℃');
  fmt('v-RH', ui.RH.toFixed(0) + ' %');
  fmt('v-W', ui.W.toFixed(2) + ' g/m³');
  fmt('v-P', ui.P.toFixed(0) + ' hPa');
  fmt('v-speed', '×' + ui.speed);
  fmt('morph', morphologyLabel(ui.T, ui.RH, ui.W));
  drawNakaya();
}

function pushParams() {
  if (engine) engine.setParams(currentParams());
  refreshLabels();
}

function bindSlider(id, key, parse = parseFloat, onChange = pushParams) {
  const el = document.getElementById(id);
  el.value = ui[key];
  el.addEventListener('input', () => { ui[key] = parse(el.value); onChange(); });
  return el;
}
bindSlider('s-T', 'T');
bindSlider('s-RH', 'RH');
bindSlider('s-W', 'W');
bindSlider('s-P', 'P');
bindSlider('s-speed', 'speed', (v) => parseInt(v), () => {
  if (engine) engine.setSpeed(ui.speed);
  refreshLabels();
});

const PRESETS = {
  dendrite: { T: -15, RH: 140, W: 0.30, P: 1013, label: '星型樹枝' },
  sector:   { T: -13, RH: 112, W: 0.18, P: 1013, label: '扇形板' },
  plate:    { T: -12, RH: 105, W: 0.30, P: 1013, label: '六角板' },
  needle:   { T: -5,  RH: 128, W: 0.30, P: 1013, label: '針' },
  column:   { T: -7,  RH: 105, W: 0.30, P: 1013, label: '角柱' },
  coldcol:  { T: -28, RH: 116, W: 0.20, P: 1013, label: '低温柱' },
};
document.querySelectorAll('[data-preset]').forEach(btn => {
  btn.addEventListener('click', () => {
    const p = PRESETS[btn.dataset.preset];
    Object.assign(ui, { T: p.T, RH: p.RH, W: p.W, P: p.P });
    for (const [sid, key] of [['s-T', 'T'], ['s-RH', 'RH'], ['s-W', 'W'], ['s-P', 'P']]) {
      document.getElementById(sid).value = ui[key];
    }
    pushParams();
  });
});

// 解像度セレクタ (WebGPU 時のみ表示)
document.querySelectorAll('[data-res]').forEach(btn => {
  const { R } = GPU_RES[btn.dataset.res];
  btn.textContent = btn.textContent.replace('{R}', R);
  if (btn.dataset.res === resChoice) btn.classList.add('active');
  btn.addEventListener('click', async () => {
    if (btn.dataset.res === resChoice) return;
    resChoice = btn.dataset.res;
    localStorage.setItem('snowlab-res', resChoice);
    document.querySelectorAll('[data-res]').forEach(b =>
      b.classList.toggle('active', b.dataset.res === resChoice));
    showToast(`計算解像度を変更しました (格子 R=${GPU_RES[resChoice].R})。再構築中…`);
    await restartEngine();
  });
});

document.getElementById('btn-run').addEventListener('click', (e) => {
  running = !running;
  if (engine) (running ? engine.run() : engine.pause());
  e.target.textContent = running ? '⏸ 一時停止' : '▶ 再開';
});
document.getElementById('btn-reset').addEventListener('click', () => {
  running = true;
  if (engine) engine.reset(currentParams());
  document.getElementById('btn-run').textContent = '⏸ 一時停止';
});
document.getElementById('tg-rotate').addEventListener('change', (e) => {
  controls.autoRotate = e.target.checked;
});
document.getElementById('tg-hq').addEventListener('change', (e) => {
  hq = e.target.checked;
  const mat = currentMat();
  for (const m of sectorMeshes) m.material = mat;
  if (seamMesh) seamMesh.material = mat;
});

let toastTimer = null;
function showToast(text) {
  const el = document.getElementById('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 6000);
}

// ---------- モバイル: 折りたたみパネル ----------
if (IS_MOBILE) {
  const panel = document.getElementById('controls');
  const head = document.getElementById('panel-head');
  panel.classList.add('mobile', 'collapsed');
  head.addEventListener('click', () => {
    panel.classList.toggle('collapsed');
    document.getElementById('panel-chevron').textContent =
      panel.classList.contains('collapsed') ? '▴' : '▾';
  });
  document.getElementById('panel-chevron').textContent = '▴';
}

// タイトルはロード後数秒だけ表示してフェードアウト (画面を占有しない)
setTimeout(() => document.getElementById('title').classList.add('hidden'), 7000);

refreshLabels();
startEngine();

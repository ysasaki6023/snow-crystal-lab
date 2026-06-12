// GPU/CPU パリティテスト (要: ローカルサーバー on :8741)
//   node test/gpu-parity.mjs [steps] [checkEvery]
import puppeteer from 'puppeteer-core';

const steps = Number(process.argv[2] || 1500);
const every = Number(process.argv[3] || 100);

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--use-angle=metal', '--enable-unsafe-webgpu'],
});
const page = await browser.newPage();
page.on('console', m => { if (m.type() === 'error') console.log('console.error:', m.text()); });
page.on('pageerror', e => console.log('PAGE ERROR:', e.message));
await page.goto('http://localhost:8741/test/parity.html', { waitUntil: 'networkidle2' });
await page.waitForFunction('window.parityReady === true', { timeout: 10000 });

const CASES = [
  ['樹枝状', -15, 140, 1013, 0.30],
  ['角柱', -7, 105, 1013, 0.30],
  ['針状', -5, 128, 1013, 0.30],
];

let fail = 0;
for (const [name, T, RH, P, W] of CASES) {
  const r = await page.evaluate(
    (T2, RH2, P2, W2, st, ev) => window.runParity(160, 96, T2, RH2, P2, W2, st, ev),
    T, RH, P, W, steps, every);
  const last = r.results[r.results.length - 1];
  const attRel = Math.abs(last.cpuAtt - last.gpuAtt) / Math.max(last.cpuAtt, 1);
  const jaccard = r.both / (r.both + r.cpuOnly + r.gpuOnly);
  const ok = attRel < 0.03 && Math.abs(last.cpuR - last.gpuR) <= 3 && jaccard > 0.97;
  if (!ok) fail++;
  console.log(`--- ${name} @step ${last.step}  ${ok ? 'OK' : 'FAIL'}`);
  console.log(`    attached cpu=${last.cpuAtt} gpu=${last.gpuAtt} (rel ${(attRel * 100).toFixed(2)}%)`);
  console.log(`    rMax cpu=${last.cpuR} gpu=${last.gpuR}  kSpan cpu=${last.cpuK} gpu=${last.gpuK}`);
  console.log(`    cell set: both=${r.both} cpuOnly=${r.cpuOnly} gpuOnly=${r.gpuOnly} (jaccard ${(jaccard * 100).toFixed(2)}%)`);
}
await browser.close();
process.exit(fail ? 1 : 0);

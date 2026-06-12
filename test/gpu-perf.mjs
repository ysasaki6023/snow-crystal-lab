import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--use-angle=metal', '--enable-unsafe-webgpu'],
});
const page = await browser.newPage();
page.on('pageerror', e => console.log('PAGE ERROR:', e.message));
await page.goto('http://localhost:8741/test/gpu-perf.html', { waitUntil: 'networkidle2' });
await page.waitForFunction('window.perfReady === true');
const [R, H, sec] = [Number(process.argv[2] || 512), Number(process.argv[3] || 320), Number(process.argv[4] || 30)];
const r = await page.evaluate((R2, H2, s) => window.runPerf(R2, H2, -15, 140, 1013, 0.30, s), R, H, sec);
for (const s of r.samples) {
  const sps = s.t > 0 ? Math.round(s.step / s.t) : 0;
  console.log(`t=${s.t}s step=${s.step} (~${sps} sps avg) att=${s.att.toLocaleString()} rMax=${s.rMax} kSpan=${s.kSpan} B=${s.B} ${s.msPerStep}ms/step`);
}
console.log('final:', JSON.stringify({ edge: r.edge, step: r.finalStep, att: r.att, rMax: r.rMax }));
await browser.close();

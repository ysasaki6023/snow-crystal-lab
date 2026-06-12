import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--use-angle=metal', '--enable-unsafe-webgpu', '--window-size=1480,920'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1480, height: 920 });
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGE: ' + e.message));
await page.goto('http://localhost:8741/index.html', { waitUntil: 'networkidle2', timeout: 30000 });
const preset = process.argv[4];
if (preset) {
  await page.waitForSelector(`[data-preset="${preset}"]`);
  await page.click(`[data-preset="${preset}"]`);
  await page.click('#btn-reset');
}
const waitSec = Number(process.argv[2] || 20);
await new Promise(r => setTimeout(r, waitSec * 1000));
const stats = await page.evaluate(() => ({
  step: document.getElementById('step')?.textContent,
  cells: document.getElementById('cells')?.textContent,
  size: document.getElementById('size')?.textContent,
  sps: document.getElementById('sps')?.textContent,
  tris: document.getElementById('tris')?.textContent,
  morph: document.getElementById('morph')?.textContent,
  badge: document.getElementById('engine-badge')?.textContent,
}));
console.log('stats:', JSON.stringify(stats));
await page.screenshot({ path: process.argv[3] || '/tmp/snowlab-gpu.png' });
console.log('errors:', errors.length ? errors.slice(0, 10) : 'none');
await browser.close();

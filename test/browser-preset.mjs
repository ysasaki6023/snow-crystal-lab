import puppeteer from 'puppeteer-core';
const preset = process.argv[2] || 'column';
const waitSec = Number(process.argv[3] || 60);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--use-angle=metal', '--window-size=1480,920'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1480, height: 920 });
const errors = [];
page.on('pageerror', e => errors.push('PAGE: ' + e.message));
await page.goto('http://localhost:8741/index.html', { waitUntil: 'networkidle2', timeout: 30000 });
await page.click(`[data-preset="${preset}"]`);
await page.click('#btn-reset');
await new Promise(r => setTimeout(r, waitSec * 1000));
const stats = await page.evaluate(() => ({
  step: document.getElementById('step')?.textContent,
  size: document.getElementById('size')?.textContent,
  morph: document.getElementById('morph')?.textContent,
}));
console.log(JSON.stringify(stats));
await page.screenshot({ path: `/tmp/snowsim-${preset}.png` });
console.log('errors:', errors.length ? errors : 'none');
await browser.close();

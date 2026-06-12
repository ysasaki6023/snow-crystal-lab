import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--use-angle=metal', '--window-size=1480,920'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1480, height: 920 });
const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => errors.push('PAGE: ' + e.message));
await page.goto('https://ysasaki6023.github.io/snow-crystal-lab/', { waitUntil: 'networkidle2', timeout: 45000 });
await new Promise(r => setTimeout(r, 50000));
const stats = await page.evaluate(() => ({
  step: document.getElementById('step')?.textContent,
  size: document.getElementById('size')?.textContent,
  morph: document.getElementById('morph')?.textContent,
}));
console.log(JSON.stringify(stats));
await page.screenshot({ path: '/tmp/snowsim-live.png' });
console.log('errors:', errors.length ? errors.slice(0, 5) : 'none');
await browser.close();

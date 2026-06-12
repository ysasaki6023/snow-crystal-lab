import puppeteer from 'puppeteer-core';
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
// 一時停止/再開とライト品質の動作確認
await page.click('#btn-run');
await new Promise(r => setTimeout(r, 1500));
const s1 = await page.evaluate(() => document.getElementById('step').textContent);
await new Promise(r => setTimeout(r, 2500));
const s2 = await page.evaluate(() => document.getElementById('step').textContent);
console.log(`pause check: ${s1} → ${s2} (一致なら停止OK)`);
await page.click('#btn-run');
await page.evaluate(() => { const t = document.getElementById('tg-hq'); t.checked = false; t.dispatchEvent(new Event('change')); });
await new Promise(r => setTimeout(r, 4000));
await page.screenshot({ path: '/tmp/snowsim-lite.png' });
await page.evaluate(() => { const t = document.getElementById('tg-hq'); t.checked = true; t.dispatchEvent(new Event('change')); });
console.log('growing for hero shot (~220s)...');
await new Promise(r => setTimeout(r, 220000));
const stats = await page.evaluate(() => ({
  step: document.getElementById('step')?.textContent,
  size: document.getElementById('size')?.textContent,
  tris: document.getElementById('tris')?.textContent,
}));
console.log(JSON.stringify(stats));
await page.screenshot({ path: '/tmp/snowsim-hero.png' });
console.log('errors:', errors.length ? errors : 'none');
await browser.close();

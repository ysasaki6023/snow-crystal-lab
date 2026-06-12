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
await page.click('[data-preset="column"]');
await page.click('#btn-reset');
console.log('柱を成長中 (70s)...');
await new Promise(r => setTimeout(r, 70000));
await page.screenshot({ path: '/tmp/snowsim-cap1.png' });
// 気温を -15℃ に変更 (成長中のパラメータ変更)
await page.evaluate(() => {
  const s = document.getElementById('s-T');
  s.value = -15;
  s.dispatchEvent(new Event('input'));
  const rh = document.getElementById('s-RH');
  rh.value = 138;
  rh.dispatchEvent(new Event('input'));
});
console.log('T→-15℃ に変更、板の成長待ち (150s)...');
await new Promise(r => setTimeout(r, 150000));
const stats = await page.evaluate(() => ({
  step: document.getElementById('step')?.textContent,
  size: document.getElementById('size')?.textContent,
}));
console.log(JSON.stringify(stats));
await page.screenshot({ path: '/tmp/snowsim-cap2.png' });
console.log('errors:', errors.length ? errors : 'none');
await browser.close();

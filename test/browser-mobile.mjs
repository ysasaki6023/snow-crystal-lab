import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--use-angle=metal'],
});
const page = await browser.newPage();
await page.emulate({
  viewport: { width: 390, height: 844, deviceScaleFactor: 3, isMobile: true, hasTouch: true },
  userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
});
const errors = [];
page.on('pageerror', e => errors.push('PAGE: ' + e.message));
await page.goto('https://ysasaki6023.github.io/snow-crystal-lab/', { waitUntil: 'networkidle2', timeout: 30000 });
const env = await page.evaluate(() => ({
  coarse: matchMedia('(pointer: coarse)').matches,
  w: innerWidth,
  collapsed: document.getElementById('controls').classList.contains('collapsed'),
}));
console.log('env:', JSON.stringify(env));
await new Promise(r => setTimeout(r, 40000));
const stats = await page.evaluate(() => ({
  step: document.getElementById('step')?.textContent,
  size: document.getElementById('size')?.textContent,
}));
console.log(JSON.stringify(stats));
await page.screenshot({ path: '/tmp/snowsim-mobile1.png' });
// パネルを開く
await page.tap('#panel-head');
await new Promise(r => setTimeout(r, 800));
await page.screenshot({ path: '/tmp/snowsim-mobile2.png' });
console.log('errors:', errors.length ? errors.slice(0, 5) : 'none');
await browser.close();

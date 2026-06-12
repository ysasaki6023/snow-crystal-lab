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
await page.goto('http://localhost:8741/index.html', { waitUntil: 'networkidle2', timeout: 30000 });
for (const [sec, file] of [[45, '/tmp/snowsim-a.png'], [30, '/tmp/snowsim-b.png'], [45, '/tmp/snowsim-c.png']]) {
  await new Promise(r => setTimeout(r, sec * 1000));
  await page.screenshot({ path: file });
  const stats = await page.evaluate(() => ({
    step: document.getElementById('step')?.textContent,
    size: document.getElementById('size')?.textContent,
    tris: document.getElementById('tris')?.textContent,
  }));
  console.log(file, JSON.stringify(stats));
}
console.log('errors:', errors.length ? errors.slice(0, 10) : 'none');
await browser.close();

/** Real-asset visual smoke check. Requires npm run serve. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {chromium} from 'playwright';
const manifest=JSON.parse(fs.readFileSync(new URL('../../../public/camp-sherman/scene.json',import.meta.url)));
const directory=new URL('./scratchpad/',import.meta.url);fs.mkdirSync(directory,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH}:{})});
const results=[];
try {
 for(const mobile of [false,true]) {
  const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1440,height:1000},isMobile:mobile,hasTouch:mobile});
  const errors=[],remote=[];page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(!request.url().startsWith('http://127.0.0.1:8086/')&&!request.url().startsWith('data:')&&!request.url().startsWith('blob:'))remote.push(request.url());});
  const started=Date.now();await page.goto('http://127.0.0.1:8086/');await page.locator('body[data-ready=true]').waitFor({timeout:60000});
  const readyMilliseconds=Date.now()-started;
  await page.screenshot({path:new URL(`final-${mobile?'mobile':'desktop'}-home.png`,directory).pathname});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth),false);
  if(!mobile) {
   await page.locator('#trees').click();await page.screenshot({path:new URL('final-no-trees.png',directory).pathname});
   await page.locator('#roof').click();await page.screenshot({path:new URL('final-cutaway.png',directory).pathname});
   await page.locator('#reset').click();
   for(const point of manifest.waypoints){
    await page.locator('#viewpoint').selectOption(point.id);
    await page.screenshot({path:new URL(`final-view-${point.id}.png`,directory).pathname});
   }
  }
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  results.push({mode:mobile?'touch':'desktop',readyMilliseconds,errors,remoteRequests:remote.length});await page.close();
 }
 console.log(JSON.stringify(results,null,2));
}finally{await browser.close();}

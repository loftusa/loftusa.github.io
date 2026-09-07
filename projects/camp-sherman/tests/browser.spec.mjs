import {test,expect} from '@playwright/test';
import {fixtureGlb,fixtureManifest} from './fixture.mjs';
import {PNG} from 'pngjs';

async function fixture(page) {
  await page.route('**/scene.json',route=>route.fulfill({json:fixtureManifest}));
  await page.route('**/assets/camp-sherman.glb',route=>route.fulfill({contentType:'model/gltf-binary',body:fixtureGlb()}));
}
test('local model loads, roof cuts away, walking viewpoints work and reset returns to site', async ({page}) => {
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await fixture(page);await page.goto('/');
  await expect(page.locator('#loading')).toBeHidden();
  await expect(page.getByRole('button',{name:'Walk',exact:true})).toBeEnabled();
  const original=await page.locator('canvas').screenshot();
  await page.screenshot({path:'tests/scratchpad/fixture-desktop.png'});
  await page.getByRole('button',{name:'Hide roof'}).click();
  await expect(page.getByRole('button',{name:'Show roof'})).toHaveAttribute('aria-pressed','true');
  expect(Buffer.compare(original,await page.locator('canvas').screenshot())).not.toBe(0);
  await page.getByLabel('Go to a viewpoint').selectOption('entry');
  await expect(page.locator('body')).toHaveAttribute('data-mode','walk');
  const before=await page.locator('canvas').screenshot();
  await page.keyboard.down('w');await page.waitForTimeout(500);await page.keyboard.up('w');
  expect(Buffer.compare(before,await page.locator('canvas').screenshot())).not.toBe(0);
  await page.getByRole('button',{name:'Reset view'}).click();
  await expect(page.locator('body')).toHaveAttribute('data-mode','orbit');
  await expect(page.getByLabel('Go to a viewpoint')).toHaveValue('');
  expect(errors).toEqual([]);
});
test('failed model load is actionable and retry succeeds', async ({page}) => {
  await fixture(page);let failed=false;
  await page.route('**/assets/camp-sherman.glb',route=>{
    if (!failed) {failed=true;return route.fulfill({status:503,body:'Unavailable'});}
    return route.fulfill({contentType:'model/gltf-binary',body:fixtureGlb()});
  });
  await page.goto('/');await expect(page.getByRole('button',{name:'Try again'})).toBeVisible();
  await page.getByRole('button',{name:'Try again'}).click();
  await expect(page.locator('#loading')).toBeHidden();
});
test('trees can be hidden independently and reset restores their visibility',async({page})=>{
  await fixture(page);await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  const before=await page.locator('canvas').screenshot();
  await page.getByRole('button',{name:'Hide trees'}).click();
  await expect(page.getByRole('button',{name:'Show trees'})).toHaveAttribute('aria-pressed','true');
  expect(Buffer.compare(before,await page.locator('canvas').screenshot())).not.toBe(0);
  await page.getByRole('button',{name:'Reset view'}).click();
  await expect(page.getByRole('button',{name:'Hide trees'})).toHaveAttribute('aria-pressed','false');
});
test('an optional local HDR loads, while a failed HDR falls back without losing the model',async({page})=>{
  await fixture(page);
  await page.route('**/scene.json',route=>route.fulfill({json:{...fixtureManifest,assets:{...fixtureManifest.assets,environment:'./assets/sunset_forest_1k.hdr'}}}));
  let hdrRequested=false;page.on('request',request=>{if(request.url().endsWith('.hdr'))hdrRequested=true;});
  const successfulHdr=page.waitForResponse(response=>response.url().endsWith('.hdr'));
  await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  expect((await successfulHdr).status()).toBe(200);
  expect(hdrRequested).toBe(true);
  await page.route('**/*.hdr',route=>route.fulfill({status:503,body:'Unavailable'}));
  await page.reload();await expect(page.locator('#loading')).toBeHidden();
  await expect(page.getByRole('button',{name:'Walk',exact:true})).toBeEnabled();
});
test('an overview viewpoint stays in orbit mode instead of falling to the ground',async({page})=>{
  await fixture(page);
  await page.route('**/scene.json',route=>route.fulfill({json:{...fixtureManifest,waypoints:[...fixtureManifest.waypoints,{id:'overview',label:'Whole property',mode:'orbit',position:[15,12,18],lookAt:[0,0,0]}]}}));
  await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  await page.getByLabel('Go to a viewpoint').selectOption('overview');
  await expect(page.locator('body')).toHaveAttribute('data-mode','orbit');
  await expect(page.locator('#view-label')).toHaveText('Whole property');
});
test('touch layout fits a narrow screen and offers movement after entering walk mode', async ({page}) => {
  await page.setViewportSize({width:390,height:844});
  await fixture(page);await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  await page.getByRole('button',{name:'Walk',exact:true}).click();
  await expect(page.getByRole('button',{name:'Move forward'})).toBeVisible();
  await page.screenshot({path:'tests/scratchpad/fixture-mobile.png'});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
test('metallic interior surfaces retain readable lighting without an external environment download', async ({page}) => {
  const remote=[];page.on('request',request=>{if(!request.url().startsWith('http://127.0.0.1:8086/'))remote.push(request.url());});
  await fixture(page);await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  await page.getByLabel('Go to a viewpoint').selectOption('entry');
  const picture=PNG.sync.read(await page.locator('canvas').screenshot());
  const offset=((Math.floor(picture.height/2)*picture.width)+Math.floor(picture.width/2))*4;
  assertReadable(picture.data.subarray(offset,offset+3));
  expect(remote).toEqual([]);
});
function assertReadable(rgb) {expect((rgb[0]+rgb[1]+rgb[2])/3).toBeGreaterThan(30);}

for(const capture of ['unavailable','rejected','error-event'])test(`click enables mouse look when capture is ${capture}, and Escape stops it`,async({page})=>{
  await page.addInitScript(capture=>{
    Element.prototype.requestPointerLock=capture==='unavailable'?undefined:capture==='rejected'?()=>Promise.reject(new DOMException('Capture denied','NotAllowedError')):()=>document.dispatchEvent(new Event('pointerlockerror'));
  },capture);
  await fixture(page);await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  await page.getByLabel('Go to a viewpoint').selectOption('entry');
  await page.mouse.click(600,350);
  const before=await page.locator('canvas').screenshot();
  await page.mouse.move(750,390);
  expect(Buffer.compare(before,await page.locator('canvas').screenshot())).not.toBe(0);
  await page.keyboard.press('Escape');
  const stopped=await page.locator('canvas').screenshot();
  await page.mouse.move(850,420);
  expect(Buffer.compare(stopped,await page.locator('canvas').screenshot())).toBe(0);
});

test('mouse dragging turns in walk mode without enabling hover look on release',async({page})=>{
  await fixture(page);await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  await page.getByLabel('Go to a viewpoint').selectOption('entry');
  await page.mouse.move(600,350);
  const before=await page.locator('canvas').screenshot();
  await page.mouse.down();await page.mouse.move(750,400,{steps:5});
  expect(Buffer.compare(before,await page.locator('canvas').screenshot())).not.toBe(0);
  await page.mouse.up();
  const stopped=await page.locator('canvas').screenshot();
  await page.mouse.move(850,420);
  expect(Buffer.compare(stopped,await page.locator('canvas').screenshot())).toBe(0);
});

test('native click capture turns the camera and clicking again releases it',async({page})=>{
  await fixture(page);await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  await page.getByLabel('Go to a viewpoint').selectOption('entry');
  await page.mouse.click(600,350);
  await expect.poll(()=>page.evaluate(()=>document.pointerLockElement?.tagName)).toBe('CANVAS');
  const before=await page.locator('canvas').screenshot();
  await page.mouse.move(750,400);
  expect(Buffer.compare(before,await page.locator('canvas').screenshot())).not.toBe(0);
  await page.mouse.click(750,400);
  await expect.poll(()=>page.evaluate(()=>document.pointerLockElement===null)).toBe(true);
});

test('touch dragging changes the view and holding the touch pad moves through the room',async ({browser})=>{
  const context=await browser.newContext({viewport:{width:390,height:844},hasTouch:true,isMobile:true});
  const page=await context.newPage();
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await fixture(page);await page.goto('http://127.0.0.1:8086/');await expect(page.locator('#loading')).toBeHidden();
  await page.getByRole('button',{name:'Walk',exact:true}).tap();
  const cdp=await context.newCDPSession(page);
  const before=await page.locator('canvas').screenshot();
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:200,y:350}]});
  await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:290,y:350}]});
  await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  expect(Buffer.compare(before,await page.locator('canvas').screenshot())).not.toBe(0);
  const button=await page.getByRole('button',{name:'Move forward'}).boundingBox();
  const turned=await page.locator('canvas').screenshot();
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:button.x+button.width/2,y:button.y+button.height/2}]});
  await page.waitForTimeout(500);
  await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  expect(Buffer.compare(turned,await page.locator('canvas').screenshot())).not.toBe(0);
  expect(errors).toEqual([]);
  await page.screenshot({path:'tests/scratchpad/fixture-touch.png'});
  await context.close();
});

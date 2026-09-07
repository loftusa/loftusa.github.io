import {test,expect} from '@playwright/test';
import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {fixtureGlb,fixtureManifest} from './fixture.mjs';

// Observe the real camera through a test-only bundle, with no production debug API.
const source=await readFile(new URL('../viewer.mjs',import.meta.url),'utf8');
const bundle=await build({stdin:{contents:source+'\nglobalThis.flightCamera=camera;',resolveDir:fileURLToPath(new URL('../',import.meta.url)),sourcefile:'viewer.mjs'},bundle:true,format:'esm',write:false});
async function setup(page,{clock=true}={}) {
  // Drive actual animation callbacks deterministically; virtual browser clocks
  // stalled RAF in this fixture, while software GPU speed varies by host.
  if(clock)await page.addInitScript(()=>{
    let nextId=0,time=0;const callbacks=new Map();
    window.requestAnimationFrame=callback=>{callbacks.set(++nextId,callback);return nextId;};
    window.cancelAnimationFrame=id=>callbacks.delete(id);
    window.advanceFlightFrames=milliseconds=>{
      for(let elapsed=0;elapsed<milliseconds;elapsed+=16) {
        time+=16;const pending=[...callbacks.values()];callbacks.clear();
        for(const callback of pending)callback(time);
      }
    };
  });
  await page.route('**/viewer.bundle.mjs',route=>route.fulfill({contentType:'text/javascript',body:bundle.outputFiles[0].text}));
  await page.route('**/scene.json',route=>route.fulfill({json:fixtureManifest}));
  await page.route('**/assets/camp-sherman.glb',route=>route.fulfill({contentType:'model/gltf-binary',body:fixtureGlb()}));
  await page.goto('/');await expect(page.locator('#loading')).toBeHidden();
  await page.locator('#viewpoint').selectOption('entry');
}
const frames=(page,ms)=>page.evaluate(ms=>window.advanceFlightFrames(ms),ms);
const position=page=>page.evaluate(()=>globalThis.flightCamera.position.toArray());
async function hold(page,keys,ms) {
  for(const key of keys)await page.keyboard.down(key);
  await frames(page,ms);
  for(const key of [...keys].reverse())await page.keyboard.up(key);
}
test('Space rises through the roof, Control descends, and released keys hover',async({page})=>{
  await setup(page);
  await hold(page,['Space'],900);
  const risen=await position(page);expect(risen[1]).toBeGreaterThan(3.2);
  await frames(page,200);expect((await position(page))[1]).toBeCloseTo(risen[1],5);
  await hold(page,['Control'],400);expect((await position(page))[1]).toBeLessThan(risen[1]-.5);
  expect(await page.evaluate(()=>scrollY)).toBe(0);
});
test('Shift accelerates WASD through walls and releasing it restores normal speed',async({page})=>{
  await setup(page);
  await hold(page,['w'],400);const normal=1-(await position(page))[2];
  await page.locator('#viewpoint').selectOption('entry');
  await hold(page,['Shift','w'],700);const boosted=1-(await position(page))[2];
  expect(boosted).toBeGreaterThan(normal*3);
  expect((await position(page))[2]).toBeLessThan(-3.2);
  const before=(await position(page))[2];await hold(page,['w'],400);
  expect(before-(await position(page))[2]).toBeLessThan(2);
});
test('Escape and focus loss stop held flight controls',async({page})=>{
  await setup(page);await page.keyboard.down('Space');await frames(page,150);
  await page.keyboard.press('Escape');const stopped=await position(page);
  await frames(page,150);expect(await position(page)).toEqual(stopped);await page.keyboard.up('Space');
  await page.keyboard.down('w');await frames(page,150);
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));const blurred=await position(page);
  await frames(page,150);expect(await position(page)).toEqual(blurred);await page.keyboard.up('w');
});

test('touch flight buttons rise, descend, boost and stop on release',async({page})=>{
  await page.setViewportSize({width:390,height:844});await setup(page,{clock:false});
  const cdp=await page.context().newCDPSession(page);
  await cdp.send('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:2});
  const start=async(names)=>{
    const points=[];
    for(const name of names) {
      const box=await page.getByRole('button',{name,exact:true}).boundingBox();
      points.push({id:points.length+1,x:box.x+box.width/2,y:box.y+box.height/2});
      await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:points});
    }
  };
  const stop=()=>cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  await start(['Fly up']);await expect.poll(async()=>(await position(page))[1]).toBeGreaterThan(2.4);await stop();
  const risen=(await position(page))[1];await page.waitForTimeout(200);expect((await position(page))[1]).toBe(risen);
  await start(['Fly down']);await expect.poll(async()=>(await position(page))[1]).toBeLessThan(risen-.8);await stop();
  await start(['Fly faster','Move forward']);await expect.poll(async()=>(await position(page))[2]).toBeLessThan(-3.2);await stop();
  const stopped=await position(page);await page.waitForTimeout(200);expect(await position(page)).toEqual(stopped);
});

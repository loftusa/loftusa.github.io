import test from 'node:test';
import assert from 'node:assert/strict';
import { movementVector, frameSeconds, advanceWalker, validateManifest, orbitFramingScale } from '../navigation.mjs';

const bounds = {min: [-5, -5, -5], max: [5, 8, 5]};
const state = (x=0,y=1.65,z=0) => ({position: [x,y,z], verticalSpeed: 0});
const floor = {groundAt: () => 0, blockedAt: () => false};

test('narrow orbit views preserve horizontal scene coverage while landscape views retain authored distance',()=>{
  assert.equal(orbitFramingScale(1.5),1);
  assert.equal(orbitFramingScale(.5)*.5,1.2);
  assert.equal(orbitFramingScale(.8)*.8,1.2);
});

test('diagonal input moves at the same speed as forward and follows yaw', () => {
  const vector = movementVector(1,1,0);
  assert.ok(Math.abs(Math.hypot(vector.x,vector.z)-1) < 1e-8);
  const east = movementVector(0,1,-Math.PI/2);
  assert.ok(Math.abs(east.x-1) < 1e-8);
  assert.ok(Math.abs(east.z) < 1e-8);
});
test('tab suspension cannot create an unbounded movement step', () => {
  assert.equal(frameSeconds(5000), .05);
  assert.equal(frameSeconds(-10), 0);
  assert.equal(frameSeconds(16), .016);
});
test('walker slides along a wall without crossing it', () => {
  const next = advanceWalker(state(), {x:1,z:1}, .05, {...floor, blockedAt: ([x]) => x > .02}, bounds);
  assert.equal(next.position[0], 0);
  assert.ok(next.position[2] > 0);
});
test('walkable stair riser is climbed but a tall platform blocks movement', () => {
  const step = advanceWalker(state(), {x:1,z:0}, .05, {...floor, groundAt: (x) => x > 0 ? .18 : 0}, bounds);
  assert.ok(Math.abs(step.position[1]-1.83)<1e-8);
  const tall = advanceWalker(state(), {x:1,z:0}, .05, {...floor, groundAt: (x) => x > 0 ? .7 : 0}, bounds);
  assert.equal(tall.position[0],0);
});
test('gravity lands on a lower floor without falling through it', () => {
  let walker = state(0,3,0);
  for (let i=0;i<100;i++) walker = advanceWalker(walker, {x:0,z:0}, .016, floor, bounds);
  assert.equal(walker.position[1],1.65);
  assert.equal(walker.verticalSpeed,0);
});
test('walking bounds keep the eye away from the site edge', () => {
  const next = advanceWalker(state(4.7), {x:1,z:0}, .05, floor, bounds);
  assert.ok(next.position[0] <= 4.72);
});
const manifest = {title:'Camp Sherman', assets:{scene:'./assets/camp-sherman.glb'}, home:{camera:[0,3,5],target:[0,0,0]},bounds,waypoints:[{id:'entry',label:'Entry',position:[0,1.65,0],lookAt:[1,1.65,0]}]};
test('manifest accepts finite Y-up coordinates and local GLB paths', () => {
  assert.equal(validateManifest(structuredClone(manifest)).waypoints[0].id,'entry');
});
test('optional HDR lighting must be a local asset rather than a remote or parent path', () => {
  const lit=structuredClone(manifest);lit.assets.environment='./assets/sunset_forest_1k.hdr';
  assert.equal(validateManifest(lit).assets.environment,'./assets/sunset_forest_1k.hdr');
  for (const path of ['https://example.com/sky.hdr','./assets/../secret.hdr','./assets/sky.exe']) {
    lit.assets.environment=path;assert.throws(()=>validateManifest(lit));
  }
});
test('orbit viewpoints are accepted but unrecognized navigation modes fail early',()=>{
  const overview=structuredClone(manifest);overview.waypoints[0].mode='orbit';
  assert.equal(validateManifest(overview).waypoints[0].mode,'orbit');
  overview.waypoints[0].mode='fly';assert.throws(()=>validateManifest(overview));
});
test('practical lights require finite positions and nonnegative bounded brightness',()=>{
  const lit=structuredClone(manifest);lit.lights=[{position:[0,2,0],color:'#ffe4bc',intensity:4,distance:8}];
  assert.equal(validateManifest(lit).lights.length,1);
  for(const mutate of [m=>m.lights[0].position[0]=Infinity,m=>m.lights[0].intensity=-1,m=>m.lights[0].color='red;',m=>m.lights[0].distance=0]){
    const invalid=structuredClone(lit);mutate(invalid);assert.throws(()=>validateManifest(invalid));
  }
});
test('malformed bounds, nonfinite positions, duplicate waypoints and remote scenes fail before loading', () => {
  for (const mutate of [m => m.bounds.max[0]=-10,m => m.home.camera[0]=NaN,m => m.waypoints.push(m.waypoints[0]),m => m.assets.scene='https://example.com/model.glb']) {
    const invalid = structuredClone(manifest); mutate(invalid);
    assert.throws(() => validateManifest(invalid));
  }
});

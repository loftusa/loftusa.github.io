import test from 'node:test';
import assert from 'node:assert/strict';
import * as navigation from '../navigation.mjs';

function fly(position,input={},dt=.05) {
  assert.equal(typeof navigation.advanceFlight,'function','Free flight must replace grounded movement');
  return navigation.advanceFlight(position,{right:0,forward:0,up:0,yaw:0,pitch:0,boost:false,...input},dt);
}
const distance=p=>Math.hypot(...p);
test('flight hovers without gravity and passes through arbitrary ground and wall coordinates',()=>{
  assert.deepEqual(fly([2,30,-3]),[2,30,-3]);
  let position=[0,1.65,1];
  for(let i=0;i<60;i++)position=fly(position,{forward:1});
  assert.ok(position[2]<-6,'Fly through the fixture back wall and former site boundary');
  assert.equal(position[1],1.65);
});
test('forward flight follows the camera pitch and yaw',()=>{
  const up=fly([0,0,0],{forward:1,pitch:Math.PI/2});
  assert.ok(up[1]>.12);assert.ok(Math.abs(up[2])<1e-10);
  const left=fly([0,0,0],{forward:1,yaw:Math.PI/2});
  assert.ok(left[0]<-.12);assert.ok(Math.abs(left[2])<1e-10);
});
test('Space and Control move vertically, and Shift multiplies speed by four',()=>{
  const rise=fly([0,0,0],{up:1,yaw:1,pitch:1});
  assert.deepEqual(rise,[0,.125,0]);
  assert.deepEqual(fly([0,0,0],{up:-1}),[0,-.125,0]);
  assert.equal(distance(fly([0,0,0],{up:1,boost:true})),distance(rise)*4);
});
test('combined flight inputs cannot exceed the selected speed',()=>{
  const straight=distance(fly([0,0,0],{forward:1}));
  assert.ok(Math.abs(distance(fly([0,0,0],{right:1,forward:1,up:1,pitch:.6}))-straight)<1e-10);
});
test('tab suspension cannot teleport a flying camera',()=>{
  assert.deepEqual(fly([0,0,0],{forward:1,boost:true},20),fly([0,0,0],{forward:1,boost:true},.05));
  assert.deepEqual(fly([0,0,0],{forward:1},-1),[0,0,0]);
});

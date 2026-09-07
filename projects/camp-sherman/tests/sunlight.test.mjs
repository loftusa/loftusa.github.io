import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {createSunlightController} from '../sunlight.mjs';

function fixture() {
  const light=new THREE.DirectionalLight();
  return {light,control:createSunlightController(light)};
}
function shadowPosition(light,point) {
  light.updateMatrixWorld();light.target.updateMatrixWorld();light.shadow.updateMatrices(light);
  return new THREE.Vector3(...point).applyMatrix4(light.shadow.matrix);
}
function assertCovered(light,point) {
  const p=shadowPosition(light,point);
  assert.ok(p.x>0&&p.x<1&&p.y>0&&p.y<1&&p.z>0&&p.z<1,`${point}: shadow ${p.toArray()}`);
}

test('garage and shelter walking focuses keep buildings and tall tree crowns inside sunlight shadow capture',()=>{
  const {light,control}=fixture();
  for(const focus of [[62.08975186,2.707289876,2.715292498],[39.817744355,1.464613859,-23.621571644]]) {
    assert.equal(control.update(new THREE.Vector3(...focus),{force:true}),true);
    for(const dx of [-15,15])for(const dz of [-12,12])for(const dy of [-3,10])
      assertCovered(light,[focus[0]+dx,focus[1]+dy,focus[2]+dz]);
    assertCovered(light,[focus[0],focus[1]+32,focus[2]]);
  }
});

test('shadows stay cached for small walking moves and recenter after sixteen meters',()=>{
  const {light,control}=fixture();
  assert.equal(control.update(new THREE.Vector3(2,2,0)),true);
  assert.equal(control.update(new THREE.Vector3(17,2,0)),false);
  assert.deepEqual(light.target.position.toArray(),[2,2,0]);
  assert.equal(control.update(new THREE.Vector3(19,2,0)),true);
  assert.deepEqual(light.target.position.toArray(),[19,2,0]);
  assert.equal(control.update(new THREE.Vector3(19,2,0)),false);
  assert.equal(control.update(new THREE.Vector3(19,2,0),{force:true}),false);
});

test('home keeps detailed shadows while overview grows in stable bounded steps',()=>{
  const {light,control}=fixture(),focus=new THREE.Vector3(2,2.4,0);
  control.update(focus,{orbitDistance:35});assert.equal(light.shadow.camera.right,42);
  assert.equal(control.update(focus,{orbitDistance:36}),false);
  assert.equal(control.update(focus,{orbitDistance:150}),true);
  assert.ok(light.shadow.camera.right>=90&&light.shadow.camera.right<=100);
  for(const point of [[62,0,3],[40,0,-24],[-15,0,15],[62,30,3]])assertCovered(light,point);
  assert.equal(control.update(focus,{orbitDistance:1000}),true);
  assert.equal(light.shadow.camera.right,100);
  const sunDirection=light.position.clone().sub(light.target.position).normalize();
  assert.ok(sunDirection.distanceTo(new THREE.Vector3(-35,32,-45).normalize())<1e-12);
  assert.equal(light.shadow.camera.far,400);
});

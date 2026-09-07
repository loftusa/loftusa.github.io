import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { SceneWorld } from '../world.mjs';
import {advanceWalker} from '../navigation.mjs';

function box(scene,x,y,z,w,h,d,metadata) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshBasicMaterial());
  mesh.position.set(x,y,z); Object.assign(mesh.userData,metadata); scene.add(mesh); return mesh;
}
test('metadata surfaces support walking while roof and scenery do not', () => {
  const scene = new THREE.Group();
  box(scene,0,-.1,0,10,.2,10,{walkable:true,collider:true});
  box(scene,0,3,0,10,.2,10,{layer:'roof',collider:true});
  box(scene,0,.5,0,1,.2,1,{});
  const world = new SceneWorld(scene);
  assert.ok(Math.abs(world.groundAt(0,0,4)) < 1e-7);
  assert.equal(world.groundAt(20,0,4),null);
  assert.equal(world.roofs.length,1);
});
test('vegetation visibility is separate from floors and architecture',()=>{
  const scene=new THREE.Group();box(scene,0,-.1,0,10,.2,10,{walkable:true});
  const tree=box(scene,2,2,0,1,4,1,{layer:'vegetation'});
  const world=new SceneWorld(scene);
  assert.deepEqual(world.vegetation,[tree]);
});
test('walls block the body but floors and actual doorway gaps remain traversable', () => {
  const scene = new THREE.Group();
  box(scene,0,-.1,0,10,.2,10,{walkable:true,collider:true});
  box(scene,0,1.5,-1, .2,3,1,{collider:true});
  box(scene,0,1.5,1, .2,3,1,{collider:true});
  const world = new SceneWorld(scene);
  assert.equal(world.blockedAt([.2,1.65,-1]),true);
  assert.equal(world.blockedAt([0,1.65,0]),false);
  assert.equal(world.blockedAt([2,1.65,0]),false);
});
test('navigation metadata inherited from a parent group applies to exported child meshes', () => {
  const scene = new THREE.Group();
  const parent = new THREE.Group(); parent.userData.walkable = true; scene.add(parent);
  box(parent,0,-.1,0,2,.2,2,{});
  const world = new SceneWorld(scene);
  assert.ok(Math.abs(world.groundAt(0,0,1)) < 1e-7);
});
test('a low ceiling blocks entry even when no vertical wall is nearby', () => {
  const scene = new THREE.Group();
  box(scene,0,-.1,0,10,.2,10,{walkable:true});
  box(scene,0,1.4,0,3,.2,3,{collider:true});
  const world = new SceneWorld(scene);
  assert.equal(world.blockedAt([0,1.65,0]),true);
});
test('a walker climbs successive real stair meshes without being stopped by their risers', () => {
  const scene = new THREE.Group();
  box(scene,0,-.1,0,20,.2,10,{walkable:true,collider:true});
  for (let i=0;i<8;i++) {
    const top=.18*(i+1);
    box(scene,.3+i*.6,top/2,0,.6,top,2,{walkable:true,collider:true});
  }
  const world=new SceneWorld(scene);
  let walker={position:[-.5,1.65,0],verticalSpeed:0};
  for(let i=0;i<120;i++)walker=advanceWalker(walker,{x:1,z:0},.016,world,{min:[-10,-1,-5],max:[10,5,5]});
  assert.ok(walker.position[0]>4.2);
  assert.ok(Math.abs(walker.position[1]-3.09)<.001);
});
test('dense terrain keeps 300 floor queries within the interactive CPU budget',()=>{
  const scene=new THREE.Group();
  const terrain=new THREE.Mesh(new THREE.PlaneGeometry(100,100,350,350),new THREE.MeshBasicMaterial());
  terrain.rotation.x=-Math.PI/2;terrain.userData.walkable=true;scene.add(terrain);
  const world=new SceneWorld(scene);const started=performance.now();
  for(let i=0;i<300;i++)assert.ok(Math.abs(world.groundAt(i/10-15,0,2))<.000001);
  const elapsed=performance.now()-started;
  assert.ok(elapsed<500,`300 queries took ${elapsed.toFixed(0)} ms; dense geometry must use spatial acceleration.`);
});

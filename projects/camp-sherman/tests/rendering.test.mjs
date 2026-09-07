import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {prepareSceneMaterials,createPracticalLights,batchVegetation} from '../rendering.mjs';

test('architectural glass transmits the room and reflects the environment without casting opaque shadows',()=>{
  const root=new THREE.Group();
  const glass=new THREE.MeshStandardMaterial({transparent:true,opacity:.22});glass.name='glass';
  const pane=new THREE.Mesh(new THREE.BoxGeometry(1,1,.01),glass);root.add(pane);
  prepareSceneMaterials(root,8);
  assert.equal(pane.material.isMeshPhysicalMaterial,true);
  assert.ok(pane.material.transmission>.8);
  assert.equal(pane.material.metalness,0);
  assert.ok(pane.material.roughness<=.02,'window glass should remain optically clear');
  assert.equal(pane.castShadow,false);
  assert.equal(pane.material.opacity,1);
});
test('opaque textures preserve authored materials and gain anisotropy for grazing views',()=>{
  const root=new THREE.Group();const texture=new THREE.Texture();
  const material=new THREE.MeshStandardMaterial({map:texture,color:0x765432});material.name='wood';
  const wall=new THREE.Mesh(new THREE.BoxGeometry(),material);root.add(wall);
  prepareSceneMaterials(root,8);
  assert.equal(wall.material,material);assert.equal(texture.anisotropy,8);
  assert.equal(wall.castShadow,true);assert.equal(wall.receiveShadow,true);
});
test('collision proxy meshes remain available for raycasts without being drawn',()=>{
  const root=new THREE.Group();const proxy=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());
  proxy.userData.collision_only=true;root.add(proxy);prepareSceneMaterials(root,4);
  assert.equal(proxy.visible,false);
  assert.equal(root.children.includes(proxy),true);
});
test('practical lighting preserves authored position and uses inverse-square falloff',()=>{
  const lights=createPracticalLights([{position:[1,2,3],color:'#ffe4bc',intensity:4,distance:8}]);
  assert.deepEqual(lights.children[0].position.toArray(),[1,2,3]);
  assert.equal(lights.children[0].intensity,4);assert.equal(lights.children[0].distance,8);
  assert.equal(lights.children[0].decay,2);assert.equal(lights.children[0].castShadow,false);
});
test('repeated decorative trees share a GPU batch with exact world transforms and separate collision proxies',()=>{
  const root=new THREE.Group();root.position.set(3,4,5);root.rotation.y=.3;
  const group=new THREE.Group();group.position.set(1,2,3);root.add(group);
  const geometry=new THREE.BoxGeometry(),material=new THREE.MeshStandardMaterial();
  const originals=[];
  for(let i=0;i<3;i++){
    const tree=new THREE.Mesh(geometry,material);tree.position.set(i*3,0,i);
    tree.rotation.y=i*.3;tree.scale.setScalar(1+i*.2);
    tree.userData={vegetation_instance:true,layer:'vegetation'};tree.castShadow=true;
    group.add(tree);originals.push(tree);
  }
  const proxy=new THREE.Mesh(geometry,material);proxy.userData={collision_only:true,collider:true,layer:'vegetation',vegetation_instance:true};root.add(proxy);
  root.updateMatrixWorld(true);const expected=originals.map(tree=>tree.matrixWorld.clone());
  batchVegetation(root);root.updateMatrixWorld(true);
  const batches=[];root.traverse(node=>{if(node.isInstancedMesh)batches.push(node);});
  assert.equal(batches.length,1);assert.equal(batches[0].count,3);
  assert.equal(batches[0].userData.layer,'vegetation');assert.equal(batches[0].castShadow,true);
  assert.equal(proxy.parent,root);assert.ok(originals.every(tree=>tree.parent===null));
  for(let i=0;i<3;i++){
    const matrix=new THREE.Matrix4();batches[0].getMatrixAt(i,matrix);matrix.premultiply(batches[0].matrixWorld);
    assert.ok(matrix.elements.every((value,index)=>Math.abs(value-expected[i].elements[index])<1e-5));
  }
});

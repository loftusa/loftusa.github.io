import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { ArchitecturalGTAOPass, createPhotographicRenderer } from '../photography.mjs';

function fixture() {
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  const mesh=material=>{const node=new THREE.Mesh(new THREE.BoxGeometry(),material);scene.add(node);return node;};
  const wall=mesh(new THREE.MeshStandardMaterial());
  const glass=mesh(new THREE.MeshPhysicalMaterial({transmission:1}));
  const foliage=mesh(new THREE.MeshStandardMaterial({alphaTest:.5}));
  const roof=mesh(new THREE.MeshStandardMaterial());roof.visible=false;
  const hiddenFoliage=mesh(new THREE.MeshStandardMaterial({alphaTest:.5}));hiddenFoliage.visible=false;
  const line=new THREE.Line();scene.add(line);
  const renderer={
    autoClear:true,color:new THREE.Color(0x123456),alpha:.4,target:null,
    getClearColor(out){return out.copy(this.color);},getClearAlpha(){return this.alpha;},
    setClearColor(value){this.color.set(value);},setClearAlpha(value){this.alpha=value;},
    getRenderTarget(){return this.target;},setRenderTarget(value){this.target=value;},clear(){},
    render(){},
  };
  return {scene,camera,wall,glass,foliage,roof,hiddenFoliage,line,renderer};
}

test('AO omits transparent surfaces only during geometry capture and preserves hidden roofs',()=>{
  const f=fixture(),pass=new ArchitecturalGTAOPass(f.scene,f.camera);
  let geometryCalls=0,postCalls=0;
  f.renderer.render=scene=>{
    if(scene===f.scene){
      geometryCalls++;
      assert.equal(f.wall.visible,true);
      for(const node of [f.glass,f.foliage,f.roof,f.hiddenFoliage,f.line])assert.equal(node.visible,false);
    }else{
      postCalls++;
      for(const node of [f.glass,f.foliage,f.line])assert.equal(node.visible,true);
      for(const node of [f.roof,f.hiddenFoliage])assert.equal(node.visible,false);
    }
  };
  pass.render(f.renderer,new THREE.WebGLRenderTarget(),new THREE.WebGLRenderTarget());
  assert.equal(geometryCalls,1);assert.ok(postCalls>0);pass.dispose();
});

test('failed normal capture restores glass, hidden state, renderer state and scene override',()=>{
  const f=fixture(),pass=new ArchitecturalGTAOPass(f.scene,f.camera);
  const override=new THREE.MeshBasicMaterial(),target={name:'original'};
  f.scene.overrideMaterial=override;f.renderer.target=target;
  f.renderer.render=()=>{throw new Error('capture failed');};
  assert.throws(()=>pass.render(f.renderer,null,null),/capture failed/);
  for(const node of [f.wall,f.glass,f.foliage,f.line])assert.equal(node.visible,true);
  for(const node of [f.roof,f.hiddenFoliage])assert.equal(node.visible,false);
  assert.equal(f.scene.overrideMaterial,override);assert.equal(f.renderer.target,target);
  assert.equal(f.renderer.autoClear,true);assert.equal(f.renderer.alpha,.4);
  assert.equal(f.renderer.color.getHex(),0x123456);pass.dispose();
});

test('AO target size remains bounded at high pixel ratios without changing aspect',()=>{
  const f=fixture(),pass=new ArchitecturalGTAOPass(f.scene,f.camera,{maxAoDimension:1000});
  pass.setSize(3840,2160);
  assert.equal(pass.width,1000);assert.equal(pass.height,563);
  assert.equal(pass.normalRenderTarget.width,1000);assert.equal(pass.pdRenderTarget.height,563);
  pass.setSize(600,300);assert.equal(pass.width,600);assert.equal(pass.height,300);pass.dispose();
});


test('HDR composition retains full-resolution MSAA beauty and outputs glass before display transform',()=>{
  const f=fixture();
  Object.assign(f.renderer,{
    capabilities:{maxSamples:4},getPixelRatio:()=>2,getSize:out=>out.set(800,600),
    toneMapping:THREE.ACESFilmicToneMapping,toneMappingExposure:1,outputColorSpace:THREE.SRGBColorSpace,
  });
  const frames=[];
  f.renderer.render=(scene)=>frames.push({scene,target:f.renderer.target,glass:f.glass.visible});
  const photography=createPhotographicRenderer(f.renderer,f.scene,f.camera,{maxAoDimension:900});
  photography.resize(1000,500,2);photography.render(0);
  const beauty=frames[0];
  assert.equal(beauty.scene,f.scene);assert.equal(beauty.glass,true);
  assert.equal(beauty.target.width,2000);assert.equal(beauty.target.height,1000);
  assert.equal(beauty.target.samples,4);assert.equal(beauty.target.texture.type,THREE.HalfFloatType);
  assert.equal(frames[1].glass,false);assert.equal(frames[1].target.width,900);
  assert.equal(frames.at(-1).target,null);assert.equal(frames.at(-1).glass,true);
  let disposed=false;beauty.target.addEventListener('dispose',()=>{disposed=true;});
  photography.dispose();assert.equal(disposed,true);
});

test('AO disposal releases shader materials omitted by upstream Three r180',()=>{
  const f=fixture(),pass=new ArchitecturalGTAOPass(f.scene,f.camera);
  const disposed=[];
  for(const name of ['gtaoMaterial','blendMaterial','normalMaterial','pdMaterial'])
    pass[name].addEventListener('dispose',()=>disposed.push(name));
  pass.dispose();
  assert.deepEqual(disposed.sort(),['blendMaterial','gtaoMaterial','normalMaterial','pdMaterial']);
});

import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {RGBELoader} from 'three/addons/loaders/RGBELoader.js';
import {MeshoptDecoder} from 'three/addons/libs/meshopt_decoder.module.js';
import {advanceWalker, movementVector, frameSeconds, validateManifest, orbitFramingScale} from './navigation.mjs';
import {SceneWorld} from './world.mjs';
import {prepareSceneMaterials,createPracticalLights,batchVegetation} from './rendering.mjs';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['scene','orbit','walk','roof','trees','reset','viewpoint','loading','loading-title','loading-message','progress','retry','announcement','view-label','walk-hint'].map(id=>[id,$(id)]));
const scene = new THREE.Scene();
scene.background = new THREE.Color('#e8ece3');
const camera = new THREE.PerspectiveCamera(48,1,.06,1200);
camera.rotation.order = 'YXZ';
let renderer, orbit, model, world, manifest, environmentTarget, practicalLights;
let ready=false, loading=false, mode='orbit', roofHidden=false, treesHidden=false, walker, lastGoodPosition;
let previousTime=0;
let renderNeeded=true;
const keys = new Set();
const touchKeys = new Set();
let lookPointer=null;
const light = new THREE.DirectionalLight('#fff0d8',2.5);
scene.add(new THREE.HemisphereLight('#dbe8f1','#697455',.45));
scene.add(light);scene.add(light.target);
light.castShadow=true;
light.shadow.mapSize.set(4096,4096);
light.shadow.bias=-.00008;
light.shadow.normalBias=.012;
light.shadow.radius=3;

function announce(text) { ui.announcement.textContent=text; }
function clearInput() { keys.clear();touchKeys.clear();lookPointer=null; }
function releaseMouse() { if (document.pointerLockElement) document.exitPointerLock(); }
function synchronizeControls() {
  document.body.dataset.mode=mode;
  ui.orbit.setAttribute('aria-pressed',String(mode==='orbit'));
  ui.walk.setAttribute('aria-pressed',String(mode==='walk'));
  orbit.enabled=mode==='orbit';
  const isTouch = matchMedia('(pointer:coarse)').matches || innerWidth <= 760;
  ui['walk-hint'].textContent=isTouch?'Drag to look · Hold arrows to move':'Click the scene to look · WASD to move';
}
function setRoof(hidden) {
  roofHidden=hidden;
  for (const roof of world.roofs) roof.visible=!hidden;
  ui.roof.setAttribute('aria-pressed',String(hidden));
  ui.roof.setAttribute('aria-label',hidden?'Show roof':'Hide roof');
  ui.roof.querySelector('span').textContent=hidden?'Show roof':'Hide roof';
  renderNeeded=true;renderer.shadowMap.needsUpdate=true;
}
function setTrees(hidden) {
  treesHidden=hidden;
  for(const tree of world.vegetation) tree.visible=!hidden && tree.userData.collision_only!==true;
  ui.trees.setAttribute('aria-pressed',String(hidden));
  ui.trees.setAttribute('aria-label',hidden?'Show trees':'Hide trees');
  ui.trees.querySelector('span').textContent=hidden?'Show trees':'Hide trees';
  renderNeeded=true;renderer.shadowMap.needsUpdate=true;
}
function homeView({resetRoof=false}={}) {
  releaseMouse();clearInput();mode='orbit';walker=null;
  camera.position.fromArray(manifest.home.camera);
  orbit.target.fromArray(manifest.home.target);
  camera.position.sub(orbit.target).multiplyScalar(orbitFramingScale(camera.aspect)).add(orbit.target);
  orbit.enabled=true;orbit.update();
  ui.viewpoint.value='';ui['view-label'].textContent='The property';
  if (resetRoof) {setRoof(false);setTrees(false);}
  synchronizeControls();announce('Site view. Drag to orbit the property.');
}
function walkTo(point) {
  releaseMouse();clearInput();mode='walk';
  camera.position.fromArray(point.position);camera.lookAt(new THREE.Vector3(...point.lookAt));
  walker={position:[...point.position],verticalSpeed:0};
  lastGoodPosition=[...point.position];
  ui.viewpoint.value=point.id;ui['view-label'].textContent=point.label;
  synchronizeControls();announce(`${point.label}. Walk with WASD or the direction controls.`);
  renderer.domElement.focus({preventScroll:true});
  renderNeeded=true;
}
function goToViewpoint(point) {
  if(point.mode!=='orbit'){walkTo(point);return;}
  releaseMouse();clearInput();mode='orbit';walker=null;
  camera.position.fromArray(point.position);orbit.target.fromArray(point.lookAt);
  camera.position.sub(orbit.target).multiplyScalar(orbitFramingScale(camera.aspect)).add(orbit.target);
  orbit.enabled=true;orbit.update();ui.viewpoint.value=point.id;
  ui['view-label'].textContent=point.label;synchronizeControls();announce(`${point.label}. Drag to orbit.`);
}
function resize() {
  if (!renderer) return;
  const nextAspect=innerWidth/innerHeight;
  if(orbit&&mode==='orbit')camera.position.sub(orbit.target).multiplyScalar(orbitFramingScale(nextAspect)/orbitFramingScale(camera.aspect)).add(orbit.target);
  camera.aspect=nextAspect;camera.updateProjectionMatrix();
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.setSize(innerWidth,innerHeight);
  renderNeeded=true;
  if (orbit) synchronizeControls();
}
function showError(error) {
  console.error('Unable to open the property:',error);
  ready=false;loading=false;document.body.dataset.ready='false';
  ui.loading.hidden=false;ui.loading.dataset.error='true';ui.retry.hidden=false;ui.progress.hidden=true;
  ui['loading-title'].textContent='The clearing is out of reach';
  ui['loading-message'].textContent=renderer?'The model could not be opened. Please check your connection and try again.':'This browser could not start the 3D view. Try again, or open this page in a browser with WebGL enabled.';
  for (const control of [ui.orbit,ui.walk,ui.roof,ui.trees,ui.reset,ui.viewpoint]) control.disabled=true;
  releaseMouse();announce('The model could not be opened. Try again is available.');
}
function rotateLook(dx,dy) {
  camera.rotation.y-=dx*.0025;
  camera.rotation.x=THREE.MathUtils.clamp(camera.rotation.x-dy*.0025,-1.45,1.45);
  renderNeeded=true;
}
function createRenderer() {
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.95;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate=false;
  const environment=new RoomEnvironment();
  const generator=new THREE.PMREMGenerator(renderer);
  environmentTarget=generator.fromScene(environment,.04);
  scene.environment=environmentTarget.texture;
  scene.environmentIntensity=.7;
  environment.dispose();generator.dispose();
  if (matchMedia('(pointer:coarse)').matches) light.shadow.mapSize.set(2048,2048);
  const canvas=renderer.domElement;
  canvas.tabIndex=0;canvas.setAttribute('aria-label','3D property view. Drag to orbit. In Walk mode, use WASD or arrow keys to move.');
  ui.scene.replaceChildren(canvas);
  orbit=new OrbitControls(camera,canvas);
  orbit.enableDamping=true;orbit.dampingFactor=.07;
  orbit.minDistance=1;orbit.maxPolarAngle=Math.PI*.495;
  orbit.screenSpacePanning=true;
  orbit.addEventListener('change',()=>{renderNeeded=true;});
  canvas.addEventListener('pointerdown',event=>{
    if (!ready || mode!=='walk') return;
    canvas.focus({preventScroll:true});
    if (event.pointerType==='touch' || event.pointerType==='pen') {
      event.preventDefault();
      if (lookPointer===null) {lookPointer={id:event.pointerId,x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);}
    }
  });
  canvas.addEventListener('click',event=>{
    if (!ready || mode!=='walk' || event.pointerType==='touch' || event.pointerType==='pen') return;
    canvas.focus({preventScroll:true});
    if (canvas.requestPointerLock && !document.pointerLockElement) {
      try {const promise=canvas.requestPointerLock();promise?.catch(()=>announce('Use Q and E to turn, or drag on a touch screen.'));}
      catch {announce('Use Q and E to turn.');}
    }
  });
  canvas.addEventListener('pointermove',event=>{
    if (lookPointer?.id!==event.pointerId) return;
    rotateLook(event.clientX-lookPointer.x,event.clientY-lookPointer.y);
    lookPointer.x=event.clientX;lookPointer.y=event.clientY;
  });
  const endLook=event=>{if (lookPointer?.id===event.pointerId) lookPointer=null;};
  canvas.addEventListener('pointerup',endLook);canvas.addEventListener('pointercancel',endLook);
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();showError(new Error('The graphics context was lost.'));});
  resize();
}

function disposeModel() {
  if (!model) return;
  scene.remove(model);
  model.traverse(node=>{
    if (!node.isMesh) return;
    node.geometry.dispose();
    for (const material of Array.isArray(node.material)?node.material:[node.material]) {
      for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
      material.dispose();
    }
  });
  model=null;
}

async function loadEnvironment(path) {
  if(!path)return;
  const texture=await new RGBELoader().loadAsync(path);
  const generator=new THREE.PMREMGenerator(renderer);
  const target=generator.fromEquirectangular(texture);
  texture.dispose();generator.dispose();environmentTarget?.dispose();
  environmentTarget=target;scene.environment=target.texture;
  scene.environmentIntensity=.95;
}

async function loadScene() {
  if (loading) return;
  loading=true;ready=false;
  ui.loading.hidden=false;ui.loading.dataset.error='false';ui.retry.hidden=true;ui.progress.hidden=false;ui.progress.removeAttribute('value');
  ui['loading-title'].textContent='Entering the clearing';ui['loading-message'].textContent='Preparing the house and landscape…';
  try {
    if (!renderer) createRenderer();
    if (renderer.getContext().isContextLost()) {location.reload();return;}
    const response=await fetch('./scene.json',{cache:'no-cache'});
    if (!response.ok) throw new Error(`Scene description returned ${response.status}.`);
    manifest=validateManifest(await response.json());
    const environmentReady=loadEnvironment(manifest.assets.environment).catch(error=>console.warn('Using neutral lighting because the forest environment could not be loaded.',error));
    const gltf=await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(manifest.assets.scene,event=>{
      if (event.lengthComputable) {ui.progress.max=event.total;ui.progress.value=event.loaded;ui['loading-message'].textContent=`Opening the landscape · ${Math.round(event.loaded/event.total*100)}%`;}
      else ui['loading-message'].textContent=`Opening the landscape · ${(event.loaded/1048576).toFixed(1)} MB`;
    });
    await environmentReady;
    prepareSceneMaterials(gltf.scene,Math.min(8,renderer.capabilities.getMaxAnisotropy()));
    batchVegetation(gltf.scene);
    const nextWorld=new SceneWorld(gltf.scene);
    disposeModel();model=gltf.scene;world=nextWorld;
    scene.add(model);
    if(practicalLights)scene.remove(practicalLights);
    practicalLights=createPracticalLights(manifest.lights);scene.add(practicalLights);
    const center=new THREE.Vector3(...manifest.home.target);
    const diagonal=new THREE.Vector3(...manifest.bounds.max).sub(new THREE.Vector3(...manifest.bounds.min)).length();
    const shadowExtent=Math.min(80,Math.max(20,diagonal*.35));
    light.position.copy(center).add(new THREE.Vector3(-45,70,30));light.target.position.copy(center);
    Object.assign(light.shadow.camera,{left:-shadowExtent,right:shadowExtent,top:shadowExtent,bottom:-shadowExtent,near:.5,far:250});
    light.shadow.camera.updateProjectionMatrix();
    renderer.shadowMap.needsUpdate=true;
    orbit.maxDistance=Math.max(diagonal*1.4,100);camera.far=Math.max(1200,diagonal*3);camera.updateProjectionMatrix();
    scene.fog=new THREE.Fog('#e8ece3',diagonal*1.1,diagonal*3);
    ui.viewpoint.replaceChildren(new Option('Choose a viewpoint',''),...manifest.waypoints.map(point=>new Option(point.label,point.id)));
    for (const control of [ui.orbit,ui.walk,ui.roof,ui.trees,ui.reset,ui.viewpoint]) control.disabled=false;
    ui.trees.disabled=world.vegetation.length===0;
    homeView({resetRoof:true});
    ready=true;loading=false;document.body.dataset.ready='true';
    renderer.render(scene,camera);renderNeeded=false;ui.loading.hidden=true;
  } catch(error) {showError(error);}
}

ui.orbit.addEventListener('click',()=>homeView());
ui.walk.addEventListener('click',()=>{if (mode!=='walk') walkTo(manifest.waypoints.find(point=>point.mode!=='orbit'));});
ui.viewpoint.addEventListener('change',()=>{const point=manifest.waypoints.find(p=>p.id===ui.viewpoint.value);if(point)goToViewpoint(point);});
ui.reset.addEventListener('click',()=>homeView({resetRoof:true}));
ui.roof.addEventListener('click',()=>{setRoof(!roofHidden);announce(roofHidden?'Roof hidden. The interior plan is visible.':'Roof shown.');});
ui.trees.addEventListener('click',()=>{setTrees(!treesHidden);announce(treesHidden?'Trees hidden. The architecture is unobstructed.':'Trees shown.');});
ui.retry.addEventListener('click',loadScene);
for (const button of document.querySelectorAll('[data-move]')) {
  button.addEventListener('pointerdown',event=>{event.preventDefault();touchKeys.add(button.dataset.move);button.setPointerCapture(event.pointerId);});
  const stop=()=>touchKeys.delete(button.dataset.move);
  button.addEventListener('pointerup',stop);button.addEventListener('pointercancel',stop);button.addEventListener('lostpointercapture',stop);
  button.addEventListener('keydown',event=>{if (event.key===' '||event.key==='Enter') {event.preventDefault();touchKeys.add(button.dataset.move);}});
  button.addEventListener('keyup',stop);button.addEventListener('blur',stop);
}
document.addEventListener('mousemove',event=>{if (mode==='walk'&&document.pointerLockElement===renderer?.domElement)rotateLook(event.movementX,event.movementY);});
document.addEventListener('pointerlockchange',()=>{document.body.dataset.locked=String(Boolean(document.pointerLockElement));clearInput();});
document.addEventListener('keydown',event=>{
  if (!ready || mode!=='walk' || /^(INPUT|TEXTAREA|SELECT|BUTTON|SUMMARY)$/.test(event.target.tagName)) return;
  const key=event.key.toLowerCase();
  if (['w','a','s','d','arrowup','arrowdown','arrowleft','arrowright','q','e'].includes(key)) {event.preventDefault();keys.add(key);}
});
document.addEventListener('keyup',event=>keys.delete(event.key.toLowerCase()));
window.addEventListener('blur',clearInput);
document.addEventListener('visibilitychange',clearInput);
window.addEventListener('resize',resize);
function animate(time) {
  requestAnimationFrame(animate);
  const dt=frameSeconds(time-previousTime);previousTime=time;
  if (!ready || document.hidden) return;
  if (mode==='orbit') orbit.update();
  else {
    const held=key=>keys.has(key)||touchKeys.has(key);
    const turn=((held('q')?1:0)-(held('e')?1:0))*dt*1.5;
    camera.rotation.y+=turn;
    if(turn)renderNeeded=true;
    const input=movementVector((held('d')||held('arrowright')?1:0)-(held('a')||held('arrowleft')?1:0),(held('w')||held('arrowup')?1:0)-(held('s')||held('arrowdown')?1:0),camera.rotation.y);
    walker=advanceWalker(walker,input,dt,world,manifest.bounds);
    if (walker.position[1]<manifest.bounds.min[1]-5) {
      walker={position:[...lastGoodPosition],verticalSpeed:0};announce('Returned to the last walking surface.');
    } else if (walker.verticalSpeed===0 && !world.blockedAt(walker.position)) lastGoodPosition=[...walker.position];
    if(camera.position.distanceToSquared(new THREE.Vector3(...walker.position))>1e-12)renderNeeded=true;
    camera.position.fromArray(walker.position);
  }
  if(renderNeeded){renderer.render(scene,camera);renderNeeded=false;}
}
requestAnimationFrame(animate);
loadScene();

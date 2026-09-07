import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {RGBELoader} from 'three/addons/loaders/RGBELoader.js';
import {MeshoptDecoder} from 'three/addons/libs/meshopt_decoder.module.js';
import {Sky} from 'three/addons/objects/Sky.js';
import {createPhotographicRenderer} from './photography.mjs';
import {createSunlightController} from './sunlight.mjs';
import {advanceFlight, frameSeconds, validateManifest, orbitFramingScale} from './navigation.mjs';
import {SceneWorld} from './world.mjs';
import {prepareSceneMaterials,createPracticalLights,batchVegetation} from './rendering.mjs';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries(['scene','orbit','walk','enter-house','roof','trees','reset','viewpoint','loading','loading-title','loading-message','progress','retry','announcement','view-label','walk-hint'].map(id=>[id,$(id)]));
const scene = new THREE.Scene();
scene.background = new THREE.Color('#e8ece3');
const camera = new THREE.PerspectiveCamera(48,1,.06,1200);
camera.rotation.order = 'YXZ';
let renderer, photography, orbit, model, world, manifest, environmentTarget, practicalLights;
let ready=false, loading=false, mode='orbit', roofHidden=false, treesHidden=false;
let previousTime=0;
let renderNeeded=true;
const keys = new Set();
const touchKeys = new Set();
let lookPointer=null;
let mouseLook=null, dragDistance=0;
const light = new THREE.DirectionalLight('#ffddb1',3.1);
scene.add(new THREE.HemisphereLight('#cbdbea','#70604b',.28));
scene.add(light);scene.add(light.target);
light.castShadow=true;
light.shadow.mapSize.set(4096,4096);
light.shadow.bias=-.000025;
light.shadow.normalBias=.006;
light.shadow.radius=3;
const sunlight=createSunlightController(light);
function updateSunlight(force=false) {
  const focus=mode==='walk'?camera.position:orbit.target;
  const orbitDistance=mode==='orbit'?camera.position.distanceTo(orbit.target):0;
  if(sunlight.update(focus,{orbitDistance,force})) {
    renderer.shadowMap.needsUpdate=true;renderNeeded=true;
  }
}
const sky=new Sky();sky.scale.setScalar(1500);scene.add(sky);
Object.assign(sky.material.uniforms.turbidity,{value:3.4});
Object.assign(sky.material.uniforms.rayleigh,{value:1.7});
sky.material.uniforms.mieCoefficient.value=.004;
sky.material.uniforms.mieDirectionalG.value=.82;
sky.material.uniforms.sunPosition.value.set(-35,32,-45).normalize();

function announce(text) { ui.announcement.textContent=text; }
function clearInput() { keys.clear();touchKeys.clear();lookPointer=null;mouseLook=null;updateLookHint(); }
function releaseMouse() { if (document.pointerLockElement) document.exitPointerLock(); }
function updateLookHint() {
  const isTouch = matchMedia('(pointer:coarse)').matches;
  ui['walk-hint'].textContent=mouseLook?'Mouse to look · Click/Esc to stop · WASD · Shift faster · Space ↑ · Ctrl ↓':isTouch?'Drag to look · Hold arrows to fly':'Click or drag to look · WASD · Shift faster · Space ↑ · Ctrl ↓';
}
function synchronizeControls() {
  document.body.dataset.mode=mode;
  ui.orbit.setAttribute('aria-pressed',String(mode==='orbit'));
  ui.walk.setAttribute('aria-pressed',String(mode==='walk'));
  orbit.enabled=mode==='orbit';
  updateLookHint();
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
  releaseMouse();clearInput();mode='orbit';
  camera.position.fromArray(manifest.home.camera);
  orbit.target.fromArray(manifest.home.target);
  camera.position.sub(orbit.target).multiplyScalar(orbitFramingScale(camera.aspect)).add(orbit.target);
  orbit.enabled=true;orbit.update();
  ui.viewpoint.value='';ui['view-label'].textContent='The property';
  if (resetRoof) {setRoof(false);setTrees(false);}
  updateSunlight(true);
  synchronizeControls();announce('Site view. Drag to orbit the property.');
}
function walkTo(point) {
  releaseMouse();clearInput();mode='walk';
  camera.position.fromArray(point.position);camera.lookAt(new THREE.Vector3(...point.lookAt));
  ui.viewpoint.value=point.id;ui['view-label'].textContent=point.label;
  updateSunlight(true);
  synchronizeControls();announce(`${point.label}. Fly with WASD. Shift speeds up, Space rises, Control descends.`);
  renderer.domElement.focus({preventScroll:true});
  renderNeeded=true;
}
function enterHouse() {
  const point=manifest.waypoints.find(point=>point.id==='living'&&point.mode!=='orbit')
    ??manifest.waypoints.find(point=>point.mode!=='orbit');
  if(point)walkTo(point);
}
function goToViewpoint(point) {
  if(point.mode!=='orbit'){walkTo(point);return;}
  releaseMouse();clearInput();mode='orbit';
  camera.position.fromArray(point.position);orbit.target.fromArray(point.lookAt);
  camera.position.sub(orbit.target).multiplyScalar(orbitFramingScale(camera.aspect)).add(orbit.target);
  orbit.enabled=true;orbit.update();ui.viewpoint.value=point.id;
  updateSunlight(true);
  ui['view-label'].textContent=point.label;synchronizeControls();announce(`${point.label}. Drag to orbit.`);
}
function resize() {
  if (!renderer) return;
  const nextAspect=innerWidth/innerHeight;
  if(orbit&&mode==='orbit')camera.position.sub(orbit.target).multiplyScalar(orbitFramingScale(nextAspect)/orbitFramingScale(camera.aspect)).add(orbit.target);
  camera.aspect=nextAspect;camera.updateProjectionMatrix();
  renderer.setPixelRatio(Math.min(devicePixelRatio,matchMedia('(pointer:coarse)').matches?1.5:2));
  renderer.setSize(innerWidth,innerHeight);
  photography?.resize(innerWidth,innerHeight,renderer.getPixelRatio());
  renderNeeded=true;
  if (orbit) synchronizeControls();
}
function showError(error) {
  console.error('Unable to open the property:',error);
  ready=false;loading=false;document.body.dataset.ready='false';
  ui.loading.hidden=false;ui.loading.dataset.error='true';ui.retry.hidden=false;ui.progress.hidden=true;
  ui['loading-title'].textContent='The clearing is out of reach';
  ui['loading-message'].textContent=renderer?'The model could not be opened. Please check your connection and try again.':'This browser could not start the 3D view. Try again, or open this page in a browser with WebGL enabled.';
  for (const control of [ui.orbit,ui.walk,ui['enter-house'],ui.roof,ui.trees,ui.reset,ui.viewpoint]) control.disabled=true;
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
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.0;
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
  canvas.tabIndex=0;canvas.setAttribute('aria-label','3D property view. Drag to orbit. In Fly mode, use WASD or arrow keys to move, Shift to speed up, Space to rise and Control to descend.');
  ui.scene.replaceChildren(canvas);
  orbit=new OrbitControls(camera,canvas);
  orbit.enableDamping=true;orbit.dampingFactor=.07;
  orbit.minDistance=1;orbit.maxPolarAngle=Math.PI*.495;
  orbit.screenSpacePanning=true;
  orbit.addEventListener('change',()=>{renderNeeded=true;});
  canvas.addEventListener('pointerdown',event=>{
    if (!ready || mode!=='walk' || event.button!==0) return;
    canvas.focus({preventScroll:true});
    dragDistance=0;
    if (!mouseLook && !document.pointerLockElement && lookPointer===null) {
      if(event.pointerType!=='mouse')event.preventDefault();
      lookPointer={id:event.pointerId,x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);
    }
  });
  canvas.addEventListener('click',event=>{
    if (!ready || mode!=='walk' || event.button!==0 || event.pointerType==='touch' || event.pointerType==='pen' || dragDistance>4) return;
    canvas.focus({preventScroll:true});
    if(mouseLook || document.pointerLockElement===canvas){releaseMouse();clearInput();return;}
    // Keep mouse look usable in embedded browsers that cannot capture the pointer.
    mouseLook={x:event.clientX,y:event.clientY};updateLookHint();
    if (canvas.requestPointerLock && !document.pointerLockElement) {
      try {const promise=canvas.requestPointerLock();promise?.catch(()=>{});}
      catch { /* Click-to-look remains active without pointer capture. */ }
    }
  });
  canvas.addEventListener('pointermove',event=>{
    if(mode!=='walk' || document.pointerLockElement===canvas)return;
    if(mouseLook && event.pointerType==='mouse'){
      rotateLook(event.clientX-mouseLook.x,event.clientY-mouseLook.y);
      mouseLook={x:event.clientX,y:event.clientY};return;
    }
    if (lookPointer?.id!==event.pointerId) return;
    dragDistance+=Math.hypot(event.clientX-lookPointer.x,event.clientY-lookPointer.y);
    rotateLook(event.clientX-lookPointer.x,event.clientY-lookPointer.y);
    lookPointer.x=event.clientX;lookPointer.y=event.clientY;
  });
  const endLook=event=>{if (lookPointer?.id===event.pointerId) lookPointer=null;};
  canvas.addEventListener('pointerup',endLook);canvas.addEventListener('pointercancel',endLook);
  canvas.addEventListener('lostpointercapture',endLook);
  canvas.addEventListener('pointerleave',()=>{mouseLook=null;updateLookHint();});
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();showError(new Error('The graphics context was lost.'));});
  resize();
  const mobile=matchMedia('(pointer:coarse)').matches;
  photography=createPhotographicRenderer(renderer,scene,camera,{aoIntensity:.38,aoRadius:.65,maxAoDimension:mobile?800:1200,samples:mobile?2:4});
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
  scene.environmentIntensity=.3;
  scene.environmentRotation.y=.5;
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
    const diagonal=new THREE.Vector3(...manifest.bounds.max).sub(new THREE.Vector3(...manifest.bounds.min)).length();
    orbit.maxDistance=Math.max(diagonal*1.4,100);camera.far=Math.max(1200,diagonal*3);camera.updateProjectionMatrix();
    scene.fog=new THREE.FogExp2('#cbd3cd',.0017);
    ui.viewpoint.replaceChildren(new Option('Choose a viewpoint',''),...manifest.waypoints.map(point=>new Option(point.label,point.id)));
    for (const control of [ui.orbit,ui.walk,ui['enter-house'],ui.roof,ui.trees,ui.reset,ui.viewpoint]) control.disabled=false;
    ui.trees.disabled=world.vegetation.length===0;
    homeView({resetRoof:true});
    ready=true;loading=false;document.body.dataset.ready='true';
    photography.render();renderNeeded=false;ui.loading.hidden=true;
  } catch(error) {showError(error);}
}

ui.orbit.addEventListener('click',()=>homeView());
ui.walk.addEventListener('click',()=>{if (mode!=='walk') enterHouse();});
ui['enter-house'].addEventListener('click',enterHouse);
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
document.addEventListener('pointerdown',event=>{if(event.target!==renderer?.domElement){mouseLook=null;updateLookHint();}});
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'){releaseMouse();clearInput();return;}
  if (!ready || mode!=='walk' || /^(INPUT|TEXTAREA|SELECT|BUTTON|SUMMARY)$/.test(event.target.tagName)) return;
  const key=event.key.toLowerCase();
  if (['w','a','s','d','arrowup','arrowdown','arrowleft','arrowright','q','e','shift',' ','control'].includes(key)) {event.preventDefault();keys.add(key);}
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
    const next=advanceFlight(camera.position.toArray(),{
      right:(held('d')||held('arrowright')?1:0)-(held('a')||held('arrowleft')?1:0),
      forward:(held('w')||held('arrowup')?1:0)-(held('s')||held('arrowdown')?1:0),
      up:(held(' ')?1:0)-(held('control')?1:0),
      yaw:camera.rotation.y,pitch:camera.rotation.x,boost:held('shift'),
    },dt);
    if(next.some((value,index)=>value!==camera.position.getComponent(index)))renderNeeded=true;
    camera.position.fromArray(next);
  }
  updateSunlight();
  if(renderNeeded){photography.render(dt);renderNeeded=false;}
}
requestAnimationFrame(animate);
loadScene();

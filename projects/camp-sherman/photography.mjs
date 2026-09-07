import { Color, HalfFloatType, Vector2, WebGLRenderTarget } from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';

/** GTAO's normal override cannot represent alpha cutouts or transmissive panes. */
function omitFromNormals(node) {
  if (!node.isMesh) return false;
  const materials=Array.isArray(node.material)?node.material:[node.material];
  return materials.some(material=>material && (
    material.transparent || material.alphaTest>0 || material.alphaHash || material.transmission>0
  ));
}

/** Preserve the authored visibility state even when a GPU render fails. */
export class ArchitecturalGTAOPass extends GTAOPass {
  constructor(scene,camera,{maxAoDimension=1100,aoIntensity=.28,aoRadius=.7}={}) {
    super(scene,camera,1,1);
    if(!Number.isFinite(maxAoDimension)||maxAoDimension<1)throw new RangeError('maxAoDimension must be positive');
    this.maxAoDimension=maxAoDimension;
    this.blendIntensity=aoIntensity;
    this.updateGtaoMaterial({radius:aoRadius,thickness:.4,distanceExponent:1,distanceFallOff:1,samples:16,screenSpaceRadius:false});
    this.updatePdMaterial({lumaPhi:10,depthPhi:2,normalPhi:3,radius:4,samples:16});
    this.savedClearColor=new Color();
  }

  setSize(width,height) {
    const scale=Math.min(1,this.maxAoDimension/Math.max(width,height));
    super.setSize(Math.max(1,Math.round(width*scale)),Math.max(1,Math.round(height*scale)));
  }

  _overrideVisibility() {
    super._overrideVisibility();
    this.scene.traverse(node=>{
      if(node.visible && omitFromNormals(node)) {
        this._visibilityCache.push(node);
        node.visible=false;
      }
    });
  }

  render(renderer,writeBuffer,readBuffer,...rest) {
    const override=this.scene.overrideMaterial;
    const autoClear=renderer.autoClear,clearAlpha=renderer.getClearAlpha();
    const target=renderer.getRenderTarget();
    const shadowAutoUpdate=renderer.shadowMap?.autoUpdate;
    renderer.getClearColor(this.savedClearColor);
    // The beauty pass already updated shadows. Geometry capture needs no second shadow render.
    if(renderer.shadowMap)renderer.shadowMap.autoUpdate=false;
    try {
      super.render(renderer,writeBuffer,readBuffer,...rest);
    } finally {
      this._restoreVisibility();
      this.scene.overrideMaterial=override;
      renderer.autoClear=autoClear;
      renderer.setClearColor(this.savedClearColor);
      renderer.setClearAlpha(clearAlpha);
      renderer.setRenderTarget(target);
      if(renderer.shadowMap)renderer.shadowMap.autoUpdate=shadowAutoUpdate;
    }
  }

  dispose() {
    // Three r180's GTAOPass.dispose omits these two owned materials.
    this.gtaoMaterial.dispose();
    this.blendMaterial.dispose();
    super.dispose();
  }
}

/**
 * Linear HDR beauty + modest contact occlusion + one final display transform.
 * Renderer/camera sizing remains the caller's responsibility; resize takes CSS pixels.
 * @param {import('three').WebGLRenderer} renderer
 * @param {import('three').Scene} scene
 * @param {import('three').Camera} camera
 * @param {{aoIntensity?:number,aoRadius?:number,maxAoDimension?:number,samples?:number}} options
 */
export function createPhotographicRenderer(renderer,scene,camera,options={}) {
  const target=new WebGLRenderTarget(1,1,{
    type:HalfFloatType,
    samples:Math.min(options.samples??4,renderer.capabilities.maxSamples),
  });
  target.texture.name='Photographic HDR beauty';
  const composer=new EffectComposer(renderer,target);
  const beauty=new RenderPass(scene,camera);
  const occlusion=new ArchitecturalGTAOPass(scene,camera,options);
  const output=new OutputPass();
  composer.addPass(beauty);composer.addPass(occlusion);composer.addPass(output);
  const resize=(width,height,pixelRatio=renderer.getPixelRatio())=>{
    if(![width,height,pixelRatio].every(value=>Number.isFinite(value)&&value>0))throw new RangeError('Photographic render dimensions must be positive');
    composer.setPixelRatio(pixelRatio);
    composer.setSize(width,height);
  };
  const size=renderer.getSize(new Vector2());resize(size.x,size.y);
  return {
    render:(deltaTime)=>composer.render(deltaTime),
    resize,
    dispose:()=>{beauty.dispose();occlusion.dispose();output.dispose();composer.dispose();},
  };
}

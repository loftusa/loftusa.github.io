import * as THREE from 'three';

/** A tiny, untextured test room; never used as the published property model. */
export function fixtureGlb() {
  const json = {asset:{version:'2.0'},scene:0,scenes:[{nodes:[]}],nodes:[],meshes:[],materials:[{pbrMetallicRoughness:{baseColorFactor:[.62,.42,.24,1],roughnessFactor:1}}],buffers:[{byteLength:0}],bufferViews:[],accessors:[]};
  const chunks = []; let offset = 0;
  function attribute(array,type,componentType,target,min,max) {
    const bytes = Buffer.from(array.buffer,array.byteOffset,array.byteLength);
    const view = json.bufferViews.push({buffer:0,byteOffset:offset,byteLength:bytes.length,target})-1;
    chunks.push(bytes);offset += bytes.length;
    if (offset%4) {const pad=4-offset%4;chunks.push(Buffer.alloc(pad));offset+=pad;}
    return json.accessors.push({bufferView:view,componentType,count:array.length/(type==='VEC3'?3:1),type,...(min?{min,max}:{})})-1;
  }
  const boxes = [
    {name:'floor',size:[12,.2,12],position:[0,-.1,0],extras:{walkable:true,collider:true}},
    {name:'wall',size:[.2,3,6],position:[2,1.5,0],extras:{collider:true}},
    {name:'back-wall',size:[6,3,.2],position:[0,1.5,-3],extras:{collider:true}},
    {name:'roof',size:[6,.2,6],position:[0,3.1,0],extras:{collider:true,layer:'roof'}},
    {name:'tree',size:[.5,4,.5],position:[-4,2,-2],extras:{layer:'vegetation'}},
  ];
  for (const spec of boxes) {
    const geometry = new THREE.BoxGeometry(...spec.size); geometry.computeBoundingBox();
    const attributes = {POSITION:attribute(geometry.attributes.position.array,'VEC3',5126,34962,geometry.boundingBox.min.toArray(),geometry.boundingBox.max.toArray()),NORMAL:attribute(geometry.attributes.normal.array,'VEC3',5126,34962)};
    const indices=attribute(geometry.index.array,'SCALAR',5123,34963);
    const mesh=json.meshes.push({primitives:[{attributes,indices,material:0}]})-1;
    json.nodes.push({mesh,name:spec.name,translation:spec.position,extras:spec.extras});
    json.scenes[0].nodes.push(json.nodes.length-1);
  }
  json.buffers[0].byteLength=offset;
  let document=Buffer.from(JSON.stringify(json)); document=Buffer.concat([document,Buffer.alloc((4-document.length%4)%4,32)]);
  const binary=Buffer.concat(chunks);
  const header=Buffer.alloc(20); header.writeUInt32LE(0x46546c67,0);header.writeUInt32LE(2,4);header.writeUInt32LE(28+document.length+binary.length,8);header.writeUInt32LE(document.length,12);header.writeUInt32LE(0x4e4f534a,16);
  const binHeader=Buffer.alloc(8);binHeader.writeUInt32LE(binary.length,0);binHeader.writeUInt32LE(0x004e4942,4);
  return Buffer.concat([header,document,binHeader,binary]);
}

export const fixtureManifest = {title:'Camp Sherman',assets:{scene:'./assets/camp-sherman.glb'},home:{camera:[12,9,14],target:[0,0,0]},bounds:{min:[-6,-2,-6],max:[6,10,6]},waypoints:[{id:'entry',label:'Entry',position:[0,1.65,1],lookAt:[0,1.65,-2]},{id:'garden',label:'Garden',position:[-4,1.65,4],lookAt:[0,1.2,0]}]};

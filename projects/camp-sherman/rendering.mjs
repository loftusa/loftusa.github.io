import * as THREE from 'three';

/** Batch repeated scenery while retaining authored transforms and collision meshes. */
export function batchVegetation(root) {
  root.updateMatrixWorld(true);
  const groups=new Map(),inverseRoot=root.matrixWorld.clone().invert();
  root.traverse(node=>{
    if(!node.isMesh||node.isInstancedMesh||node.children.length||node.userData.vegetation_instance!==true)return;
    for(let ancestor=node;ancestor;ancestor=ancestor.parent){
      if(!ancestor.visible||ancestor.userData.collider||ancestor.userData.walkable||ancestor.userData.collision_only)return;
    }
    const matrix=new THREE.Matrix4().multiplyMatrices(inverseRoot,node.matrixWorld);
    if(matrix.determinant()<=0)return;
    const materials=Array.isArray(node.material)?node.material:[node.material];
    const key=[node.geometry.uuid,...materials.map(material=>material.uuid),node.castShadow,node.receiveShadow,node.renderOrder].join(':');
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push({node,matrix});
  });
  for(const entries of groups.values()){
    if(entries.length<2)continue;
    const first=entries[0].node;
    const batch=new THREE.InstancedMesh(first.geometry,first.material,entries.length);
    batch.name=`Vegetation batch: ${first.name}`;
    batch.userData={layer:'vegetation',vegetation_instance:true};
    batch.castShadow=first.castShadow;batch.receiveShadow=first.receiveShadow;batch.renderOrder=first.renderOrder;
    entries.forEach(({node,matrix},index)=>{batch.setMatrixAt(index,matrix);node.removeFromParent();});
    batch.instanceMatrix.needsUpdate=true;batch.computeBoundingBox();batch.computeBoundingSphere();root.add(batch);
  }
}

/** Preserve authored PBR surfaces; render window panes as dielectric glass. */
export function prepareSceneMaterials(root,maxAnisotropy) {
  const replacements=new Map();
  root.traverse(node=>{
    if (node.userData.collision_only===true) node.visible=false;
    if (!node.isMesh) return;
    let hasGlass=false;
    const prepare=material=>{
      for(const value of Object.values(material)) if(value?.isTexture) value.anisotropy=maxAnisotropy;
      // Thin needles lose alpha coverage in distant mip levels.
      if (/conifer.*needles/i.test(material.name) && material.alphaTest>0) {
        material.alphaTest=.12;
        material.alphaToCoverage=true;
      }
      if (!/^glass(?:[_. ]|$)/i.test(material.name)) return material;
      hasGlass=true;
      if (!replacements.has(material)) {
        const glass=new THREE.MeshPhysicalMaterial({
          name:material.name,color:0xffffff,metalness:0,roughness:0,
          transmission:1,thickness:.003,ior:1.45,opacity:1,
          attenuationColor:new THREE.Color('#def0e9'),attenuationDistance:15,
          side:THREE.DoubleSide,envMapIntensity:.7,
        });
        replacements.set(material,glass);
      }
      return replacements.get(material);
    };
    node.material=Array.isArray(node.material)?node.material.map(prepare):prepare(node.material);
    node.castShadow=!hasGlass;
    node.receiveShadow=true;
  });
  for(const original of replacements.keys()) original.dispose();
}

export function createPracticalLights(specifications=[]) {
  const group=new THREE.Group();group.name='Interior practical lighting';
  for(const specification of specifications) {
    const light=new THREE.PointLight(specification.color,specification.intensity,specification.distance,2);
    light.position.fromArray(specification.position);group.add(light);
  }
  return group;
}

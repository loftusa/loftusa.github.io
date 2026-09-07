import * as THREE from 'three';
import {MeshBVH,acceleratedRaycast} from 'three-mesh-bvh';
import { BODY_RADIUS, EYE_HEIGHT } from './navigation.mjs';

function inherited(node,key) {
  for (let current=node;current;current=current.parent) if (current.userData[key] !== undefined) return current.userData[key];
  return undefined;
}

/** Raycast real mesh geometry; bounding boxes only reduce the candidate set. */
export class SceneWorld {
  constructor(root) {
    root.updateMatrixWorld(true);
    this.colliders = []; this.walkable = []; this.roofs = [];this.vegetation=[];
    this.ray = new THREE.Raycaster();
    this.normalMatrix = new THREE.Matrix3();
    this.normal = new THREE.Vector3();
    root.traverse(node => {
      if (node.userData.layer === 'roof') this.roofs.push(node);
      if (node.userData.layer === 'vegetation') this.vegetation.push(node);
      if (!node.isMesh) return;
      const collider=inherited(node,'collider')===true;
      const walkable=inherited(node,'walkable')===true;
      if(collider||walkable) {
        if(!node.geometry.boundsTree)node.geometry.boundsTree=new MeshBVH(node.geometry,{indirect:true});
        node.raycast=acceleratedRaycast;
      }
      const record = {mesh:node,box:new THREE.Box3().setFromObject(node)};
      if (collider) this.colliders.push(record);
      if (walkable) this.walkable.push(record);
    });
    if (!this.walkable.length) throw new Error('The model has no walking surfaces.');
  }

  groundAt(x,z,ceiling) {
    const meshes = this.walkable.filter(({box}) => x >= box.min.x && x <= box.max.x && z >= box.min.z && z <= box.max.z && box.min.y <= ceiling+.01).map(r=>r.mesh);
    this.ray.set(new THREE.Vector3(x,ceiling+.01,z),new THREE.Vector3(0,-1,0));
    this.ray.firstHitOnly=false;
    this.ray.near = 0; this.ray.far = 300;
    for (const hit of this.ray.intersectObjects(meshes,false)) {
      this.normalMatrix.getNormalMatrix(hit.object.matrixWorld);
      this.normal.copy(hit.face.normal).applyMatrix3(this.normalMatrix).normalize();
      if (this.normal.y >= .64) return hit.point.y;
    }
    return null;
  }

  blockedAt(position) {
    const [x,eye,z] = position;
    const feet = eye-EYE_HEIGHT;
    const meshes = this.colliders.filter(({box}) => x+BODY_RADIUS >= box.min.x && x-BODY_RADIUS <= box.max.x && z+BODY_RADIUS >= box.min.z && z-BODY_RADIUS <= box.max.z && eye+.08 >= box.min.y && feet+.36 <= box.max.y).map(r=>r.mesh);
    if (!meshes.length) return false;
    this.ray.firstHitOnly=true;
    this.ray.set(new THREE.Vector3(x,feet+.36,z),new THREE.Vector3(0,1,0));
    this.ray.near=0;this.ray.far=EYE_HEIGHT-.36+.08;
    if (this.ray.intersectObjects(meshes,false).length) return true;
    this.ray.near = 0; this.ray.far = BODY_RADIUS;
    for (const height of [feet+.36,feet+.85,eye+.03]) {
      for (let i=0;i<8;i++) {
        const angle = i*Math.PI/4;
        this.ray.set(new THREE.Vector3(x,height,z),new THREE.Vector3(Math.cos(angle),0,Math.sin(angle)));
        if (this.ray.intersectObjects(meshes,false).length) return true;
      }
    }
    return false;
  }
}

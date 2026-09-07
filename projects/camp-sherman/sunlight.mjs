import {Vector3} from 'three';

const SUN_DIRECTION=new Vector3(-35,32,-45).normalize();
const DETAIL_RADIUS=42,MAX_RADIUS=100,MOVE_THRESHOLD=16;

/** Follow the occupied part of the property without regenerating shadows each frame. */
export function createSunlightController(light) {
  let initialized=false,radius=DETAIL_RADIUS;
  const focus=new Vector3();
  return {
    /** @param {Vector3} nextFocus @param {{orbitDistance?:number,force?:boolean}} options */
    update(nextFocus,{orbitDistance=0,force=false}={}) {
      if(![nextFocus.x,nextFocus.y,nextFocus.z,orbitDistance].every(Number.isFinite)||orbitDistance<0)
        throw new RangeError('Sunlight focus and orbit distance must be finite');
      // Eight-meter steps prevent tiny orbit zoom changes from rebuilding shadows.
      const nextRadius=Math.min(MAX_RADIUS,DETAIL_RADIUS+Math.max(0,Math.ceil((orbitDistance*.6-DETAIL_RADIUS)/8))*8);
      const movement=focus.distanceToSquared(nextFocus);
      if(initialized&&nextRadius===radius&&(movement===0||(!force&&movement<=MOVE_THRESHOLD**2)))return false;
      initialized=true;radius=nextRadius;focus.copy(nextFocus);
      light.target.position.copy(focus);
      light.position.copy(focus).addScaledVector(SUN_DIRECTION,200);
      Object.assign(light.shadow.camera,{left:-radius,right:radius,top:radius,bottom:-radius,near:.5,far:400});
      light.shadow.camera.updateProjectionMatrix();
      return true;
    },
  };
}

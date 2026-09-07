/** Navigation uses meters and Y-up coordinates. Ground queries return a surface Y or null. */
export const EYE_HEIGHT = 1.65;
export const BODY_RADIUS = .28;
export const MAX_STEP = .34;
const SPEED = 2.5;

export function orbitFramingScale(aspect) {
  return Math.max(1,1.2/aspect);
}

export function frameSeconds(milliseconds) {
  return Math.max(0, Math.min(.05, milliseconds / 1000));
}

export function movementVector(right, forward, yaw) {
  const length = Math.max(1, Math.hypot(right, forward));
  return {x: (right*Math.cos(yaw)-forward*Math.sin(yaw))/length,
    z: (-right*Math.sin(yaw)-forward*Math.cos(yaw))/length};
}

/**
 * Free flight in meters: camera-relative WASD, world-vertical rise/descent, no collisions.
 * @param {number[]} position
 * @param {{right:number,forward:number,up:number,yaw:number,pitch:number,boost:boolean}} input
 * @param {number} dt Elapsed seconds, capped for tab suspension.
 * @returns {number[]}
 */
export function advanceFlight(position,{right,forward,up,yaw,pitch,boost},dt) {
  const x=right*Math.cos(yaw)-forward*Math.sin(yaw)*Math.cos(pitch);
  const y=up+forward*Math.sin(pitch);
  const z=-right*Math.sin(yaw)-forward*Math.cos(yaw)*Math.cos(pitch);
  const scale=SPEED*(boost?4:1)*Math.min(.05,Math.max(0,dt))/Math.max(1,Math.hypot(x,y,z));
  return [position[0]+x*scale,position[1]+y*scale,position[2]+z*scale];
}

/** Resolve one short walking frame. Axis separation provides sliding at corners. */
export function advanceWalker(state, movement, dt, world, bounds) {
  dt = Math.min(.05, Math.max(0,dt));
  const position = [...state.position];
  const distance = SPEED*dt;
  const pieces = Math.max(1, Math.ceil(distance/.06));
  for (let i=0;i<pieces;i++) {
    for (const [axis,delta] of [[0,movement.x],[2,movement.z]]) {
      if (!delta) continue;
      const candidate = [...position];
      candidate[axis] = Math.min(bounds.max[axis]-BODY_RADIUS, Math.max(bounds.min[axis]+BODY_RADIUS, candidate[axis]+delta*distance/pieces));
      const feet = position[1]-EYE_HEIGHT;
      const ground = world.groundAt(candidate[0],candidate[2],feet+MAX_STEP);
      if (ground !== null && ground > feet+MAX_STEP+.001) continue;
      if (ground !== null && ground > feet) candidate[1] = ground+EYE_HEIGHT;
      if (!world.blockedAt(candidate)) position.splice(0,3,...candidate);
    }
  }
  let verticalSpeed = state.verticalSpeed-9.81*dt;
  const ground = world.groundAt(position[0],position[2],position[1]-EYE_HEIGHT+MAX_STEP);
  const nextY = position[1]+verticalSpeed*dt;
  if (ground !== null && nextY-EYE_HEIGHT <= ground && ground <= position[1]-EYE_HEIGHT+MAX_STEP) {
    position[1] = ground+EYE_HEIGHT;
    verticalSpeed = 0;
  } else position[1] = nextY;
  return {position, verticalSpeed};
}

export function validateManifest(manifest) {
  const finiteVector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
  if (!manifest || typeof manifest.title !== 'string' || !manifest.title.trim()) throw new Error('The scene title is missing.');
  if (typeof manifest.assets?.scene !== 'string' || !/^\.\/assets\/[a-zA-Z0-9_./-]+\.glb$/.test(manifest.assets.scene) || manifest.assets.scene.includes('..')) throw new Error('The scene asset path is invalid.');
  if (manifest.assets.environment !== undefined && (typeof manifest.assets.environment !== 'string' || !/^\.\/assets\/[a-zA-Z0-9_./-]+\.hdr$/.test(manifest.assets.environment) || manifest.assets.environment.includes('..'))) throw new Error('The environment asset path is invalid.');
  if (!finiteVector(manifest.home?.camera) || !finiteVector(manifest.home?.target)) throw new Error('The opening view is invalid.');
  if (!finiteVector(manifest.bounds?.min) || !finiteVector(manifest.bounds?.max) || manifest.bounds.min.some((v,i) => v >= manifest.bounds.max[i])) throw new Error('The site boundary is invalid.');
  if (!Array.isArray(manifest.waypoints) || !manifest.waypoints.length) throw new Error('The scene has no walking viewpoints.');
  if (manifest.lights!==undefined) {
    if(!Array.isArray(manifest.lights)||manifest.lights.length>32)throw new Error('The practical lights are invalid.');
    for(const light of manifest.lights) if(!finiteVector(light.position)||!/^#[a-fA-F0-9]{6}$/.test(light.color)||!Number.isFinite(light.intensity)||light.intensity<0||light.intensity>100||!Number.isFinite(light.distance)||light.distance<=0)throw new Error('A practical light is invalid.');
  }
  const ids = new Set();
  for (const point of manifest.waypoints) {
    if (!point || typeof point.id !== 'string' || !point.id || ids.has(point.id) || typeof point.label !== 'string' || !point.label.trim() || !finiteVector(point.position) || !finiteVector(point.lookAt)) throw new Error('A viewpoint is invalid.');
    ids.add(point.id);
    if(point.mode!==undefined&&!['walk','orbit'].includes(point.mode))throw new Error('A viewpoint navigation mode is invalid.');
  }
  return manifest;
}

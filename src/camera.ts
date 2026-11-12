import { cross, normalize, sub, type Vec3 } from './math';

/** A pinhole camera described by its position and three unit axes. */
export interface Camera {
  position: Vec3;
  right: Vec3;
  up: Vec3;
  /** Viewing direction. Camera-space depth is measured along this axis. */
  forward: Vec3;
  /** Vertical field of view in radians. */
  fovY: number;
}

/** A camera that orbits a target point; world up is +Y (right-handed, like minity). */
export interface Orbit {
  target: Vec3;
  /** Rotation around the world Y axis, radians. */
  yaw: number;
  /** Elevation above the horizon, radians. */
  pitch: number;
  distance: number;
  fovY: number;
}

export function orbitCamera(o: Orbit): Camera {
  const cp = Math.cos(o.pitch);
  const position: Vec3 = [
    o.target[0] + o.distance * cp * Math.sin(o.yaw),
    o.target[1] + o.distance * Math.sin(o.pitch),
    o.target[2] + o.distance * cp * Math.cos(o.yaw),
  ];
  const forward = normalize(sub(o.target, position));
  const right = normalize(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  return { position, right, up, forward, fovY: o.fovY };
}

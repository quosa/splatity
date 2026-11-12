import { cross, dot, normalize, rotateAround, type Vec3 } from './math';

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

/**
 * A camera that circles a target point like a turntable. `up` is the
 * turntable's axis: +Y for our own scenes (right-handed, like minity), but
 * captured scenes are often tilted, so it can be changed with `roll`.
 */
export interface Orbit {
  target: Vec3;
  /** Unit vector: the axis the camera circles around. */
  up: Vec3;
  /** Unit vector from the target towards the camera. */
  direction: Vec3;
  distance: number;
  /** Vertical field of view in radians. */
  fovY: number;
}

/** Direction for a camera `yaw` radians around +Y and `pitch` radians above the horizon. */
export function directionFromAngles(yaw: number, pitch: number): Vec3 {
  const cp = Math.cos(pitch);
  return [cp * Math.sin(yaw), Math.sin(pitch), cp * Math.cos(yaw)];
}

export function orbitCamera(o: Orbit): Camera {
  const d = o.direction;
  const position: Vec3 = [
    o.target[0] + d[0] * o.distance,
    o.target[1] + d[1] * o.distance,
    o.target[2] + d[2] * o.distance,
  ];
  const forward: Vec3 = [-d[0], -d[1], -d[2]];
  const right = normalize(cross(forward, o.up));
  const up = cross(right, forward);
  return { position, right, up, forward, fovY: o.fovY };
}

/** Moves the camera around the turntable axis (yaw) and over the top (pitch). */
export function turn(o: Orbit, yaw: number, pitch: number): void {
  // Keep a little away from looking straight down or up the axis.
  const LIMIT = 1.5;
  const elevation = Math.asin(Math.max(-1, Math.min(1, dot(o.direction, o.up))));
  const clamped = Math.max(-LIMIT, Math.min(LIMIT, elevation + pitch));
  const right = normalize(cross(o.up, o.direction));
  // Rotating around `right` by a negative angle lifts the camera towards `up`.
  const lifted = rotateAround(o.direction, right, -(clamped - elevation));
  o.direction = normalize(rotateAround(lifted, o.up, yaw));
}

/** Tilts the turntable axis around the viewing direction, to level a scene. */
export function roll(o: Orbit, angle: number): void {
  o.up = normalize(rotateAround(o.up, o.direction, angle));
}

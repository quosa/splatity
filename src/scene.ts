import {
  add, normalize, dot, quatBetween, quatFromAxisAngle, quatMul, rotate,
  type Quat, type Vec3,
} from './math';

/**
 * One gaussian "splat". This is ALL the data a splat scene consists of:
 * there are no triangles, no textures and no lights - just a big list of these.
 *
 * (Trained scenes store colour as spherical harmonics so it can change with the
 * viewing direction; here we keep a single plain RGB colour per splat.)
 */
export interface Splat {
  /** Centre of the gaussian in world space. */
  position: Vec3;
  /** Standard deviation along the splat's three local axes. */
  scale: Vec3;
  /** Orientation of those local axes. */
  rotation: Quat;
  /** Linear RGB, 0..1. */
  color: Vec3;
  /** Peak opacity at the centre, 0..1. */
  opacity: number;
}

/** Small deterministic PRNG so the scene is identical on every load. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hsv(h: number, s: number, v: number): Vec3 {
  const f = (n: number) => {
    const k = (n + h * 6) % 6;
    return v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return [f(5), f(3), f(1)];
}

const LIGHT = normalize([0.5, 0.8, 0.4]);
/** Lighting is "baked" into the splat colours, as it is in a captured scene. */
const shade = (n: Vec3) => 0.3 + 0.7 * Math.max(0, dot(n, LIGHT));

/**
 * Builds a small scene: a checkered floor, a ball, a rainbow torus and a puff
 * of fog. Surfaces are made of thin, pancake-shaped gaussians lying flat on the
 * surface; the fog is made of big, round, nearly transparent ones.
 */
export function createScene(seed = 1): Splat[] {
  const rnd = mulberry32(seed);
  const range = (lo: number, hi: number) => lo + (hi - lo) * rnd();
  const splats: Splat[] = [];

  /** A flat disc-like splat lying on a surface with the given normal. */
  const surfel = (position: Vec3, normal: Vec3, size: number, color: Vec3, opacity: number) => {
    // Local Z is the thin axis: point it along the normal, then spin randomly.
    const facing = quatBetween([0, 0, 1], normal);
    const spin = quatFromAxisAngle([0, 0, 1], range(0, Math.PI));
    const jitter = range(0.94, 1.06);
    splats.push({
      position,
      scale: [size * range(0.7, 1.3), size * range(0.7, 1.3), size * 0.1],
      rotation: quatMul(facing, spin),
      color: [color[0] * jitter, color[1] * jitter, color[2] * jitter],
      opacity,
    });
  };

  // Floor: a disc of radius 3.2 in the XZ plane, fading out towards the rim.
  const floorRadius = 3.2;
  const spacing = 0.12;
  for (let gx = -floorRadius; gx <= floorRadius; gx += spacing) {
    for (let gz = -floorRadius; gz <= floorRadius; gz += spacing) {
      // A jittered grid: evenly spread, but without visible rows.
      const x = gx + range(-0.4, 0.4) * spacing;
      const z = gz + range(-0.4, 0.4) * spacing;
      const r = Math.hypot(x, z);
      if (r > floorRadius) continue;
      const dark = (Math.floor(x) + Math.floor(z)) & 1;
      const tone = dark ? 0.22 : 0.62;
      const fade = Math.min(1, (floorRadius - r) / 0.8);
      surfel([x, 0, z], [0, 1, 0], 0.065, [tone, tone, tone * 1.08], 0.9 * fade);
    }
  }

  // Ball.
  const ballCentre: Vec3 = [-1.0, 0.8, 0.1];
  const ballRadius = 0.8;
  for (let i = 0; i < 1300; i++) {
    const y = range(-1, 1);
    const a = range(0, 2 * Math.PI);
    const s = Math.sqrt(1 - y * y);
    const n: Vec3 = [s * Math.cos(a), y, s * Math.sin(a)];
    const l = shade(n);
    surfel(
      add(ballCentre, [n[0] * ballRadius, n[1] * ballRadius, n[2] * ballRadius]),
      n, 0.055, [1.0 * l, 0.45 * l, 0.12 * l], 0.9,
    );
  }

  // Torus, tilted up on its side. Hue runs around the ring.
  const torusCentre: Vec3 = [1.15, 1.15, 0.0];
  const tilt = quatMul(
    quatFromAxisAngle([0, 1, 0], -0.5),
    quatFromAxisAngle([1, 0, 0], 1.25),
  );
  const R = 0.85;
  const r = 0.28;
  for (let i = 0; i < 1500; ) {
    const u = range(0, 2 * Math.PI);
    const v = range(0, 2 * Math.PI);
    // Rejection sampling so splats are spread evenly over the surface area.
    if (rnd() > (R + r * Math.cos(v)) / (R + r)) continue;
    i++;
    const ring = R + r * Math.cos(v);
    const p = rotate(tilt, [ring * Math.cos(u), r * Math.sin(v), ring * Math.sin(u)]);
    const n = rotate(tilt, [Math.cos(v) * Math.cos(u), Math.sin(v), Math.cos(v) * Math.sin(u)]);
    const c = hsv(u / (2 * Math.PI), 0.75, shade(n));
    surfel(add(torusCentre, p), n, 0.05, c, 0.9);
  }

  // Fog: large, round, barely opaque gaussians. Not a surface at all - this is
  // the kind of thing triangles are bad at and splats get for free.
  for (let i = 0; i < 40; i++) {
    const size = range(0.14, 0.3);
    splats.push({
      position: [range(-0.9, 0.9), range(2.2, 2.7), range(-1.1, 0.3)],
      scale: [size * 1.6, size * 0.7, size * 1.2],
      rotation: quatFromAxisAngle([0, 1, 0], range(0, Math.PI)),
      color: [0.95, 0.85, 0.95],
      opacity: range(0.06, 0.16),
    });
  }

  return splats;
}

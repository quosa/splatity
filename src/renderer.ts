// The gaussian splatting pipeline, written out by hand on the CPU.
//
// For every frame:
//   1. PROJECT  each 3D gaussian becomes a 2D gaussian (an ellipse) on screen
//   2. SORT     the 2D gaussians by depth, nearest first
//   3. BLEND    each one is drawn into every pixel it covers, front to back
//
// It follows the reference rasterizer from "3D Gaussian Splatting for Real-Time
// Radiance Field Rendering" (Kerbl et al. 2023), minus the screen tiles and
// spherical harmonics that are only there for speed and view-dependent colour.

import type { Camera } from './camera';
import { quatToMat3, type Quat } from './math';
import type { Splat } from './scene';

/** Gaussians contributing less than this to a pixel are skipped. */
const MIN_ALPHA = 1 / 255;
/** No single gaussian may fully cover a pixel (keeps the maths stable). */
const MAX_ALPHA = 0.99;
/** Once a pixel lets through less light than this, it is finished. */
const MIN_TRANSMITTANCE = 1e-4;
/** Anything closer to the camera than this is not drawn. */
const NEAR = 0.2;
/** Blur (pixels², added to the 2D variance) so no splat is thinner than a pixel. */
const LOW_PASS = 0.3;
/** Number of depth slices used by the sort. */
const DEPTH_BUCKETS = 65536;

/** Splats packed into flat arrays, with the 3D covariance worked out up front. */
export interface SplatBuffer {
  count: number;
  /** xyz per splat. */
  positions: Float32Array;
  /** Symmetric 3x3 covariance per splat: xx, xy, xz, yy, yz, zz. */
  covariances: Float32Array;
  /** rgb per splat. */
  colors: Float32Array;
  opacities: Float32Array;
}

/**
 * Step 0, done once: turn a splat's scale + rotation into a covariance matrix.
 * With S = diag(scale) and R the rotation matrix,
 *
 *     Σ = R · S · Sᵀ · Rᵀ
 *
 * Σ describes the ellipsoid: how far the gaussian spreads in every direction.
 * Writes the six unique values (xx, xy, xz, yy, yz, zz) to out[offset..].
 */
export function writeCovariance(
  scale: ArrayLike<number>,
  rotation: Quat,
  out: Float32Array,
  offset: number,
): void {
  const r = quatToMat3(rotation);
  // M = R · S  (scale each column of R), so that Σ = M · Mᵀ.
  const m0 = r[0] * scale[0], m1 = r[1] * scale[1], m2 = r[2] * scale[2];
  const m3 = r[3] * scale[0], m4 = r[4] * scale[1], m5 = r[5] * scale[2];
  const m6 = r[6] * scale[0], m7 = r[7] * scale[1], m8 = r[8] * scale[2];
  out[offset] = m0 * m0 + m1 * m1 + m2 * m2; // xx
  out[offset + 1] = m0 * m3 + m1 * m4 + m2 * m5; // xy
  out[offset + 2] = m0 * m6 + m1 * m7 + m2 * m8; // xz
  out[offset + 3] = m3 * m3 + m4 * m4 + m5 * m5; // yy
  out[offset + 4] = m3 * m6 + m4 * m7 + m5 * m8; // yz
  out[offset + 5] = m6 * m6 + m7 * m7 + m8 * m8; // zz
}

export function createSplatBuffer(count: number): SplatBuffer {
  return {
    count,
    positions: new Float32Array(count * 3),
    covariances: new Float32Array(count * 6),
    colors: new Float32Array(count * 3),
    opacities: new Float32Array(count),
  };
}

export function buildSplatBuffer(splats: Splat[]): SplatBuffer {
  const buffer = createSplatBuffer(splats.length);
  splats.forEach((s, i) => {
    writeCovariance(s.scale, s.rotation, buffer.covariances, i * 6);
    buffer.positions.set(s.position, i * 3);
    buffer.colors.set(s.color, i * 3);
    buffer.opacities[i] = s.opacity;
  });
  return buffer;
}

export interface RenderOptions {
  /** Multiplies every splat's size. 1 = as authored, near 0 = a point cloud. */
  splatScale?: number;
  /** Linear RGB shown where no splat covers a pixel. */
  background?: [number, number, number];
  /** Only draw the first this-many splats of the buffer (a quick preview). */
  maxSplats?: number;
}

export interface RenderStats {
  /** Splats that survived culling and were rasterized. */
  drawn: number;
}

export class Renderer {
  readonly width: number;
  readonly height: number;
  /** Accumulated colour per pixel (rgb). */
  private readonly color: Float32Array;
  /** Transmittance per pixel: the fraction of light still passing through. */
  private readonly transmittance: Float32Array;

  // Per-splat results of the projection step, reused between frames.
  private capacity = 0;
  private screenX = new Float32Array(0);
  private screenY = new Float32Array(0);
  private depth = new Float32Array(0);
  /** Inverse 2D covariance ("conic"): a, b, c per splat. */
  private conic = new Float32Array(0);
  private radius = new Float32Array(0);
  private order = new Uint32Array(0);
  private unsorted = new Uint32Array(0);
  private readonly buckets = new Uint32Array(DEPTH_BUCKETS + 1);

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.color = new Float32Array(width * height * 3);
    this.transmittance = new Float32Array(width * height);
  }

  private reserve(count: number): void {
    if (count <= this.capacity) return;
    this.capacity = count;
    this.screenX = new Float32Array(count);
    this.screenY = new Float32Array(count);
    this.depth = new Float32Array(count);
    this.conic = new Float32Array(count * 3);
    this.radius = new Float32Array(count);
    this.order = new Uint32Array(count);
    this.unsorted = new Uint32Array(count);
  }

  /** Renders the splats into `out`, an RGBA byte array of width * height pixels. */
  render(
    splats: SplatBuffer,
    camera: Camera,
    out: Uint8ClampedArray,
    options: RenderOptions = {},
  ): RenderStats {
    const { width, height } = this;
    const sizeSq = (options.splatScale ?? 1) ** 2;
    const background = options.background ?? [0, 0, 0];
    const count = Math.min(splats.count, options.maxSplats ?? splats.count);
    this.reserve(count);

    const { positions, covariances, colors, opacities } = splats;
    const { screenX, screenY, depth, conic, radius, order, unsorted } = this;
    const [camX, camY, camZ] = camera.position;
    const [rx, ry, rz] = camera.right;
    const [ux, uy, uz] = camera.up;
    const [fx, fy, fz] = camera.forward;
    // Focal length in pixels: how many pixels one unit spans at depth 1.
    const focal = height / 2 / Math.tan(camera.fovY / 2);

    // ---- 1. PROJECT -------------------------------------------------------
    let visible = 0;
    let nearest = Infinity;
    let farthest = 0;
    for (let i = 0; i < count; i++) {
      if (opacities[i] < MIN_ALPHA) continue; // invisible anyway
      // Centre in camera space: x right, y up, z = distance in front of camera.
      const dx = positions[i * 3] - camX;
      const dy = positions[i * 3 + 1] - camY;
      const dz = positions[i * 3 + 2] - camZ;
      const x = rx * dx + ry * dy + rz * dz;
      const y = ux * dx + uy * dy + uz * dz;
      const z = fx * dx + fy * dy + fz * dz;
      if (z < NEAR) continue;

      // Perspective projection of the centre. Screen y grows downwards.
      const sx = width / 2 + (focal * x) / z;
      const sy = height / 2 - (focal * y) / z;

      // A gaussian pushed through a perspective projection is no longer a
      // gaussian, so we use the projection's local linear approximation: its
      // Jacobian J at the splat centre. Combined with the camera rotation W
      // (rows: right, up, forward) this gives a 2x3 matrix T = J · W whose two
      // rows say how a small world-space step moves the point on screen.
      const k = focal / z;
      const kx = (focal * x) / (z * z);
      const ky = (focal * y) / (z * z);
      const t0x = k * rx - kx * fx, t0y = k * ry - kx * fy, t0z = k * rz - kx * fz;
      const t1x = ky * fx - k * ux, t1y = ky * fy - k * uy, t1z = ky * fz - k * uz;

      // 2D covariance Σ' = T · Σ · Tᵀ  - the ellipse the ellipsoid casts.
      const c = i * 6;
      const cxx = covariances[c], cxy = covariances[c + 1], cxz = covariances[c + 2];
      const cyy = covariances[c + 3], cyz = covariances[c + 4], czz = covariances[c + 5];
      const v0x = cxx * t0x + cxy * t0y + cxz * t0z;
      const v0y = cxy * t0x + cyy * t0y + cyz * t0z;
      const v0z = cxz * t0x + cyz * t0y + czz * t0z;
      const v1x = cxx * t1x + cxy * t1y + cxz * t1z;
      const v1y = cxy * t1x + cyy * t1y + cyz * t1z;
      const v1z = cxz * t1x + cyz * t1y + czz * t1z;
      const a = sizeSq * (t0x * v0x + t0y * v0y + t0z * v0z) + LOW_PASS;
      const b = sizeSq * (t0x * v1x + t0y * v1y + t0z * v1z);
      const d = sizeSq * (t1x * v1x + t1y * v1y + t1z * v1z) + LOW_PASS;

      // Its inverse (the "conic") is what the per-pixel falloff needs.
      const det = a * d - b * b;
      if (det <= 0) continue;

      // The larger eigenvalue of Σ' is the variance along the ellipse's long
      // axis. Three standard deviations cover 99.7% of the gaussian.
      const mid = (a + d) / 2;
      const lambda = mid + Math.sqrt(Math.max(0, mid * mid - det));
      const rad = Math.ceil(3 * Math.sqrt(lambda));
      if (sx + rad < 0 || sx - rad >= width || sy + rad < 0 || sy - rad >= height) continue;

      screenX[i] = sx;
      screenY[i] = sy;
      depth[i] = z;
      conic[i * 3] = d / det;
      conic[i * 3 + 1] = -b / det;
      conic[i * 3 + 2] = a / det;
      radius[i] = rad;
      unsorted[visible++] = i;
      if (z < nearest) nearest = z;
      if (z > farthest) farthest = z;
    }

    // ---- 2. SORT ----------------------------------------------------------
    // Nearest first. Unlike triangles with a z-buffer, blending transparent
    // things only gives the right picture if it happens in depth order.
    //
    // A counting sort: chop the depth range into many thin slices ("buckets"),
    // count the splats in each, and from the counts work out where each
    // bucket starts in the output. Much faster than a comparison sort for
    // hundreds of thousands of splats; splats sharing a slice stay unordered,
    // which is invisible with slices this thin.
    const buckets = this.buckets;
    buckets.fill(0);
    const toBucket = (DEPTH_BUCKETS - 1) / Math.max(farthest - nearest, 1e-6);
    for (let n = 0; n < visible; n++) {
      buckets[(((depth[unsorted[n]] - nearest) * toBucket) | 0) + 1]++;
    }
    for (let k = 1; k <= DEPTH_BUCKETS; k++) buckets[k] += buckets[k - 1];
    for (let n = 0; n < visible; n++) {
      const i = unsorted[n];
      order[buckets[((depth[i] - nearest) * toBucket) | 0]++] = i;
    }
    const sorted = order;

    // ---- 3. BLEND ---------------------------------------------------------
    const pixels = this.color;
    const transmittance = this.transmittance;
    pixels.fill(0);
    transmittance.fill(1);

    for (let n = 0; n < visible; n++) {
      const i = sorted[n];
      const sx = screenX[i];
      const sy = screenY[i];
      const rad = radius[i];
      const conA = conic[i * 3], conB = conic[i * 3 + 1], conC = conic[i * 3 + 2];
      const opacity = opacities[i];
      const red = colors[i * 3], green = colors[i * 3 + 1], blue = colors[i * 3 + 2];
      // exp(power) * opacity < MIN_ALPHA  <=>  power < this. Saves most exp() calls.
      const minPower = Math.log(MIN_ALPHA / opacity);

      const y0 = Math.max(0, Math.floor(sy - rad));
      const y1 = Math.min(height - 1, Math.ceil(sy + rad));

      for (let py = y0; py <= y1; py++) {
        const oy = py + 0.5 - sy;
        // Which pixels of this row does the ellipse touch? Solving
        // power(ox) = minPower for ox is a quadratic; its two roots are the
        // left and right edge. This only avoids visiting pixels that the
        // alpha test below would reject anyway.
        const half = conB * conB * oy * oy - conA * (conC * oy * oy + 2 * minPower);
        if (half < 0) continue; // the row misses the ellipse
        const mid = sx - (conB * oy) / conA;
        const reach = Math.sqrt(half) / conA;
        const x0 = Math.max(0, Math.floor(mid - reach - 0.5));
        const x1 = Math.min(width - 1, Math.ceil(mid + reach - 0.5));

        for (let px = x0; px <= x1; px++) {
          const p = py * width + px;
          const t = transmittance[p];
          if (t < MIN_TRANSMITTANCE) continue; // already opaque: early out

          // Gaussian falloff: exp(-½ · offsetᵀ · Σ'⁻¹ · offset)
          const ox = px + 0.5 - sx;
          const power = -0.5 * (conA * ox * ox + conC * oy * oy) - conB * ox * oy;
          if (power < minPower) continue;
          const alpha = Math.min(MAX_ALPHA, opacity * Math.exp(power));

          // Front-to-back "over" blending. This splat adds its colour, dimmed
          // by everything in front of it, then dims what is behind it.
          const weight = alpha * t;
          pixels[p * 3] += red * weight;
          pixels[p * 3 + 1] += green * weight;
          pixels[p * 3 + 2] += blue * weight;
          transmittance[p] = t * (1 - alpha);
        }
      }
    }

    // ---- Resolve: whatever light is left shows the background --------------
    for (let p = 0; p < width * height; p++) {
      const t = transmittance[p];
      out[p * 4] = (pixels[p * 3] + t * background[0]) * 255;
      out[p * 4 + 1] = (pixels[p * 3 + 1] + t * background[1]) * 255;
      out[p * 4 + 2] = (pixels[p * 3 + 2] + t * background[2]) * 255;
      out[p * 4 + 3] = 255;
    }

    return { drawn: visible };
  }
}

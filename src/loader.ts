import type { Vec3 } from './math';
import { createSplatBuffer, writeCovariance, type SplatBuffer } from './renderer';

/**
 * Reads a `.splat` file: the simple format introduced by antimatter15's web
 * viewer. There is no header - the file is just one 32-byte record per
 * gaussian, already decoded from the trainer's `.ply` output:
 *
 *   bytes  0-11  position   3 x float32
 *   bytes 12-23  scale      3 x float32  (standard deviation per local axis)
 *   bytes 24-27  colour     4 x uint8    (red, green, blue, opacity)
 *   bytes 28-31  rotation   4 x uint8    (quaternion w, x, y, z; -1..1 as 0..255)
 *
 * Compare with `Splat` in scene.ts: it is exactly the same five properties.
 * Records are stored most important first (biggest and most opaque), so the
 * start of the file is a usable preview of the whole scene.
 */
export function parseSplat(data: ArrayBuffer): SplatBuffer {
  const RECORD = 32;
  const count = Math.floor(data.byteLength / RECORD);
  if (count === 0) throw new Error('Not a .splat file: it is smaller than one 32-byte record.');
  const floats = new Float32Array(data, 0, count * (RECORD / 4));
  const bytes = new Uint8Array(data, 0, count * RECORD);
  const buffer = createSplatBuffer(count);

  for (let i = 0; i < count; i++) {
    const f = i * 8; // 8 floats per record
    const b = i * RECORD;

    // Captured scenes follow the photogrammetry convention of +Y pointing
    // down. Turning everything half a turn around the X axis (y and z change
    // sign) puts it the right way up for our +Y-up world.
    buffer.positions[i * 3] = floats[f];
    buffer.positions[i * 3 + 1] = -floats[f + 1];
    buffer.positions[i * 3 + 2] = -floats[f + 2];

    buffer.colors[i * 3] = bytes[b + 24] / 255;
    buffer.colors[i * 3 + 1] = bytes[b + 25] / 255;
    buffer.colors[i * 3 + 2] = bytes[b + 26] / 255;
    buffer.opacities[i] = bytes[b + 27] / 255;

    const w = (bytes[b + 28] - 128) / 128;
    const x = (bytes[b + 29] - 128) / 128;
    const y = (bytes[b + 30] - 128) / 128;
    const z = (bytes[b + 31] - 128) / 128;
    const norm = Math.hypot(w, x, y, z) || 1;
    // The same half turn applied to the orientation: multiplying by the
    // quaternion (x=1, y=0, z=0, w=0) from the left maps (x, y, z, w) to
    // (w, -z, y, -x).
    writeCovariance(
      floats.subarray(f + 3, f + 6),
      [w / norm, -z / norm, y / norm, -x / norm],
      buffer.covariances,
      i * 6,
    );
  }
  return buffer;
}

/**
 * Where to point the camera: the middle of the scene and roughly how big it
 * is. Ignores the outermost 5% on each side, because captures have stray
 * "floater" splats far away from everything else.
 */
export function frameScene(splats: SplatBuffer): { centre: Vec3; radius: number } {
  const step = Math.max(1, Math.floor(splats.count / 20000)); // a sample is plenty
  const centre: Vec3 = [0, 0, 0];
  let radius = 0;
  for (let axis = 0; axis < 3; axis++) {
    const values: number[] = [];
    for (let i = 0; i < splats.count; i += step) values.push(splats.positions[i * 3 + axis]);
    values.sort((p, q) => p - q);
    const low = values[Math.floor((values.length - 1) * 0.05)];
    const high = values[Math.floor((values.length - 1) * 0.95)];
    centre[axis] = (low + high) / 2;
    radius = Math.max(radius, (high - low) / 2);
  }
  return { centre, radius };
}

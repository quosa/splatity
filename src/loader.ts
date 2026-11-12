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
    store(
      buffer, i,
      floats[f], floats[f + 1], floats[f + 2],
      floats[f + 3], floats[f + 4], floats[f + 5],
      bytes[b + 24] / 255, bytes[b + 25] / 255, bytes[b + 26] / 255, bytes[b + 27] / 255,
      (bytes[b + 28] - 128) / 128, (bytes[b + 29] - 128) / 128,
      (bytes[b + 30] - 128) / 128, (bytes[b + 31] - 128) / 128,
    );
  }
  return buffer;
}

/** Bytes taken by each PLY number type. */
const PLY_TYPES: Record<string, number> = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

/**
 * Reads a `.ply` file as written by the original 3D gaussian splatting
 * trainer and by most capture apps. This is the trainer's own output, before
 * anything is simplified.
 *
 * A PLY file starts with a few lines of text that describe the table that
 * follows: how many rows ("element vertex 123456") and which columns, in
 * order ("property float x"). After the line "end_header" come the rows as
 * raw binary numbers. For a splat scene each row is one gaussian:
 *
 *   x, y, z                position
 *   scale_0..2             LOG of the size along each local axis
 *   rot_0..3               rotation quaternion (w, x, y, z), not normalised
 *   opacity                opacity BEFORE the sigmoid squashes it into 0..1
 *   f_dc_0..2              base colour, as a spherical-harmonics coefficient
 *   f_rest_0..44           further coefficients: how colour changes with the
 *                          viewing direction (ignored here)
 *   nx, ny, nz             unused
 *
 * Sizes and opacities are stored the roundabout way for the reason given in
 * fit2d.ts: these are the numbers the optimiser works on, saved as they are.
 */
export function parsePly(data: ArrayBuffer): SplatBuffer {
  // The header is plain text; find where it ends.
  const bytes = new Uint8Array(data);
  const END = 'end_header\n';
  const headerText = new TextDecoder('latin1').decode(bytes.subarray(0, Math.min(bytes.length, 65536)));
  const headerEnd = headerText.indexOf(END);
  if (!headerText.startsWith('ply') || headerEnd < 0) throw new Error('Not a PLY file.');
  const lines = headerText.slice(0, headerEnd).split(/\r?\n/);
  if (!lines.includes('format binary_little_endian 1.0')) {
    throw new Error('Only binary little-endian PLY files are supported.');
  }

  // Work out the byte offset of every column of the "vertex" table.
  let count = 0;
  let rowSize = 0;
  let inVertex = false;
  const columns = new Map<string, { offset: number; type: string }>();
  for (const line of lines) {
    const words = line.trim().split(/\s+/);
    if (words[0] === 'element') {
      // The vertex table comes first; anything after it does not concern us.
      if (inVertex) break;
      inVertex = words[1] === 'vertex';
      if (inVertex) count = Number(words[2]);
    } else if (words[0] === 'property' && inVertex) {
      const size = PLY_TYPES[words[1]];
      if (!size) throw new Error(`Unsupported PLY property type "${words[1]}".`);
      columns.set(words[2], { offset: rowSize, type: words[1] });
      rowSize += size;
    }
  }
  const start = headerEnd + END.length;
  if (!count || start + count * rowSize > data.byteLength) {
    throw new Error('The PLY file is shorter than its header says.');
  }

  // A reader for one column: given a row number, returns that column's value.
  const view = new DataView(data, start);
  const column = (name: string): ((row: number) => number) => {
    const found = columns.get(name);
    if (!found) throw new Error(`Not a gaussian splat PLY: it has no "${name}" property.`);
    const { offset, type } = found;
    if (type === 'float' || type === 'float32') return (row) => view.getFloat32(row * rowSize + offset, true);
    if (type === 'double' || type === 'float64') return (row) => view.getFloat64(row * rowSize + offset, true);
    throw new Error(`Expected "${name}" to be a float, not ${type}.`);
  };
  const [x, y, z] = ['x', 'y', 'z'].map(column);
  const [s0, s1, s2] = ['scale_0', 'scale_1', 'scale_2'].map(column);
  const [r0, r1, r2, r3] = ['rot_0', 'rot_1', 'rot_2', 'rot_3'].map(column);
  const [red, green, blue] = ['f_dc_0', 'f_dc_1', 'f_dc_2'].map(column);
  const opacity = column('opacity');

  const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v));
  // The first spherical-harmonics function is the constant 0.28209..., and
  // the trainer stores colours shifted by a half, so this recovers plain RGB.
  const SH_C0 = 0.28209479177387814;
  const channel = (v: number): number => Math.max(0, Math.min(1, 0.5 + SH_C0 * v));

  // The trainer saves gaussians in no useful order. The viewer's preview
  // draws only the start of the buffer while the camera moves, so put the
  // most important first: the biggest and most opaque. (A .splat file is
  // made from a .ply by exactly this sort, plus the decoding below.)
  const importance = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    importance[row] = Math.exp(s0(row) + s1(row) + s2(row)) * sigmoid(opacity(row));
  }
  const order = new Uint32Array(count).map((_, i) => i).sort((a, b) => importance[b] - importance[a]);

  const buffer = createSplatBuffer(count);
  for (let i = 0; i < count; i++) {
    const row = order[i];
    store(
      buffer, i,
      x(row), y(row), z(row),
      Math.exp(s0(row)), Math.exp(s1(row)), Math.exp(s2(row)),
      channel(red(row)), channel(green(row)), channel(blue(row)), sigmoid(opacity(row)),
      r0(row), r1(row), r2(row), r3(row),
    );
  }
  return buffer;
}

/** Reads a splat scene, choosing the format from the file's first bytes. */
export function parseScene(data: ArrayBuffer): SplatBuffer {
  const magic = new TextDecoder('latin1').decode(new Uint8Array(data, 0, Math.min(4, data.byteLength)));
  return magic.startsWith('ply') ? parsePly(data) : parseSplat(data);
}

const scratchScale = new Float32Array(3);

/**
 * Stores one decoded gaussian in slot `i` of the buffer. The quaternion is
 * given as (w, x, y, z) and need not be normalised.
 */
function store(
  buffer: SplatBuffer, i: number,
  x: number, y: number, z: number,
  sx: number, sy: number, sz: number,
  red: number, green: number, blue: number, opacity: number,
  qw: number, qx: number, qy: number, qz: number,
): void {
  // Captured scenes follow the photogrammetry convention of +Y pointing
  // down. Turning everything half a turn around the X axis (y and z change
  // sign) puts it the right way up for our +Y-up world.
  buffer.positions[i * 3] = x;
  buffer.positions[i * 3 + 1] = -y;
  buffer.positions[i * 3 + 2] = -z;
  buffer.colors[i * 3] = red;
  buffer.colors[i * 3 + 1] = green;
  buffer.colors[i * 3 + 2] = blue;
  buffer.opacities[i] = opacity;

  const norm = Math.hypot(qw, qx, qy, qz) || 1;
  scratchScale[0] = sx;
  scratchScale[1] = sy;
  scratchScale[2] = sz;
  // The same half turn applied to the orientation: multiplying by the
  // quaternion (x=1, y=0, z=0, w=0) from the left maps (x, y, z, w) to
  // (w, -z, y, -x).
  writeCovariance(
    scratchScale,
    [qw / norm, -qz / norm, qy / norm, -qx / norm],
    buffer.covariances,
    i * 6,
  );
}

/**
 * Where to point the camera: the middle of the scene and roughly how big its
 * main subject is.
 *
 * The bounding box is useless for this. A capture of something outdoors also
 * contains the far background and sky, spread over an area many times larger
 * than the subject, plus stray "floater" splats. But the trainer spends most
 * splats where the photos show detail, so the subject is where splats are
 * dense: the median position is its middle, and the distance within which
 * three quarters of all splats lie is a good measure of its size.
 *
 * `surrounded` says whether there is such a far background at all. If so, the
 * photos were taken from in amongst the scene, and it only looks right from
 * roughly where the photographer stood: further out, the camera ends up behind
 * blobs that no photo ever constrained.
 */
export function frameScene(splats: SplatBuffer): {
  centre: Vec3;
  radius: number;
  surrounded: boolean;
} {
  const step = Math.max(1, Math.floor(splats.count / 20000)); // a sample is plenty
  const { positions } = splats;
  const quantile = (values: number[], fraction: number): number => {
    values.sort((p, q) => p - q);
    return values[Math.floor((values.length - 1) * fraction)];
  };

  const centre: Vec3 = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) {
    const values: number[] = [];
    for (let i = 0; i < splats.count; i += step) values.push(positions[i * 3 + axis]);
    centre[axis] = quantile(values, 0.5);
  }

  const distances: number[] = [];
  for (let i = 0; i < splats.count; i += step) {
    distances.push(Math.hypot(
      positions[i * 3] - centre[0],
      positions[i * 3 + 1] - centre[1],
      positions[i * 3 + 2] - centre[2],
    ));
  }
  const radius = quantile(distances, 0.75);
  return { centre, radius, surrounded: quantile(distances, 0.95) > 3 * radius };
}

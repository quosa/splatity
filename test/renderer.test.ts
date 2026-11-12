import { describe, expect, it } from 'vitest';
import { directionFromAngles, orbitCamera, roll, turn, type Orbit } from '../src/camera';
import { frameScene, parsePly, parseScene, parseSplat } from '../src/loader';
import { quatBetween, quatFromAxisAngle, quatToMat3, rotate } from '../src/math';
import { buildSplatBuffer, Renderer } from '../src/renderer';
import type { Splat } from '../src/scene';

const SIZE = 101;
const CENTRE = 50;
// Camera at (0, 0, 5) looking down -Z at the origin.
const orbit = (): Orbit => ({
  target: [0, 0, 0],
  up: [0, 1, 0],
  direction: directionFromAngles(0, 0),
  distance: 5,
  fovY: Math.PI / 2,
});
const camera = orbitCamera(orbit());
const FOCAL = SIZE / 2; // tan(45°) = 1

function draw(splats: Splat[]) {
  const out = new Uint8ClampedArray(SIZE * SIZE * 4);
  const stats = new Renderer(SIZE, SIZE).render(buildSplatBuffer(splats), camera, out);
  const pixel = (x: number, y: number) => [...out.subarray((y * SIZE + x) * 4, (y * SIZE + x) * 4 + 3)];
  return { pixel, stats };
}

const splat = (over: Partial<Splat>): Splat => ({
  position: [0, 0, 0],
  scale: [0.5, 0.5, 0.5],
  rotation: [0, 0, 0, 1],
  color: [1, 1, 1],
  opacity: 1,
  ...over,
});

describe('math', () => {
  it('turns a quaternion into an orthonormal rotation matrix', () => {
    const m = quatToMat3(quatFromAxisAngle([1, 2, 3], 0.7));
    const row = (i: number) => [m[i * 3], m[i * 3 + 1], m[i * 3 + 2]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const d = row(i).reduce((s, v, k) => s + v * row(j)[k], 0);
        expect(d).toBeCloseTo(i === j ? 1 : 0, 6);
      }
    }
  });

  it('rotates one vector onto another', () => {
    const v = rotate(quatBetween([0, 0, 1], [0, 1, 0]), [0, 0, 1]);
    expect(v[0]).toBeCloseTo(0);
    expect(v[1]).toBeCloseTo(1);
    expect(v[2]).toBeCloseTo(0);
    const flipped = rotate(quatBetween([0, 0, 1], [0, 0, -1]), [0, 0, 1]);
    expect(flipped[2]).toBeCloseTo(-1);
  });
});

describe('renderer', () => {
  it('projects a round gaussian to the expected size on screen', () => {
    const { pixel } = draw([splat({})]);
    // Peak alpha is capped at 0.99 over a black background.
    expect(pixel(CENTRE, CENTRE)[0]).toBeGreaterThanOrEqual(251);
    // One standard deviation away: sigma = focal * scale / depth pixels.
    const sigma = (FOCAL * 0.5) / 5;
    const expected = 255 * Math.exp(-0.5 * (sigma * sigma) / (sigma * sigma + 0.3));
    const x = Math.round(CENTRE + sigma);
    const exact = 255 * Math.exp((-0.5 * (x - CENTRE) ** 2) / (sigma * sigma + 0.3));
    expect(pixel(x, CENTRE)[0]).toBeCloseTo(exact, -1);
    expect(Math.abs(exact - expected)).toBeLessThan(20);
    // Symmetric, and gone well outside three sigma.
    expect(pixel(CENTRE, x)[0]).toBe(pixel(x, CENTRE)[0]);
    expect(pixel(CENTRE + 30, CENTRE)[0]).toBe(0);
  });

  it('stretches along the splat\'s long axis, following its rotation', () => {
    const long = splat({ scale: [1, 0.1, 0.1] });
    const horizontal = draw([long]).pixel;
    expect(horizontal(CENTRE + 10, CENTRE)[0]).toBeGreaterThan(100);
    expect(horizontal(CENTRE, CENTRE + 10)[0]).toBe(0);
    const turned = draw([{ ...long, rotation: quatFromAxisAngle([0, 0, 1], Math.PI / 2) }]).pixel;
    expect(turned(CENTRE + 10, CENTRE)[0]).toBe(0);
    expect(turned(CENTRE, CENTRE + 10)[0]).toBeGreaterThan(100);
  });

  it('blends in depth order regardless of the order in the data', () => {
    const red = splat({ position: [0, 0, 1], color: [1, 0, 0], opacity: 0.5 });
    const blue = splat({ position: [0, 0, -1], color: [0, 0, 1], opacity: 0.5 });
    for (const scene of [[red, blue], [blue, red]]) {
      const [r, , b] = draw(scene).pixel(CENTRE, CENTRE);
      expect(r).toBeCloseTo(127, -1); // nearer: 0.5
      expect(b).toBeCloseTo(63, -1); // farther: 0.5 * (1 - 0.5)
    }
  });

  it('culls splats behind the camera', () => {
    expect(draw([splat({ position: [0, 0, 6] })]).stats.drawn).toBe(0);
    expect(draw([splat({})]).stats.drawn).toBe(1);
  });
});

describe('camera', () => {
  it('looks at the target with an orthonormal basis', () => {
    expect(camera.position).toEqual([0, 0, 5]);
    expect(camera.forward).toEqual([-0, -0, -1]);
    expect(camera.right[0]).toBeCloseTo(1);
    expect(camera.up[1]).toBeCloseTo(1);
  });

  it('turns around the up axis and lifts towards it', () => {
    const o = orbit();
    turn(o, Math.PI / 2, 0);
    expect(o.direction[0]).toBeCloseTo(1);
    expect(o.direction[2]).toBeCloseTo(0);
    turn(o, 0, 0.5);
    expect(Math.asin(o.direction[1])).toBeCloseTo(0.5);
    turn(o, 0, 10); // clamped short of straight overhead
    expect(Math.asin(o.direction[1])).toBeCloseTo(1.5);
  });

  it('rolls the up axis around the viewing direction', () => {
    const o = orbit();
    roll(o, Math.PI / 2);
    expect(Math.abs(o.up[0])).toBeCloseTo(1);
    expect(o.up[1]).toBeCloseTo(0);
    expect(o.direction).toEqual(directionFromAngles(0, 0));
  });
});

describe('.splat loader', () => {
  /** One 32-byte record. */
  function record(position: number[], scale: number[], rgba: number[], wxyz: number[]): Uint8Array {
    const bytes = new Uint8Array(32);
    new Float32Array(bytes.buffer, 0, 6).set([...position, ...scale]);
    bytes.set(rgba, 24);
    bytes.set(wxyz.map((v) => v * 128 + 128), 28);
    return bytes;
  }

  it('decodes records and turns the scene the right way up', () => {
    const file = new Uint8Array(64);
    file.set(record([1, 2, 3], [0.5, 0.1, 0.1], [255, 0, 51, 255], [0.99, 0, 0, 0]), 0);
    file.set(record([-1, -2, -3], [0.1, 0.1, 0.1], [0, 255, 0, 128], [0.99, 0, 0, 0]), 32);
    const s = parseSplat(file.buffer);
    expect(s.count).toBe(2);
    expect([...s.positions.subarray(0, 3)]).toEqual([1, -2, -3]);
    expect(s.colors[0]).toBe(1);
    expect(s.colors[2]).toBeCloseTo(0.2);
    expect(s.opacities[1]).toBeCloseTo(0.5, 2);
    // Long axis stays along x after the half turn around x; variance = scale².
    expect(s.covariances[0]).toBeCloseTo(0.25);
    expect(s.covariances[3]).toBeCloseTo(0.01);
    expect(s.covariances[5]).toBeCloseTo(0.01);
  });

  it('flips a tilted splat consistently with its position', () => {
    // Long axis rotated 90° around z, so it points along the file's +y ...
    const h = Math.SQRT1_2 * 0.99;
    const s = parseSplat(record([0, 0, 0], [0.5, 0.1, 0.1], [255, 255, 255, 255], [h, 0, 0, h]).slice().buffer);
    // ... which is still the y axis (now pointing the other way) after loading.
    expect(s.covariances[0]).toBeCloseTo(0.01, 3);
    expect(s.covariances[3]).toBeCloseTo(0.25, 2);
  });

  it('frames the dense subject, not the far background', () => {
    // A subject of 300 splats within 1 unit of (5, 2, 0) ...
    const scene = Array.from({ length: 300 }, (_, i) =>
      splat({ position: [5 + Math.cos(i) * (i / 300), 2 + Math.sin(i) * (i / 300), 0] }));
    // ... and 60 background splats up to 600 units away.
    for (let i = 1; i <= 60; i++) scene.push(splat({ position: [i * 10, -i * 5, i * 7] }));
    const subject = frameScene(buildSplatBuffer(scene.slice(0, 300)));
    expect(subject.surrounded).toBe(false);
    const { centre, radius, surrounded } = frameScene(buildSplatBuffer(scene));
    expect(surrounded).toBe(true);
    expect(Math.hypot(centre[0] - 5, centre[1] - 2, centre[2])).toBeLessThan(0.5);
    expect(radius).toBeGreaterThan(0.5);
    expect(radius).toBeLessThan(1.5);
  });

  it('rejects files that are too small', () => {
    expect(() => parseSplat(new ArrayBuffer(10))).toThrow();
  });
});

describe('.ply loader', () => {
  const COLUMNS = [
    'x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'f_rest_0',
    'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
  ];
  type Row = Partial<Record<(typeof COLUMNS)[number], number>>;

  /** A gaussian splat PLY with one uchar column in front, to test offsets. */
  function ply(rows: Row[], format = 'binary_little_endian 1.0'): ArrayBuffer {
    const header =
      `ply\nformat ${format}\ncomment made by a test\nelement vertex ${rows.length}\n` +
      'property uchar flag\n' +
      COLUMNS.map((name) => `property float ${name}\n`).join('') +
      'element face 0\nproperty list uchar int vertex_indices\nend_header\n';
    const rowSize = 1 + COLUMNS.length * 4;
    const bytes = new Uint8Array(header.length + rows.length * rowSize);
    bytes.set(new TextEncoder().encode(header));
    const view = new DataView(bytes.buffer, header.length);
    rows.forEach((row, r) => {
      COLUMNS.forEach((name, c) => view.setFloat32(r * rowSize + 1 + c * 4, row[name] ?? 0, true));
    });
    return bytes.buffer;
  }

  const small: Row = { x: 1, y: 2, z: 3, scale_0: Math.log(0.5), scale_1: Math.log(0.1), scale_2: Math.log(0.1), rot_0: 2 };
  const big: Row = { x: -1, scale_0: 1, scale_1: 1, scale_2: 1, rot_0: 1, opacity: 2, f_dc_0: 1, f_dc_1: -10, f_dc_2: 0 };

  it('decodes log sizes, squashed opacity and colour coefficients', () => {
    const s = parsePly(ply([small]));
    expect(s.count).toBe(1);
    expect([...s.positions]).toEqual([1, -2, -3]);
    expect(s.covariances[0]).toBeCloseTo(0.25); // exp(log 0.5)², quaternion normalised
    expect(s.covariances[3]).toBeCloseTo(0.01);
    expect(s.opacities[0]).toBeCloseTo(0.5); // sigmoid(0)
    expect(s.colors[0]).toBeCloseTo(0.5); // a coefficient of 0 is mid grey
  });

  it('puts the biggest, most opaque gaussians first', () => {
    const s = parsePly(ply([small, big, small]));
    expect(s.positions[0]).toBe(-1);
    expect(s.opacities[0]).toBeCloseTo(1 / (1 + Math.exp(-2)));
    expect(s.colors[0]).toBeCloseTo(0.5 + 0.28209479177387814);
    expect(s.colors[1]).toBe(0); // clamped
    expect(s.positions[3]).toBe(1);
  });

  it('is picked automatically by its first bytes', () => {
    expect(parseScene(ply([small, big])).count).toBe(2);
    expect(parseScene(new ArrayBuffer(64)).count).toBe(2); // two .splat records
  });

  it('rejects files it cannot read', () => {
    expect(() => parsePly(new ArrayBuffer(64))).toThrow('Not a PLY');
    expect(() => parsePly(ply([small], 'ascii 1.0'))).toThrow('binary');
    expect(() => parsePly(ply([small, small]).slice(0, -10))).toThrow('shorter');
    const mesh = new TextEncoder().encode(
      'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n\0\0\0\0');
    expect(() => parsePly(mesh.buffer)).toThrow('no "y" property');
  });
});

describe('renderer options', () => {
  it('does not smear a splat that is far outside the view across the screen', () => {
    // Close to the camera and far off to the side: without clamping the
    // Jacobian this projects to a streak thousands of pixels long.
    const { pixel } = draw([splat({ position: [30, 0, 4], scale: [0.2, 0.2, 0.2] })]);
    expect(pixel(CENTRE, CENTRE)[0]).toBe(0);
    expect(pixel(SIZE - 1, CENTRE)[0]).toBe(0);
  });

  it('draws only the first maxSplats, and sorts many splats by depth', () => {
    const many = Array.from({ length: 200 }, (_, i) =>
      splat({ position: [0, 0, -4 + i * 0.03], color: [i === 199 ? 1 : 0, 0, 0], opacity: 0.9 }));
    const out = new Uint8ClampedArray(SIZE * SIZE * 4);
    const renderer = new Renderer(SIZE, SIZE);
    const buffer = buildSplatBuffer(many);
    // The last splat is the nearest and red: it must end up on top.
    renderer.render(buffer, camera, out);
    expect(out[(CENTRE * SIZE + CENTRE) * 4]).toBeGreaterThan(220);
    expect(renderer.render(buffer, camera, out, { maxSplats: 50 }).drawn).toBe(50);
    expect(out[(CENTRE * SIZE + CENTRE) * 4]).toBe(0);
  });
});

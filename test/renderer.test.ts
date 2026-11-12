import { describe, expect, it } from 'vitest';
import { orbitCamera } from '../src/camera';
import { quatBetween, quatFromAxisAngle, quatToMat3, rotate } from '../src/math';
import { buildSplatBuffer, Renderer } from '../src/renderer';
import type { Splat } from '../src/scene';

const SIZE = 101;
const CENTRE = 50;
// Camera at (0, 0, 5) looking down -Z at the origin.
const camera = orbitCamera({ target: [0, 0, 0], yaw: 0, pitch: 0, distance: 5, fovY: Math.PI / 2 });
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

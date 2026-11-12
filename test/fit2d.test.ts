import { describe, expect, it } from 'vitest';
import { DENSIFY_EVERY, Fit2D, PARAMS } from '../src/fit2d';

/** A small picture with some structure: a red disc on a blue-to-green gradient. */
function picture(width: number, height: number): Float32Array {
  const target = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 3;
      const inDisc = Math.hypot(x - width * 0.4, y - height * 0.5) < height * 0.25;
      target[p] = inDisc ? 0.9 : 0.1;
      target[p + 1] = inDisc ? 0.2 : x / width;
      target[p + 2] = inDisc ? 0.1 : 1 - x / width;
    }
  }
  return target;
}

describe('2D trainer', () => {
  it('computes the same gradients as nudging each parameter by hand', () => {
    const fit = new Fit2D(24, 20, picture(24, 20), { initialCount: 12, seed: 3 });
    fit.evaluate();
    const analytic = Float64Array.from(fit.grads.subarray(0, fit.count * PARAMS));
    const scale = Math.max(...analytic.map(Math.abs));

    const EPSILON = 1e-5;
    for (let k = 0; k < fit.count * PARAMS; k++) {
      const original = fit.params[k];
      fit.params[k] = original + EPSILON;
      const up = fit.evaluate();
      fit.params[k] = original - EPSILON;
      const down = fit.evaluate();
      fit.params[k] = original;
      const numeric = (up - down) / (2 * EPSILON);
      expect(Math.abs(numeric - analytic[k]) / scale, `parameter ${k}`).toBeLessThan(1e-3);
    }
  });

  it('gets closer to the picture as it trains', () => {
    const fit = new Fit2D(48, 36, picture(48, 36), { initialCount: 40, maxCount: 150 });
    const first = fit.step();
    for (let i = 0; i < 400; i++) fit.step();
    expect(fit.loss).toBeLessThan(first / 5);
    expect(fit.psnr).toBeGreaterThan(22);
  });

  it('adds gaussians when densifying, up to the limit', () => {
    const fit = new Fit2D(48, 36, picture(48, 36), { initialCount: 40, maxCount: 60 });
    for (let i = 0; i < DENSIFY_EVERY; i++) fit.step();
    expect(fit.count).toBeGreaterThan(40);
    for (let i = 0; i < DENSIFY_EVERY * 4; i++) fit.step();
    expect(fit.count).toBeLessThanOrEqual(60);
  });

  it('prunes gaussians that have become invisible', () => {
    const fit = new Fit2D(48, 36, picture(48, 36), { initialCount: 40, maxCount: 40 });
    for (let i = 0; i < 10; i++) fit.params[i * PARAMS + 8] = -10; // opacity ~ 0
    fit.densify();
    // 30 survive, and a quarter of those are then cloned or split.
    expect(fit.count).toBe(38);
    for (let i = 0; i < fit.count; i++) expect(fit.params[i * PARAMS + 8]).toBeGreaterThan(-5);
  });

  it('repeats exactly for the same seed', () => {
    const run = () => {
      const fit = new Fit2D(32, 24, picture(32, 24), { initialCount: 30, maxCount: 60, seed: 7 });
      for (let i = 0; i < 150; i++) fit.step();
      return fit.loss;
    };
    expect(run()).toBe(run());
  });
});

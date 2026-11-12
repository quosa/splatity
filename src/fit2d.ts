// Training, in miniature: fit a few hundred 2D gaussians to one picture.
//
// This is the loop that turns photographs into a splat scene, with the 3D and
// the cameras taken out so that only the idea is left:
//
//   1. RENDER    draw the current gaussians                       (forward)
//   2. COMPARE   measure how far the result is from the picture   (loss)
//   3. BLAME     work out, for every number describing every
//                gaussian, which way to nudge it to do better     (backward)
//   4. NUDGE     move every number a small step that way          (optimiser)
//   5. DENSIFY   now and then, add gaussians where the picture
//                needs more detail and delete invisible ones
//
// Real 3D training does exactly this, except that step 1 is the 3D renderer
// from renderer.ts and step 2 compares against a different photo each time.

/** Numbers stored per gaussian. All of them are learned. */
export const PARAMS = 9;
const X = 0; //            centre, in pixels
const Y = 1;
const LOG_SX = 2; //       log of the size along the two local axes, in pixels
const LOG_SY = 3;
const ANGLE = 4; //        rotation of the local axes, radians
const RED = 5; //          colour, 0..1
const GREEN = 6;
const BLUE = 7;
const OPACITY = 8; //      opacity before squashing: opacity = sigmoid(value)

// Two of these are stored in a roundabout way, the same as in real splat
// training. A size must stay positive and an opacity must stay within 0..1,
// but the optimiser just adds and subtracts. So it works on the logarithm of
// the size and on the "logit" of the opacity, which may be any number at all.

/** Gaussians contributing less than this to a pixel are skipped. */
const MIN_ALPHA = 1 / 255;
/** No single gaussian may fully cover a pixel (keeps 1 / (1 - alpha) finite). */
const MAX_ALPHA = 0.99;
/** A gaussian is drawn out to 3 standard deviations: q = 3². */
const MAX_Q = 9;
const MIN_LOG_SIZE = Math.log(0.3);

/**
 * How big a step each kind of number takes per iteration. They differ because
 * the numbers mean different things: a position moves in pixels, the others
 * are fractions.
 */
const LEARNING_RATE = [0.25, 0.25, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.05];

// Adam, the optimiser used by nearly all splat training (see `step`).
const ADAM_MOMENTUM = 0.9;
const ADAM_SCALE = 0.999;
const ADAM_EPSILON = 1e-15;

/** Iterations between densification rounds. */
export const DENSIFY_EVERY = 100;
/** Gaussians fainter than this are deleted when densifying. */
const PRUNE_OPACITY = 0.03;
/** Share of the gaussians that get cloned or split per round. */
const DENSIFY_SHARE = 0.25;

const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v));

/** Small deterministic PRNG, so a run can be repeated exactly. */
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

export interface FitOptions {
  /** Gaussians to start with. */
  initialCount?: number;
  /** Densification never grows the model beyond this. */
  maxCount?: number;
  seed?: number;
}

export class Fit2D {
  readonly width: number;
  readonly height: number;
  /** The picture to reproduce: rgb, 0..1, row by row. */
  readonly target: Float32Array;
  readonly maxCount: number;

  /** Number of gaussians in use. Gaussian 0 is drawn in front, the last at the back. */
  count = 0;
  /** `PARAMS` numbers per gaussian. This is the entire model. */
  readonly params: Float64Array;
  /** For each of those numbers: how the loss changes when it grows (step 3). */
  readonly grads: Float64Array;
  iteration = 0;
  /** Mean squared error of the last render against the target. */
  loss = 0;

  /** The rendered image: rgb per pixel. */
  readonly image: Float64Array;
  /** Per pixel, the fraction of light that passes through all gaussians. */
  private readonly transmittance: Float64Array;
  /** Per pixel and channel: how the loss changes when that value grows. */
  private readonly pixelGrads: Float64Array;
  /** Scratch for the backward pass: colour contributed by gaussians further back. */
  private readonly behind: Float64Array;

  // Adam's running averages, one per parameter.
  private readonly momentum: Float64Array;
  private readonly spread: Float64Array;
  /** Per gaussian: how hard the loss has been pulling on its position lately. */
  private readonly pull: Float64Array;
  private readonly random: () => number;

  constructor(width: number, height: number, target: Float32Array, options: FitOptions = {}) {
    this.width = width;
    this.height = height;
    this.target = target;
    this.maxCount = options.maxCount ?? 1500;
    this.random = mulberry32(options.seed ?? 1);
    const pixels = width * height;
    this.params = new Float64Array(this.maxCount * PARAMS);
    this.grads = new Float64Array(this.maxCount * PARAMS);
    this.momentum = new Float64Array(this.maxCount * PARAMS);
    this.spread = new Float64Array(this.maxCount * PARAMS);
    this.pull = new Float64Array(this.maxCount);
    this.image = new Float64Array(pixels * 3);
    this.transmittance = new Float64Array(pixels);
    this.pixelGrads = new Float64Array(pixels * 3);
    this.behind = new Float64Array(pixels * 3);
    this.scatter(Math.min(options.initialCount ?? 200, this.maxCount));
  }

  /**
   * The starting point: gaussians dropped at random, each taking the colour of
   * the picture underneath it. (3D training starts in a similar way, from the
   * coloured points that camera calibration produces.)
   */
  private scatter(count: number): void {
    const { width, height, params, target, random } = this;
    // Sized so that together they roughly cover the picture.
    const size = Math.sqrt((width * height) / count) * 0.6;
    for (let i = 0; i < count; i++) {
      const o = i * PARAMS;
      const x = random() * width;
      const y = random() * height;
      const pixel = (Math.min(height - 1, y | 0) * width + Math.min(width - 1, x | 0)) * 3;
      params[o + X] = x;
      params[o + Y] = y;
      params[o + LOG_SX] = Math.log(size * (0.6 + random() * 0.8));
      params[o + LOG_SY] = Math.log(size * (0.6 + random() * 0.8));
      params[o + ANGLE] = random() * Math.PI;
      params[o + RED] = target[pixel];
      params[o + GREEN] = target[pixel + 1];
      params[o + BLUE] = target[pixel + 2];
      params[o + OPACITY] = 0; // sigmoid(0) = 0.5
    }
    this.count = count;
  }

  /**
   * Step 1, RENDER. The same front-to-back blending as renderer.ts, on
   * gaussians that are already flat. `sizeScale` shrinks or grows every
   * gaussian, for display only.
   */
  render(sizeScale = 1): void {
    const { width, height, params, image, transmittance } = this;
    image.fill(0);
    transmittance.fill(1);

    for (let i = 0; i < this.count; i++) {
      const o = i * PARAMS;
      const x = params[o + X], y = params[o + Y];
      const sx = Math.exp(params[o + LOG_SX]) * sizeScale;
      const sy = Math.exp(params[o + LOG_SY]) * sizeScale;
      const cos = Math.cos(params[o + ANGLE]), sin = Math.sin(params[o + ANGLE]);
      const red = params[o + RED], green = params[o + GREEN], blue = params[o + BLUE];
      const opacity = sigmoid(params[o + OPACITY]);

      const reach = 3 * Math.max(sx, sy);
      const x0 = Math.max(0, Math.floor(x - reach)), x1 = Math.min(width - 1, Math.ceil(x + reach));
      const y0 = Math.max(0, Math.floor(y - reach)), y1 = Math.min(height - 1, Math.ceil(y + reach));

      for (let py = y0; py <= y1; py++) {
        const dy = py + 0.5 - y;
        for (let px = x0; px <= x1; px++) {
          const dx = px + 0.5 - x;
          // The pixel's offset, measured along the gaussian's own two axes
          // and in units of its size. q is then the squared distance from
          // the centre in standard deviations. (This is dᵀ·Σ⁻¹·d from the
          // 3D renderer, written without ever building Σ.)
          const u = (cos * dx + sin * dy) / sx;
          const v = (cos * dy - sin * dx) / sy;
          const q = u * u + v * v;
          if (q > MAX_Q) continue;
          const alpha = Math.min(MAX_ALPHA, opacity * Math.exp(-0.5 * q));
          if (alpha < MIN_ALPHA) continue;

          const p = py * width + px;
          const weight = alpha * transmittance[p];
          image[p * 3] += red * weight;
          image[p * 3 + 1] += green * weight;
          image[p * 3 + 2] += blue * weight;
          transmittance[p] *= 1 - alpha;
        }
      }
    }
  }

  /**
   * Step 2, COMPARE. The loss is the mean squared difference between the
   * render and the target: one number that says how wrong the picture is.
   * Also notes, for every pixel value, how the loss changes when it grows,
   * which is where step 3 starts from.
   */
  private compare(): void {
    const { image, target, pixelGrads } = this;
    const n = image.length;
    let sum = 0;
    for (let k = 0; k < n; k++) {
      const difference = image[k] - target[k];
      sum += difference * difference;
      pixelGrads[k] = (2 * difference) / n;
    }
    this.loss = sum / n;
  }

  /**
   * Step 3, BLAME ("backpropagation"). Fills `grads`: for every parameter,
   * the slope of the loss with respect to it.
   *
   * It is the render loop run backwards. Rendering went gaussian by gaussian
   * from front to back, each one dimming the light for those behind it. Here
   * we go from back to front, undoing that dimming as we go, and apply the
   * chain rule at every pixel a gaussian touched: loss <- pixel colour <-
   * alpha <- q <- the gaussian's own numbers.
   */
  private backward(): void {
    const { width, height, params, grads, transmittance, pixelGrads, behind } = this;
    grads.fill(0, 0, this.count * PARAMS);
    // `transmittance` still holds what the render left: the light getting
    // past every gaussian. Nothing is behind the last one (the background is
    // black), so `behind` starts at zero.
    behind.fill(0);

    for (let i = this.count - 1; i >= 0; i--) {
      const o = i * PARAMS;
      const x = params[o + X], y = params[o + Y];
      const sx = Math.exp(params[o + LOG_SX]), sy = Math.exp(params[o + LOG_SY]);
      const cos = Math.cos(params[o + ANGLE]), sin = Math.sin(params[o + ANGLE]);
      const red = params[o + RED], green = params[o + GREEN], blue = params[o + BLUE];
      const opacity = sigmoid(params[o + OPACITY]);

      const reach = 3 * Math.max(sx, sy);
      const x0 = Math.max(0, Math.floor(x - reach)), x1 = Math.min(width - 1, Math.ceil(x + reach));
      const y0 = Math.max(0, Math.floor(y - reach)), y1 = Math.min(height - 1, Math.ceil(y + reach));

      let gX = 0, gY = 0, gSx = 0, gSy = 0, gAngle = 0;
      let gRed = 0, gGreen = 0, gBlue = 0, gOpacity = 0;

      for (let py = y0; py <= y1; py++) {
        const dy = py + 0.5 - y;
        for (let px = x0; px <= x1; px++) {
          // Recompute exactly what the render computed for this pixel.
          const dx = px + 0.5 - x;
          const u = (cos * dx + sin * dy) / sx;
          const v = (cos * dy - sin * dx) / sy;
          const q = u * u + v * v;
          if (q > MAX_Q) continue;
          const unclamped = opacity * Math.exp(-0.5 * q);
          const alpha = Math.min(MAX_ALPHA, unclamped);
          if (alpha < MIN_ALPHA) continue;

          const p = py * width + px;
          const p3 = p * 3;
          // Undo this gaussian's dimming to get the light that reached it.
          const reaching = transmittance[p] / (1 - alpha);
          transmittance[p] = reaching;
          const weight = alpha * reaching;
          const lr = pixelGrads[p3], lg = pixelGrads[p3 + 1], lb = pixelGrads[p3 + 2];

          // Colour: the pixel received colour * weight, so the slope is the
          // pixel's slope times weight.
          gRed += lr * weight;
          gGreen += lg * weight;
          gBlue += lb * weight;

          // Alpha does two things to the pixel. More alpha adds more of this
          // gaussian's colour, and it hides more of everything behind it.
          const hide = 1 / (1 - alpha);
          const gAlpha =
            lr * (red * reaching - behind[p3] * hide) +
            lg * (green * reaching - behind[p3 + 1] * hide) +
            lb * (blue * reaching - behind[p3 + 2] * hide);

          // This gaussian is now "behind" for the ones still to come.
          behind[p3] += red * weight;
          behind[p3 + 1] += green * weight;
          behind[p3 + 2] += blue * weight;

          if (unclamped > MAX_ALPHA) continue; // clamped: alpha cannot change

          // alpha = sigmoid(stored) * exp(-q / 2)
          gOpacity += gAlpha * alpha * (1 - opacity);
          const gQ = gAlpha * -0.5 * alpha;
          // q = u² + v², where u and v depend on position, size and angle.
          gX += gQ * 2 * ((v * sin) / sy - (u * cos) / sx);
          gY += gQ * -2 * ((u * sin) / sx + (v * cos) / sy);
          gSx += gQ * -2 * u * u;
          gSy += gQ * -2 * v * v;
          gAngle += gQ * 2 * u * v * (sy / sx - sx / sy);
        }
      }

      grads[o + X] = gX;
      grads[o + Y] = gY;
      grads[o + LOG_SX] = gSx;
      grads[o + LOG_SY] = gSy;
      grads[o + ANGLE] = gAngle;
      grads[o + RED] = gRed;
      grads[o + GREEN] = gGreen;
      grads[o + BLUE] = gBlue;
      grads[o + OPACITY] = gOpacity;
    }
  }

  /** Steps 1 to 3 without changing anything: the loss and its gradients. */
  evaluate(): number {
    this.render();
    this.compare();
    this.backward();
    return this.loss;
  }

  /**
   * Step 4, NUDGE. Moves every parameter against its slope, using Adam.
   *
   * Plain gradient descent would be `value -= rate * slope`. The slopes here
   * differ by orders of magnitude between gaussians (a big one touches
   * thousands of pixels, a small one a handful), so no single rate suits
   * them all. Adam keeps a running average of each slope and of its typical
   * size, and divides one by the other: every parameter then moves by about
   * `rate` per step in whichever direction its slope has been pointing.
   */
  private nudge(): void {
    const { params, grads, momentum, spread } = this;
    const maxLogSize = Math.log(Math.max(this.width, this.height));
    for (let i = 0; i < this.count; i++) {
      const o = i * PARAMS;
      for (let k = 0; k < PARAMS; k++) {
        const g = grads[o + k];
        const m = (momentum[o + k] = ADAM_MOMENTUM * momentum[o + k] + (1 - ADAM_MOMENTUM) * g);
        const s = (spread[o + k] = ADAM_SCALE * spread[o + k] + (1 - ADAM_SCALE) * g * g);
        params[o + k] -= (LEARNING_RATE[k] * m) / (Math.sqrt(s) + ADAM_EPSILON);
      }
      // Keep the numbers where they make sense.
      params[o + LOG_SX] = Math.max(MIN_LOG_SIZE, Math.min(maxLogSize, params[o + LOG_SX]));
      params[o + LOG_SY] = Math.max(MIN_LOG_SIZE, Math.min(maxLogSize, params[o + LOG_SY]));
      for (let k = RED; k <= BLUE; k++) params[o + k] = Math.max(0, Math.min(1, params[o + k]));
      // Remember how hard the picture is pulling this gaussian around.
      this.pull[i] += Math.hypot(grads[o + X], grads[o + Y]);
    }
  }

  /** One full training iteration. Returns the loss before the nudge. */
  step(): number {
    this.evaluate();
    this.nudge();
    this.iteration++;
    if (this.iteration % DENSIFY_EVERY === 0) this.densify();
    return this.loss;
  }

  /**
   * Step 5, DENSIFY. Gradient descent can move and reshape gaussians, but it
   * cannot create them, so the model would be stuck with whatever it started
   * with. Every so often:
   *
   * - PRUNE gaussians that have faded to almost nothing.
   * - Find the gaussians whose position has been pulled hardest. A strong,
   *   persistent pull means the picture wants that gaussian in several
   *   places at once: the area needs more detail than one blob can give.
   *   - A small one is CLONED: a second copy to share the work.
   *   - A big one is SPLIT into two smaller ones placed inside it.
   *
   * (Real training uses a fixed threshold on the pull; taking the top share
   * is simpler and needs no tuning per picture.)
   */
  densify(): void {
    const { params, pull } = this;

    // Prune, closing the gaps while keeping the front-to-back order.
    let kept = 0;
    for (let i = 0; i < this.count; i++) {
      if (sigmoid(params[i * PARAMS + OPACITY]) < PRUNE_OPACITY) continue;
      if (kept !== i) this.move(i, kept);
      kept++;
    }
    this.count = kept;

    const budget = Math.min(this.maxCount - this.count, Math.ceil(this.count * DENSIFY_SHARE));
    if (budget > 0) {
      const byPull = Array.from({ length: this.count }, (_, i) => i).sort((a, b) => pull[b] - pull[a]);
      const sizes = byPull.map((i) => this.size(i)).sort((a, b) => a - b);
      const typicalSize = sizes[sizes.length >> 1];

      for (const i of byPull.slice(0, budget)) {
        const twin = this.count++;
        this.move(i, twin, true);
        if (this.size(i) > typicalSize) {
          // Split: shrink both and place them at random within the original.
          for (const j of [i, twin]) {
            const o = j * PARAMS;
            const sx = Math.exp(params[o + LOG_SX]), sy = Math.exp(params[o + LOG_SY]);
            const cos = Math.cos(params[o + ANGLE]), sin = Math.sin(params[o + ANGLE]);
            const a = this.gaussian() * sx, b = this.gaussian() * sy;
            params[o + X] += cos * a - sin * b;
            params[o + Y] += sin * a + cos * b;
            params[o + LOG_SX] -= Math.log(1.6);
            params[o + LOG_SY] -= Math.log(1.6);
          }
        }
      }
    }
    pull.fill(0);
  }

  /** The larger of a gaussian's two sizes, in pixels. */
  private size(i: number): number {
    return Math.exp(Math.max(this.params[i * PARAMS + LOG_SX], this.params[i * PARAMS + LOG_SY]));
  }

  /** Copies gaussian `from` to slot `to`; a `fresh` copy starts with no optimiser history. */
  private move(from: number, to: number, fresh = false): void {
    const a = from * PARAMS, b = to * PARAMS;
    this.params.copyWithin(b, a, a + PARAMS);
    if (fresh) {
      this.momentum.fill(0, b, b + PARAMS);
      this.spread.fill(0, b, b + PARAMS);
    } else {
      this.momentum.copyWithin(b, a, a + PARAMS);
      this.spread.copyWithin(b, a, a + PARAMS);
    }
    this.pull[to] = fresh ? 0 : this.pull[from];
  }

  /** A random number from the standard bell curve (Box-Muller). */
  private gaussian(): number {
    return Math.sqrt(-2 * Math.log(1 - this.random())) * Math.cos(2 * Math.PI * this.random());
  }

  /** Peak signal-to-noise ratio in decibels: the usual "how close" score. Higher is better. */
  get psnr(): number {
    return -10 * Math.log10(Math.max(this.loss, 1e-12));
  }

  /** Copies the last render into RGBA bytes for a canvas. */
  toRgba(out: Uint8ClampedArray): void {
    const { image } = this;
    for (let p = 0; p < this.width * this.height; p++) {
      out[p * 4] = image[p * 3] * 255;
      out[p * 4 + 1] = image[p * 3 + 1] * 255;
      out[p * 4 + 2] = image[p * 3 + 2] * 255;
      out[p * 4 + 3] = 255;
    }
  }
}

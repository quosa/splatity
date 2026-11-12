import { directionFromAngles, orbitCamera, roll, turn, type Orbit } from './camera';
import { frameScene, parseScene } from './loader';
import { buildSplatBuffer, Renderer, type SplatBuffer } from './renderer';
import { createScene } from './scene';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const stats = document.querySelector<HTMLElement>('#stats')!;
const hint = document.querySelector<HTMLElement>('#hint')!;
const sizeSlider = document.querySelector<HTMLInputElement>('#size')!;
const fileInput = document.querySelector<HTMLInputElement>('#file')!;
const ctx = canvas.getContext('2d')!;

const BACKGROUND: [number, number, number] = [0.05, 0.06, 0.09];
/** Longest canvas side in pixels. Every pixel is computed in JavaScript. */
const MAX_SIDE = 720;
/** While the camera moves, aim for frames about this quick (milliseconds). */
const PREVIEW_MS = 40;
/** After this long without input, draw the scene in full. */
const SETTLE_MS = 200;

let splats: SplatBuffer = buildSplatBuffer(createScene());
/**
 * Loaded files are far too big to draw in full at interactive speed, but they
 * are stored most-important-first, so while the camera moves we draw only the
 * start of the file. This is how many splats that is; it adapts to the machine.
 */
let previewBudget = Infinity;

const orbit: Orbit = {
  target: [0, 0.9, 0],
  up: [0, 1, 0],
  direction: directionFromAngles(0.6, 0.32),
  distance: 6.5,
  fovY: (45 * Math.PI) / 180,
};

/** A render target: a renderer plus the pixels it draws into. */
interface Target {
  renderer: Renderer;
  image: ImageData;
}
const makeTarget = (width: number, height: number): Target => ({
  renderer: new Renderer(width, height),
  image: new ImageData(width, height),
});
let full: Target;
/** Half the resolution (a quarter of the pixels), used while the camera moves. */
let half: Target;
const halfCanvas = document.createElement('canvas');
let autoRotate = true;
/** The view changed and needs drawing. */
let dirty = true;
/** The last frame drawn was only a preview. */
let showingPreview = false;
let lastInput = 0;

function touch(): void {
  dirty = true;
  lastInput = performance.now();
}

function resize(): void {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const fit = Math.min(1, MAX_SIDE / Math.max(w, h));
  const width = Math.max(1, Math.round(w * fit));
  const height = Math.max(1, Math.round(h * fit));
  if (full && full.renderer.width === width && full.renderer.height === height) return;
  canvas.width = width;
  canvas.height = height;
  full = makeTarget(width, height);
  half = makeTarget(Math.ceil(width / 2), Math.ceil(height / 2));
  halfCanvas.width = half.image.width;
  halfCanvas.height = half.image.height;
  touch();
}

let lastTime = performance.now();

function frame(now: number): void {
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  if (autoRotate) {
    turn(orbit, dt * 0.25, 0);
    touch();
  }
  const settled = now - lastInput > SETTLE_MS;
  if (dirty || (showingPreview && settled)) {
    const preview = dirty && !settled && previewBudget < splats.count;
    dirty = false;
    showingPreview = preview;

    const start = performance.now();
    const { renderer, image } = preview ? half : full;
    const { drawn } = renderer.render(splats, orbitCamera(orbit), image.data, {
      splatScale: Number(sizeSlider.value),
      background: BACKGROUND,
      maxSplats: preview ? previewBudget : splats.count,
    });
    if (preview) {
      halfCanvas.getContext('2d')!.putImageData(image, 0, 0);
      ctx.drawImage(halfCanvas, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.putImageData(image, 0, 0);
    }
    const ms = performance.now() - start;

    if (preview) {
      // Steer the budget towards the target frame time, a little at a time.
      const correction = Math.max(0.7, Math.min(1.3, PREVIEW_MS / ms));
      previewBudget = Math.max(2000, Math.round(previewBudget * correction));
    }
    stats.textContent =
      `${drawn.toLocaleString()} of ${splats.count.toLocaleString()} gaussians` +
      `${preview ? ' (preview)' : ''} · ${renderer.width}×${renderer.height} · ` +
      `${ms.toFixed(0)} ms on the CPU`;
  }
  requestAnimationFrame(frame);
}

// ---- Loading .splat and .ply files ---------------------------------------------------

async function load(file: File): Promise<void> {
  stats.textContent = `Loading ${file.name}…`;
  try {
    splats = parseScene(await file.arrayBuffer());
  } catch (error) {
    stats.textContent = `Could not load ${file.name}: ${(error as Error).message}`;
    return;
  }
  const { centre, radius, surrounded } = frameScene(splats);
  orbit.target = centre;
  orbit.up = [0, 1, 0];
  orbit.direction = directionFromAngles(0.6, 0.2);
  // Stand back from a lone object; stay close in a scene with surroundings.
  orbit.distance = Math.max(radius, 0.01) * (surrounded ? 1.5 : 3);
  autoRotate = false;
  previewBudget = Math.min(previewBudget, 20000);
  hint.textContent =
    'Drag to orbit · scroll or pinch to zoom · scene tilted? shift-drag or twist two fingers to level it';
  touch();
}

fileInput.addEventListener('change', () => {
  if (fileInput.files?.[0]) void load(fileInput.files[0]);
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files[0];
  if (file) void load(file);
});

// ---- Input: drag to orbit, shift-drag or twist to level, wheel or pinch to zoom

const pointers = new Map<number, { x: number; y: number }>();
/** Distance and angle between the two fingers of a pinch. */
const pinch = (): { spread: number; angle: number } => {
  const [a, b] = [...pointers.values()];
  return { spread: Math.hypot(a.x - b.x, a.y - b.y), angle: Math.atan2(b.y - a.y, b.x - a.x) };
};
const zoom = (factor: number): void => {
  orbit.distance *= factor;
  touch();
};

canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  autoRotate = false;
});
canvas.addEventListener('pointermove', (e) => {
  const prev = pointers.get(e.pointerId);
  if (!prev) return;
  if (pointers.size === 1) {
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    if (e.shiftKey) roll(orbit, dx * 0.005);
    else turn(orbit, -dx * 0.008, dy * 0.008);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    touch();
  } else if (pointers.size === 2) {
    const before = pinch();
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const after = pinch();
    // The scene follows the fingers, so the turntable axis turns with them.
    roll(orbit, after.angle - before.angle);
    zoom(before.spread / Math.max(1, after.spread));
  }
});
const release = (e: PointerEvent): void => void pointers.delete(e.pointerId);
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);
canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    autoRotate = false;
    zoom(Math.exp(e.deltaY * 0.001));
  },
  { passive: false },
);
sizeSlider.addEventListener('input', touch);
window.addEventListener('resize', resize);

resize();
requestAnimationFrame(frame);

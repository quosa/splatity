import { orbitCamera, type Orbit } from './camera';
import { buildSplatBuffer, Renderer } from './renderer';
import { createScene } from './scene';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const stats = document.querySelector<HTMLElement>('#stats')!;
const sizeSlider = document.querySelector<HTMLInputElement>('#size')!;
const ctx = canvas.getContext('2d')!;

const splats = buildSplatBuffer(createScene());
const BACKGROUND: [number, number, number] = [0.05, 0.06, 0.09];
/** Longest canvas side in pixels. Every pixel is computed in JavaScript. */
const MAX_SIDE = 720;

const orbit: Orbit = {
  target: [0, 0.9, 0],
  yaw: 0.6,
  pitch: 0.32,
  distance: 6.5,
  fovY: (45 * Math.PI) / 180,
};

let renderer: Renderer;
let image: ImageData;
let autoRotate = true;
let dirty = true;

function resize(): void {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const fit = Math.min(1, MAX_SIDE / Math.max(w, h));
  const width = Math.max(1, Math.round(w * fit));
  const height = Math.max(1, Math.round(h * fit));
  if (renderer && renderer.width === width && renderer.height === height) return;
  canvas.width = width;
  canvas.height = height;
  renderer = new Renderer(width, height);
  image = ctx.createImageData(width, height);
  dirty = true;
}

let smoothedMs = 0;
let lastTime = performance.now();

function frame(now: number): void {
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  if (autoRotate) {
    orbit.yaw += dt * 0.25;
    dirty = true;
  }
  if (dirty) {
    dirty = false;
    const start = performance.now();
    const { drawn } = renderer.render(splats, orbitCamera(orbit), image.data, {
      splatScale: Number(sizeSlider.value),
      background: BACKGROUND,
    });
    ctx.putImageData(image, 0, 0);
    const ms = performance.now() - start;
    smoothedMs = smoothedMs ? smoothedMs * 0.8 + ms * 0.2 : ms;
    stats.textContent =
      `${drawn} of ${splats.count} gaussians · ${renderer.width}×${renderer.height} · ` +
      `${smoothedMs.toFixed(0)} ms/frame on the CPU`;
  }
  requestAnimationFrame(frame);
}

// ---- Input: drag to orbit, wheel or pinch to zoom --------------------------

const pointers = new Map<number, { x: number; y: number }>();
const pinchSpread = (): number => {
  const [a, b] = [...pointers.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
};
const zoom = (factor: number): void => {
  orbit.distance = Math.min(20, Math.max(1.5, orbit.distance * factor));
  dirty = true;
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
    orbit.yaw -= (e.clientX - prev.x) * 0.008;
    orbit.pitch = Math.min(1.5, Math.max(-0.2, orbit.pitch + (e.clientY - prev.y) * 0.008));
    dirty = true;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  } else if (pointers.size === 2) {
    const before = pinchSpread();
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    zoom(before / Math.max(1, pinchSpread()));
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
sizeSlider.addEventListener('input', () => (dirty = true));
window.addEventListener('resize', resize);

resize();
requestAnimationFrame(frame);

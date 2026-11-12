import { directionFromAngles, orbitCamera } from './camera';
import { DENSIFY_EVERY, Fit2D } from './fit2d';
import { buildSplatBuffer, Renderer } from './renderer';
import { createScene } from './scene';

const targetCanvas = document.querySelector<HTMLCanvasElement>('#target')!;
const fitCanvas = document.querySelector<HTMLCanvasElement>('#fit')!;
const curveCanvas = document.querySelector<HTMLCanvasElement>('#curve')!;
const stats = document.querySelector<HTMLElement>('#stats')!;
const pauseButton = document.querySelector<HTMLButtonElement>('#pause')!;
const restartButton = document.querySelector<HTMLButtonElement>('#restart')!;
const maxSelect = document.querySelector<HTMLSelectElement>('#max')!;
const sizeSlider = document.querySelector<HTMLInputElement>('#size')!;
const fileInput = document.querySelector<HTMLInputElement>('#file')!;

/** Longest side of the training picture. Small, because this runs on the CPU. */
const MAX_SIDE = 160;
/** Time spent training per animation frame, in milliseconds. */
const FRAME_BUDGET = 30;

/** The picture being fitted, as RGBA bytes at training resolution. */
let picture: ImageData;
let fit: Fit2D;
let fitImage: ImageData;
let paused = false;
/** PSNR after each iteration, for the curve. */
let history: number[] = [];
let speed = 0;

/** The default target: our own 3D scene, rendered by the splat renderer. */
function renderScene(): ImageData {
  const width = MAX_SIDE;
  const height = Math.round(MAX_SIDE * 0.625);
  const image = new ImageData(width, height);
  const camera = orbitCamera({
    target: [0, 0.9, 0],
    up: [0, 1, 0],
    direction: directionFromAngles(0.6, 0.32),
    distance: 6.5,
    fovY: (45 * Math.PI) / 180,
  });
  new Renderer(width, height).render(buildSplatBuffer(createScene()), camera, image.data, {
    background: [0.05, 0.06, 0.09],
  });
  return image;
}

/** Scales a picture file down to training resolution. */
async function readPicture(file: File): Promise<ImageData> {
  const bitmap = await createImageBitmap(file);
  const fitSide = MAX_SIDE / Math.max(bitmap.width, bitmap.height);
  const width = Math.max(1, Math.round(bitmap.width * fitSide));
  const height = Math.max(1, Math.round(bitmap.height * fitSide));
  const scratch = document.createElement('canvas');
  scratch.width = width;
  scratch.height = height;
  const ctx = scratch.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

/** Starts training from scratch on the current picture. */
function restart(): void {
  const { width, height, data } = picture;
  const target = new Float32Array(width * height * 3);
  for (let p = 0; p < width * height; p++) {
    target[p * 3] = data[p * 4] / 255;
    target[p * 3 + 1] = data[p * 4 + 1] / 255;
    target[p * 3 + 2] = data[p * 4 + 2] / 255;
  }
  fit = new Fit2D(width, height, target, { maxCount: Number(maxSelect.value) });
  fitImage = new ImageData(width, height);
  history = [];
  for (const canvas of [targetCanvas, fitCanvas]) {
    canvas.width = width;
    canvas.height = height;
  }
  targetCanvas.getContext('2d')!.putImageData(picture, 0, 0);
  show();
}

/** Draws the current gaussians, the numbers and the curve. */
function show(): void {
  fit.render(Number(sizeSlider.value));
  fit.toRgba(fitImage.data);
  fitCanvas.getContext('2d')!.putImageData(fitImage, 0, 0);

  const toNextDensify = DENSIFY_EVERY - (fit.iteration % DENSIFY_EVERY);
  stats.textContent =
    `Iteration ${fit.iteration.toLocaleString()} · ${fit.count.toLocaleString()} gaussians · ` +
    (history.length ? `closeness ${history[history.length - 1].toFixed(1)} dB · ` : '') +
    (fit.count < fit.maxCount ? `next densify in ${toNextDensify} · ` : '') +
    (paused ? 'paused' : `${speed.toFixed(0)} iterations/s on the CPU`);
  drawCurve();
}

/** PSNR over time: a line that climbs as the fit improves. */
function drawCurve(): void {
  const width = (curveCanvas.width = curveCanvas.clientWidth);
  const height = (curveCanvas.height = curveCanvas.clientHeight);
  const ctx = curveCanvas.getContext('2d')!;
  ctx.clearRect(0, 0, width, height);
  if (history.length < 2) return;
  const low = Math.min(...history);
  const high = Math.max(...history, low + 1);
  ctx.strokeStyle = '#8ab4ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  // At most one point per horizontal pixel.
  const stride = Math.max(1, Math.floor(history.length / width));
  for (let i = 0; i < history.length; i += stride) {
    const x = (i / (history.length - 1)) * (width - 2) + 1;
    const y = height - 4 - ((history[i] - low) / (high - low)) * (height - 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#9aa3b5';
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(`${high.toFixed(1)} dB`, 4, 12);
  ctx.fillText(`${low.toFixed(1)} dB`, 4, height - 4);
}

function frame(): void {
  if (!paused) {
    const start = performance.now();
    let steps = 0;
    do {
      fit.step();
      history.push(fit.psnr);
      steps++;
    } while (performance.now() - start < FRAME_BUDGET);
    const perSecond = (steps * 1000) / (performance.now() - start);
    speed = speed ? speed * 0.9 + perSecond * 0.1 : perSecond;
    show();
  }
  requestAnimationFrame(frame);
}

async function load(file: File): Promise<void> {
  try {
    picture = await readPicture(file);
  } catch {
    stats.textContent = `Could not read ${file.name} as a picture.`;
    return;
  }
  restart();
}

pauseButton.addEventListener('click', () => {
  paused = !paused;
  pauseButton.textContent = paused ? 'Resume' : 'Pause';
  show();
});
restartButton.addEventListener('click', restart);
maxSelect.addEventListener('change', restart);
sizeSlider.addEventListener('input', show);
fileInput.addEventListener('change', () => {
  if (fileInput.files?.[0]) void load(fileInput.files[0]);
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files[0];
  if (file) void load(file);
});

picture = renderScene();
restart();
requestAnimationFrame(frame);

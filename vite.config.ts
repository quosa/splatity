import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the built site works from any sub-path (e.g. GitHub Pages).
  base: './',
  build: {
    rollupOptions: {
      // Two pages: the 3D viewer and the 2D trainer.
      input: ['index.html', 'trainer.html'],
    },
  },
});

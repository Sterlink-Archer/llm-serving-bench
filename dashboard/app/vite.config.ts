import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Chrome refuses ES-module scripts from file://, and the site must open by double-click with no server.
// So the bundle is a single classic IIFE script with stable names; data.js (window.BENCH) loads before it.
function classicScript(): Plugin {
  return {
    name: 'classic-script',
    enforce: 'post',
    transformIndexHtml(html) {
      return html.replace(/<script type="module" crossorigin src="([^"]+)"><\/script>/, '<script defer src="$1"></script>')
        .replace(/ crossorigin/g, '');
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), classicScript()],
  build: {
    outDir: '../ui',
    emptyOutDir: true,
    target: 'es2020',
    modulePreload: false,
    cssCodeSplit: false,
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        format: 'iife',
        inlineDynamicImports: true,
        entryFileNames: 'assets/app.js',
        assetFileNames: 'assets/[name][extname]',
      },
    },
  },
});

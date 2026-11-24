import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root = dirname(fileURLToPath(import.meta.url));
let torch = process.env.TORCH_PSF_WEB && resolve(process.env.TORCH_PSF_WEB);
for (let p = root; !torch && dirname(p) !== p; p = dirname(p)) {
  const candidate = join(p, 'torch-psf', 'web');
  if (existsSync(join(candidate, 'src', 'dense', 'movie.ts'))) torch = candidate;
}
if (!torch) throw Error('Set TORCH_PSF_WEB to the TorchPSF web directory.');
const { createServer, build } = await import(
  pathToFileURL(join(torch, 'node_modules/vite/dist/node/index.js')).href
);
const config = {
  root,
  configFile: false,
  base: './',
  resolve: { alias: { 'torch-dense': join(torch, 'src', 'dense'), 'torch-web': join(torch, 'src') } },
  server: { host: '127.0.0.1', port: 5177, strictPort: true, fs: { allow: [root, torch] } },
  build: { target: 'es2022' },
  worker: { format: 'es' },
};
if (process.argv.includes('--build')) await build(config);
else {
  const s = await createServer(config);
  await s.listen();
  s.printUrls();
}

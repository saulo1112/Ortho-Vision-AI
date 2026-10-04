// The web demo runs the model in the browser, so best.onnx must be served as a
// static file. public/ is copied verbatim into the web export; the file stays
// gitignored (the source of truth is model/weights, tracked with Git LFS).
const fs = require('node:fs');
const path = require('node:path');

const src = path.join(__dirname, '..', '..', 'model', 'weights', 'best.onnx');
const dest = path.join(__dirname, '..', 'public', 'best.onnx');

if (!fs.existsSync(src) || fs.statSync(src).size < 1_000_000) {
  console.error(`Missing ${src} (or it is a Git LFS pointer — run "git lfs pull").`);
  process.exit(1);
}
fs.mkdirSync(path.dirname(dest), { recursive: true });
fs.copyFileSync(src, dest);
console.log('Copied best.onnx -> mobile/public/');

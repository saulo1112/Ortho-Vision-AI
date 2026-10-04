/**
 * Browser-side port of backend/app/inference (preprocess + postprocess).
 *
 * Same YOLOv8-seg pipeline as the API — letterbox, NMS, prototype masks,
 * polygonization — in dependency-free TypeScript so the demo runs without a
 * server. Deliberately free of DOM / onnxruntime imports: the caller provides
 * the pixels and a `run` function, which keeps this unit testable in Node.
 */

import type { Detection } from '../api/types';

export const IMG_SIZE = 640;
const PAD_VALUE = 114 / 255; // Ultralytics letterbox padding
const NUM_MASK_COEFFS = 32;
const PROTO_SIZE = 160;
const MAX_DETECTIONS = 300;
const MIN_BLOB_AREA_FRAC = 0.0005;
const POLY_EPSILON_FRAC = 0.002;
/** Longest side of the working mask used for polygonization. */
const MASK_WORK_SIDE = 640;

export const DEFAULT_CONF = 0.5;
export const DEFAULT_IOU = 0.7;

export const CLASS_NAMES: Record<number, string> = {
  0: 'clavo_intramedular',
  1: 'placa_atornillada',
  2: 'protesis_articular',
};

export interface RgbaImage {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

export interface RawOutputs {
  output0: Float32Array; // (1, 4 + nc + 32, 8400)
  output0Dims: readonly number[];
  output1: Float32Array; // (1, 32, 160, 160)
}

export type RunModel = (tensor: Float32Array) => Promise<RawOutputs>;

export interface SegmentationResult {
  detections: Detection[];
  timing_ms: { preprocess: number; inference: number; postprocess: number; total: number };
}

// ---------------------------------------------------------------- preprocess

interface Letterbox {
  tensor: Float32Array; // RGB float32 CHW in [0, 1], 3 x 640 x 640
  ratio: number;
  left: number;
  top: number;
}

/** Bilinear resize + pad to a square tensor (cv2.INTER_LINEAR sampling). */
export function letterbox(img: RgbaImage): Letterbox {
  const { width: w, height: h, data } = img;
  const ratio = Math.min(IMG_SIZE / h, IMG_SIZE / w);
  const newW = Math.round(w * ratio);
  const newH = Math.round(h * ratio);
  const dw = (IMG_SIZE - newW) / 2;
  const dh = (IMG_SIZE - newH) / 2;
  const left = Math.round(dw - 0.1);
  const top = Math.round(dh - 0.1);

  const plane = IMG_SIZE * IMG_SIZE;
  const tensor = new Float32Array(3 * plane).fill(PAD_VALUE);

  const scaleX = w / newW;
  const scaleY = h / newH;
  for (let y = 0; y < newH; y++) {
    const fy = Math.max((y + 0.5) * scaleY - 0.5, 0);
    const y0 = Math.min(Math.floor(fy), h - 1);
    const y1 = Math.min(y0 + 1, h - 1);
    const wy = fy - y0;
    for (let x = 0; x < newW; x++) {
      const fx = Math.max((x + 0.5) * scaleX - 0.5, 0);
      const x0 = Math.min(Math.floor(fx), w - 1);
      const x1 = Math.min(x0 + 1, w - 1);
      const wx = fx - x0;
      const i00 = (y0 * w + x0) * 4;
      const i01 = (y0 * w + x1) * 4;
      const i10 = (y1 * w + x0) * 4;
      const i11 = (y1 * w + x1) * 4;
      const dst = (y + top) * IMG_SIZE + (x + left);
      for (let c = 0; c < 3; c++) {
        const top_ = data[i00 + c] * (1 - wx) + data[i01 + c] * wx;
        const bot_ = data[i10 + c] * (1 - wx) + data[i11 + c] * wx;
        tensor[c * plane + dst] = (top_ * (1 - wy) + bot_ * wy) / 255;
      }
    }
  }
  return { tensor, ratio, left, top };
}

// ------------------------------------------------------------------- NMS

interface Candidate {
  box: [number, number, number, number]; // x1, y1, x2, y2 in 640-space
  conf: number;
  classId: number;
  coeffs: Float32Array;
}

function nonMaxSuppression(
  output0: Float32Array,
  dims: readonly number[],
  confThres: number,
  iouThres: number,
): Candidate[] {
  const channels = dims[1];
  const n = dims[2];
  const numClasses = channels - 4 - NUM_MASK_COEFFS;

  const cands: Candidate[] = [];
  for (let i = 0; i < n; i++) {
    let best = -1;
    let bestClass = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = output0[(4 + c) * n + i];
      if (s > best) {
        best = s;
        bestClass = c;
      }
    }
    if (best < confThres) continue;
    const cx = output0[i];
    const cy = output0[n + i];
    const bw = output0[2 * n + i];
    const bh = output0[3 * n + i];
    const coeffs = new Float32Array(NUM_MASK_COEFFS);
    for (let k = 0; k < NUM_MASK_COEFFS; k++) coeffs[k] = output0[(4 + numClasses + k) * n + i];
    cands.push({
      box: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
      conf: best,
      classId: bestClass,
      coeffs,
    });
  }

  // Class-aware greedy NMS (equivalent to the backend's per-class box offset).
  cands.sort((a, b) => b.conf - a.conf);
  const keep: Candidate[] = [];
  for (const cand of cands) {
    let suppressed = false;
    for (const k of keep) {
      if (k.classId !== cand.classId) continue;
      if (iou(k.box, cand.box) > iouThres) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) keep.push(cand);
    if (keep.length >= MAX_DETECTIONS) break;
  }
  return keep;
}

function iou(a: Candidate['box'], b: Candidate['box']): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-9);
}

// ------------------------------------------------------------------- masks

/**
 * Instance mask at the working resolution (<= MASK_WORK_SIDE on the longest
 * side, original aspect ratio): prototype blend, box crop, letterbox padding
 * removed *before* resizing, threshold at 0.5.
 */
function buildMask(
  protos: Float32Array,
  cand: Candidate,
  ratio: number,
  left: number,
  top: number,
  w0: number,
  h0: number,
  outW: number,
  outH: number,
): Uint8Array {
  const P = PROTO_SIZE;
  const prob = new Float32Array(P * P);
  for (let p = 0; p < P * P; p++) {
    let acc = 0;
    for (let k = 0; k < NUM_MASK_COEFFS; k++) acc += cand.coeffs[k] * protos[k * P * P + p];
    prob[p] = 1 / (1 + Math.exp(-acc));
  }

  const scale = P / IMG_SIZE;
  const px1 = Math.round(left * scale);
  const py1 = Math.round(top * scale);
  const px2 = Math.min(px1 + Math.max(1, Math.round(w0 * ratio * scale)), P);
  const py2 = Math.min(py1 + Math.max(1, Math.round(h0 * ratio * scale)), P);

  // Zero everything outside the detection box (in proto space).
  const bx1 = Math.max(Math.trunc(cand.box[0] * scale), 0);
  const by1 = Math.max(Math.trunc(cand.box[1] * scale), 0);
  const bx2 = Math.min(Math.ceil(cand.box[2] * scale), P);
  const by2 = Math.min(Math.ceil(cand.box[3] * scale), P);

  const cw = px2 - px1;
  const ch = py2 - py1;
  const content = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) {
    const sy = py1 + y;
    if (sy < by1 || sy >= by2) continue;
    for (let x = 0; x < cw; x++) {
      const sx = px1 + x;
      if (sx >= bx1 && sx < bx2) content[y * cw + x] = prob[sy * P + sx];
    }
  }

  // Bilinear resize of the content crop to the working size, then threshold.
  const out = new Uint8Array(outW * outH);
  const sx = cw / outW;
  const sy = ch / outH;
  for (let y = 0; y < outH; y++) {
    const fy = Math.max((y + 0.5) * sy - 0.5, 0);
    const y0 = Math.min(Math.floor(fy), ch - 1);
    const y1 = Math.min(y0 + 1, ch - 1);
    const wy = fy - y0;
    for (let x = 0; x < outW; x++) {
      const fx = Math.max((x + 0.5) * sx - 0.5, 0);
      const x0 = Math.min(Math.floor(fx), cw - 1);
      const x1 = Math.min(x0 + 1, cw - 1);
      const wx = fx - x0;
      const v =
        (content[y0 * cw + x0] * (1 - wx) + content[y0 * cw + x1] * wx) * (1 - wy) +
        (content[y1 * cw + x0] * (1 - wx) + content[y1 * cw + x1] * wx) * wy;
      out[y * outW + x] = v >= 0.5 ? 1 : 0;
    }
  }
  return out;
}

// ---------------------------------------------------------------- polygons

type Pt = [number, number];

/**
 * Outer boundaries of a binary mask as pixel-corner loops. Each boundary edge
 * is directed so the foreground lies on its right, which makes outer loops
 * positive-area and holes negative-area (y-down shoelace); holes are dropped,
 * matching OpenCV's RETR_EXTERNAL used by the backend.
 */
function traceOuterLoops(mask: Uint8Array, w: number, h: number): Pt[][] {
  const stride = w + 1;
  const next = new Map<number, number[]>();
  const add = (x0: number, y0: number, x1: number, y1: number) => {
    const from = y0 * stride + x0;
    const list = next.get(from);
    const to = y1 * stride + x1;
    if (list) list.push(to);
    else next.set(from, [to]);
  };
  const on = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] === 1;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      if (!on(x, y - 1)) add(x, y, x + 1, y);
      if (!on(x + 1, y)) add(x + 1, y, x + 1, y + 1);
      if (!on(x, y + 1)) add(x + 1, y + 1, x, y + 1);
      if (!on(x - 1, y)) add(x, y + 1, x, y);
    }
  }

  const loops: Pt[][] = [];
  for (const [start] of next) {
    while (next.get(start)?.length) {
      const loop: Pt[] = [];
      let cur = start;
      do {
        loop.push([cur % stride, Math.floor(cur / stride)]);
        const outs = next.get(cur);
        if (!outs || !outs.length) break;
        cur = outs.pop()!;
      } while (cur !== start);
      if (area(loop) > 0) loops.push(loop);
    }
  }
  return loops;
}

function area(loop: Pt[]): number {
  let a = 0;
  for (let i = 0; i < loop.length; i++) {
    const [x1, y1] = loop[i];
    const [x2, y2] = loop[(i + 1) % loop.length];
    a += x1 * y2 - x2 * y1;
  }
  return a / 2;
}

function perimeter(loop: Pt[]): number {
  let p = 0;
  for (let i = 0; i < loop.length; i++) {
    const [x1, y1] = loop[i];
    const [x2, y2] = loop[(i + 1) % loop.length];
    p += Math.hypot(x2 - x1, y2 - y1);
  }
  return p;
}

function distToSegment(p: Pt, a: Pt, b: Pt): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/** Douglas-Peucker on an open polyline (iterative). Keeps both endpoints. */
function simplifyOpen(pts: Pt[], eps: number): Pt[] {
  if (pts.length < 3) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack: [number, number][] = [[0, pts.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop()!;
    let maxD = 0;
    let idx = -1;
    for (let i = s + 1; i < e; i++) {
      const d = distToSegment(pts[i], pts[s], pts[e]);
      if (d > maxD) {
        maxD = d;
        idx = i;
      }
    }
    if (idx !== -1 && maxD > eps) {
      keep[idx] = 1;
      stack.push([s, idx], [idx, e]);
    }
  }
  return pts.filter((_, i) => keep[i]);
}

/** Closed-ring Douglas-Peucker: split at the point farthest from the first. */
function simplifyClosed(ring: Pt[], eps: number): Pt[] {
  if (ring.length < 4) return ring;
  let far = 0;
  let maxD = -1;
  for (let i = 1; i < ring.length; i++) {
    const d = Math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]);
    if (d > maxD) {
      maxD = d;
      far = i;
    }
  }
  const a = simplifyOpen(ring.slice(0, far + 1), eps);
  const b = simplifyOpen([...ring.slice(far), ring[0]], eps);
  return [...a.slice(0, -1), ...b.slice(0, -1)];
}

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

function maskToPolygons(mask: Uint8Array, w: number, h: number): number[][][] {
  const minArea = MIN_BLOB_AREA_FRAC * w * h;
  const rings: number[][][] = [];
  for (const loop of traceOuterLoops(mask, w, h)) {
    if (area(loop) < minArea || loop.length < 3) continue;
    const approx = simplifyClosed(loop, POLY_EPSILON_FRAC * perimeter(loop));
    if (approx.length < 3) continue;
    rings.push(approx.map(([x, y]) => [round4(x / w), round4(y / h)]));
  }
  return rings;
}

// ------------------------------------------------------------------ entry

export async function segment(
  img: RgbaImage,
  run: RunModel,
  conf: number = DEFAULT_CONF,
  iouThres: number = DEFAULT_IOU,
): Promise<SegmentationResult> {
  const now = () => performance.now();
  const t0 = now();
  const { width: w0, height: h0 } = img;
  const { tensor, ratio, left, top } = letterbox(img);

  const t1 = now();
  const { output0, output0Dims, output1 } = await run(tensor);

  const t2 = now();
  const cands = nonMaxSuppression(output0, output0Dims, conf, iouThres);

  const k = Math.min(1, MASK_WORK_SIDE / Math.max(w0, h0));
  const outW = Math.max(1, Math.round(w0 * k));
  const outH = Math.max(1, Math.round(h0 * k));

  const detections: Detection[] = cands.map((cand, id) => {
    const mask = buildMask(output1, cand, ratio, left, top, w0, h0, outW, outH);
    const clampX = (v: number) => Math.min(Math.max((v - left) / ratio, 0), w0) / w0;
    const clampY = (v: number) => Math.min(Math.max((v - top) / ratio, 0), h0) / h0;
    return {
      id,
      class_id: cand.classId,
      class_name: CLASS_NAMES[cand.classId] ?? String(cand.classId),
      confidence: round4(cand.conf),
      bbox: {
        x1: round4(clampX(cand.box[0])),
        y1: round4(clampY(cand.box[1])),
        x2: round4(clampX(cand.box[2])),
        y2: round4(clampY(cand.box[3])),
      },
      polygons: maskToPolygons(mask, outW, outH),
    };
  });

  const t3 = now();
  return {
    detections,
    timing_ms: {
      preprocess: Math.round(t1 - t0),
      inference: Math.round(t2 - t1),
      postprocess: Math.round(t3 - t2),
      total: Math.round(t3 - t0),
    },
  };
}

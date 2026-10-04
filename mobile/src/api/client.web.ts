/**
 * Web build of the API client — no server involved.
 *
 * Metro picks this file over client.ts on web. The model runs in the browser
 * (onnxruntime-web, loaded lazily from a CDN) and the history lives in
 * localStorage, so the demo can be hosted as plain static files.
 * Exposes the same functions as client.ts so the screens are untouched.
 */

import { segment, type RawOutputs } from '../inference/segmentation';
import type {
  HealthResponse,
  InferenceDetail,
  InferenceListResponse,
  InferenceSummary,
  PredictResponse,
} from './types';

const MODEL_VERSION = 'orthovision-yolov8s-seg-1.0-onnx (in-browser)';
const ORT_VERSION = '1.19.2';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_URL = `${process.env.EXPO_BASE_URL ?? ''}/best.onnx`;
const HISTORY_KEY = 'orthovision.history.v1';
const HISTORY_MAX = 30;
const THUMB_SIDE = 512;
/** Same cap as the backend, keeps canvas memory bounded for phone photos. */
const MAX_SIDE = 4096;

export class ApiError extends Error {
  constructor(
    public status: number,
    detail: string,
  ) {
    super(detail);
  }
}

export async function getBaseUrl(): Promise<string> {
  return 'local';
}

const startedAt = Date.now();

export async function getHealth(): Promise<HealthResponse> {
  return {
    status: 'ok',
    model_version: MODEL_VERSION,
    uptime_s: Math.round((Date.now() - startedAt) / 1000),
  };
}

// ------------------------------------------------------------------- model

// Minimal surface of onnxruntime-web that we use (loaded as a global script).
interface OrtTensor {
  data: Float32Array;
  dims: readonly number[];
}
interface OrtSession {
  inputNames: string[];
  outputNames: string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
}
interface OrtGlobal {
  env: { wasm: { wasmPaths: string } };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
  InferenceSession: {
    create(model: ArrayBuffer | string, opts?: object): Promise<OrtSession>;
  };
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.onload = () => resolve();
    el.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(el);
  });
}

let sessionPromise: Promise<{ ort: OrtGlobal; session: OrtSession }> | null = null;

function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      await loadScript(`${ORT_CDN}ort.min.js`);
      const ort = (globalThis as unknown as { ort: OrtGlobal }).ort;
      ort.env.wasm.wasmPaths = ORT_CDN;
      const res = await fetch(MODEL_URL);
      if (!res.ok) throw new Error(`Model download failed (${res.status})`);
      const session = await ort.InferenceSession.create(await res.arrayBuffer(), {
        executionProviders: ['wasm'],
      });
      return { ort, session };
    })();
    // Allow a retry (e.g. after a network blip) instead of caching the failure.
    sessionPromise.catch(() => {
      sessionPromise = null;
    });
  }
  return sessionPromise;
}

// ------------------------------------------------------------------ history

type StoredInference = InferenceDetail;

function readHistory(): StoredInference[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]');
  } catch {
    return [];
  }
}

function writeHistory(items: StoredInference[]) {
  // Oldest entries go first if the browser quota is hit; the demo keeps working
  // even if nothing can be persisted at all.
  let list = items.slice(0, HISTORY_MAX);
  while (list.length) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
      return;
    } catch {
      list = list.slice(0, -1);
    }
  }
}

function toSummary(item: StoredInference): InferenceSummary {
  return {
    inference_id: item.inference_id,
    created_at: item.created_at,
    model_version: item.model_version,
    counts: item.counts,
    num_detections: item.detections.length,
    max_confidence: item.detections.length
      ? Math.max(...item.detections.map((d) => d.confidence))
      : null,
    thumbnail_b64: item.thumbnail_b64,
  };
}

// ---------------------------------------------------------------- inference

async function sha256Hex(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function predictImage(uri: string): Promise<PredictResponse> {
  let blob: Blob;
  let bitmap: ImageBitmap;
  try {
    blob = await (await fetch(uri)).blob();
    // Honors EXIF orientation by default, like the backend's exif_transpose.
    bitmap = await createImageBitmap(blob);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new ApiError(400, `File is not a decodable image: ${detail}`);
  }

  const k = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * k);
  const h = Math.round(bitmap.height * k);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0, w, h);
  const pixels = ctx.getImageData(0, 0, w, h);

  let loaded: Awaited<ReturnType<typeof getSession>>;
  try {
    loaded = await getSession();
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new ApiError(0, `Could not load the model in the browser — ${detail}`);
  }
  const { ort, session } = loaded;

  const result = await segment(
    { data: pixels.data, width: w, height: h },
    async (tensor): Promise<RawOutputs> => {
      const out = await session.run({
        [session.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, 640, 640]),
      });
      const o0 = out[session.outputNames[0]];
      const o1 = out[session.outputNames[1]];
      return { output0: o0.data, output0Dims: o0.dims, output1: o1.data };
    },
  );

  const counts: Record<string, number> = {};
  for (const d of result.detections) counts[d.class_name] = (counts[d.class_name] ?? 0) + 1;

  const response: PredictResponse = {
    inference_id: crypto.randomUUID(),
    created_at: new Date().toISOString(),
    model_version: MODEL_VERSION,
    image: { width: w, height: h, sha256: await sha256Hex(blob) },
    timing_ms: result.timing_ms,
    detections: result.detections,
    counts,
  };

  // Small JPEG preview only — the full radiograph is never persisted.
  const ts = THUMB_SIDE / Math.max(w, h);
  const thumb = document.createElement('canvas');
  thumb.width = Math.max(1, Math.round(w * Math.min(1, ts)));
  thumb.height = Math.max(1, Math.round(h * Math.min(1, ts)));
  thumb.getContext('2d')!.drawImage(canvas, 0, 0, thumb.width, thumb.height);
  const thumbnail_b64 = thumb.toDataURL('image/jpeg', 0.8).split(',')[1] ?? null;

  writeHistory([{ ...response, conf_threshold: 0.5, thumbnail_b64 }, ...readHistory()]);
  return response;
}

export async function listInferences(limit = 20, offset = 0): Promise<InferenceListResponse> {
  const all = readHistory();
  return {
    items: all.slice(offset, offset + limit).map(toSummary),
    total: all.length,
    limit,
    offset,
  };
}

export async function getInference(id: string): Promise<InferenceDetail> {
  const found = readHistory().find((i) => i.inference_id === id);
  if (!found) throw new ApiError(404, 'Inference not found (history is stored in this browser only)');
  return found;
}

export async function deleteInference(id: string): Promise<void> {
  writeHistory(readHistory().filter((i) => i.inference_id !== id));
}

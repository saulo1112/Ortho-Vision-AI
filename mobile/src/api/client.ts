import Constants from 'expo-constants';
import { File } from 'expo-file-system';

import type {
  HealthResponse,
  InferenceDetail,
  InferenceListResponse,
  PredictResponse,
} from './types';

/**
 * The backend can live in several places depending on how the app runs:
 *
 *   - `expo.extra.apiUrl` in app.json — a deployed backend (Render);
 *   - `http://localhost:8000` — dev build over USB with `adb reverse
 *     tcp:8000 tcp:8000` (the `npm run android` script sets this up);
 *   - `http://<metro-host-ip>:8000` — phone and dev machine on the same
 *     Wi-Fi. Note the Metro host IP can point at a virtual adapter
 *     (VirtualBox, WSL) that the phone cannot reach, so this is the last
 *     candidate, not the first.
 *
 * Instead of guessing, we probe each candidate's /health once and stick with
 * the first that answers. The probe result is memoized for the app session.
 */
function candidateUrls(): string[] {
  const candidates: string[] = [];
  const configured = Constants.expoConfig?.extra?.apiUrl as string | undefined;
  if (configured) candidates.push(configured.replace(/\/$/, ''));

  candidates.push('http://localhost:8000');

  const host = Constants.expoConfig?.hostUri?.split(':')[0];
  if (host && host !== 'localhost' && host !== '127.0.0.1') {
    candidates.push(`http://${host}:8000`);
  }
  return candidates;
}

async function probe(url: string, timeoutMs = 2500): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${url}/health`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

let baseUrlPromise: Promise<string> | null = null;

export function getBaseUrl(): Promise<string> {
  if (!baseUrlPromise) {
    baseUrlPromise = (async () => {
      const candidates = candidateUrls();
      for (const url of candidates) {
        const ok = await probe(url);
        console.log(`[api] probe ${url}/health -> ${ok ? 'OK' : 'unreachable'}`);
        if (ok) return url;
      }
      // Nothing answered: forget the memo so the next call re-probes
      // (the server may simply not be up yet), and report what was tried.
      baseUrlPromise = null;
      throw new ApiError(
        0,
        `Cannot reach the server. Tried: ${candidates.join(', ')}. ` +
          'Is the backend running? (see README — "Test on a physical device")',
      );
    })();
    // If resolution fails, allow retrying instead of caching the rejection.
    baseUrlPromise.catch(() => {
      baseUrlPromise = null;
    });
  }
  return baseUrlPromise;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    detail: string,
  ) {
    super(detail);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const baseUrl = await getBaseUrl();
  let res: Response;
  try {
    res = await fetch(`${baseUrl}${path}`, init);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    console.log(`[api] ${init?.method ?? 'GET'} ${baseUrl}${path} failed: ${detail}`);
    throw new ApiError(0, `Cannot reach the server at ${baseUrl} — ${detail}`);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // non-JSON error body — keep the generic message
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export function getHealth(): Promise<HealthResponse> {
  return request('/health');
}

export function predictImage(uri: string): Promise<PredictResponse> {
  // Since SDK 57 the global fetch is expo/fetch (WinterCG), which rejects the
  // classic React Native {uri, name, type} FormData part. The File class from
  // the new expo-file-system implements Blob and wraps the picker's file:// URI.
  let file: File;
  try {
    file = new File(uri);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new ApiError(0, `Cannot open the selected image (${uri}): ${detail}`);
  }
  const form = new FormData();
  form.append('image', file as unknown as Blob, 'radiograph.jpg');
  return request('/v1/predict', { method: 'POST', body: form });
}

export function listInferences(limit = 20, offset = 0): Promise<InferenceListResponse> {
  return request(`/v1/inferences?limit=${limit}&offset=${offset}`);
}

export function getInference(id: string): Promise<InferenceDetail> {
  return request(`/v1/inferences/${id}`);
}

import Constants from 'expo-constants';

import type {
  HealthResponse,
  InferenceDetail,
  InferenceListResponse,
  PredictResponse,
} from './types';

/**
 * Base URL resolution:
 * 1. `expo.extra.apiUrl` in app.json (set this for the deployed backend);
 * 2. in dev, the Metro host IP with port 8000 — so a physical phone on the
 *    same Wi-Fi reaches the backend running on the dev machine with zero config.
 */
function resolveBaseUrl(): string {
  const configured = Constants.expoConfig?.extra?.apiUrl as string | undefined;
  if (configured) return configured.replace(/\/$/, '');
  const host = Constants.expoConfig?.hostUri?.split(':')[0];
  return host ? `http://${host}:8000` : 'http://localhost:8000';
}

export const BASE_URL = resolveBaseUrl();

export class ApiError extends Error {
  constructor(
    public status: number,
    detail: string,
  ) {
    super(detail);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Cannot reach the server at ${BASE_URL}`);
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
  const form = new FormData();
  // React Native FormData file part — not a browser Blob.
  form.append('image', { uri, name: 'radiograph.jpg', type: 'image/jpeg' } as unknown as Blob);
  return request('/v1/predict', { method: 'POST', body: form });
}

export function listInferences(limit = 20, offset = 0): Promise<InferenceListResponse> {
  return request(`/v1/inferences?limit=${limit}&offset=${offset}`);
}

export function getInference(id: string): Promise<InferenceDetail> {
  return request(`/v1/inferences/${id}`);
}

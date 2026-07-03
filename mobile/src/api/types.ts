/** Mirror of the backend contract — see docs/api-contract.md. */

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  id: number;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BBox;
  /** Rings of [x, y] points, normalized to [0, 1] of the original image. */
  polygons: number[][][];
}

export interface PredictResponse {
  inference_id: string;
  created_at: string;
  model_version: string;
  image: { width: number; height: number; sha256: string };
  timing_ms: { preprocess: number; inference: number; postprocess: number; total: number };
  detections: Detection[];
  counts: Record<string, number>;
}

export interface InferenceSummary {
  inference_id: string;
  created_at: string;
  model_version: string;
  counts: Record<string, number>;
  num_detections: number;
  max_confidence: number | null;
  thumbnail_b64: string | null;
}

export interface InferenceDetail extends PredictResponse {
  conf_threshold: number;
  thumbnail_b64: string | null;
}

export interface InferenceListResponse {
  items: InferenceSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  uptime_s: number;
}

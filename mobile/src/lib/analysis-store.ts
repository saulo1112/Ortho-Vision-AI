/**
 * In-memory handoff of the latest analysis from Home to the Result screen.
 * Router params would force serializing a multi-KB payload into the URL;
 * a module-level slot is simpler and the data is intentionally ephemeral
 * (history is served by the backend).
 */
import type { PredictResponse } from '../api/types';

export interface Analysis {
  imageUri: string;
  imageWidth: number;
  imageHeight: number;
  response: PredictResponse;
}

let current: Analysis | null = null;

export function setAnalysis(analysis: Analysis) {
  current = analysis;
}

export function getAnalysis(): Analysis | null {
  return current;
}

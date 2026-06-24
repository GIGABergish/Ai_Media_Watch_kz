import { DemoCase } from '../types';
import { API_BASE } from '../config';

export interface AnalysisMeta {
  engineMode: string;
  degraded: string[];
  lanesRun: string[];
  elapsedMs: number;
  components: Record<string, number>;
  notes: string[];
}

export interface AnalysisResponse {
  case: DemoCase;
  meta: AnalysisMeta;
}

export interface AnalyzeUrlBody {
  url: string;
  title?: string;
  platform?: string;
  description?: string;
  hashtags?: string;
}

async function ensureOk(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res;
}

export async function health(): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/health`);
  await ensureOk(res);
  return res.json();
}

export async function analyzeFile(
  file: File,
  opts?: { title?: string; platform?: string; description?: string; hashtags?: string },
): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append('file', file);
  if (opts?.title) form.append('title', opts.title);
  if (opts?.platform) form.append('platform', opts.platform);
  if (opts?.description) form.append('description', opts.description);
  if (opts?.hashtags) form.append('hashtags', opts.hashtags);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: form,
  });
  await ensureOk(res);
  return res.json() as Promise<AnalysisResponse>;
}

export async function analyzeUrl(body: AnalyzeUrlBody): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/analyze/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  await ensureOk(res);
  return res.json() as Promise<AnalysisResponse>;
}

export async function listCases(): Promise<DemoCase[]> {
  const res = await fetch(`${API_BASE}/api/cases`);
  await ensureOk(res);
  return res.json() as Promise<DemoCase[]>;
}

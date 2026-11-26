const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const API_KEY = process.env.NEXT_PUBLIC_SAFESPACE_KEY || "";

export function getApiUrl(): string {
  return API_URL.replace(/\/$/, "");
}

/** Append ?key= to media URLs (img/video cannot send headers). No-op without a key. */
export function withKey(url: string): string {
  if (!API_KEY) return url;
  return `${url}${url.includes("?") ? "&" : "?"}key=${encodeURIComponent(API_KEY)}`;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function isAuthError(err: unknown): boolean {
  return err instanceof ApiError && err.status === 401;
}

const AUTH_MSG = "API key missing or invalid (check NEXT_PUBLIC_SAFESPACE_KEY)";


export async function readError(res: Response): Promise<string> {
  if (res.status === 401) return AUTH_MSG;
  let text = "";
  try {
    text = await res.text();
  } catch {
    return res.statusText || "Request failed";
  }
  try {
    const data = JSON.parse(text);
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail)) {
      return data.detail
        .map((d: { msg?: string }) => d.msg || JSON.stringify(d))
        .join("; ");
    }
    return JSON.stringify(data);
  } catch {
    return text.trim().slice(0, 300) || res.statusText || `Request failed (${res.status})`;
  }
}

export type ApiInit = RequestInit & { timeoutMs?: number };

/** fetch wrapper: adds auth header, optional timeout, throws ApiError on !ok. */
export async function apiFetch(path: string, init: ApiInit = {}): Promise<Response> {
  const { timeoutMs, signal, headers, ...rest } = init;
  const h = new Headers(headers);
  if (API_KEY) h.set("X-SafeSpace-Key", API_KEY);

  const ctrl = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  if (timeoutMs) {
    timer = setTimeout(() => {
      timedOut = true;
      ctrl.abort();
    }, timeoutMs);
  }
  if (signal) {
    if (signal.aborted) ctrl.abort();
    else signal.addEventListener("abort", () => ctrl.abort(), { once: true });
  }
  try {
    const res = await fetch(`${getApiUrl()}${path}`, {
      cache: "no-store",
      ...rest,
      headers: h,
      signal: ctrl.signal,
    });
    if (!res.ok) throw new ApiError(await readError(res), res.status);
    return res;
  } catch (err) {
    if (timedOut) throw new Error("Request timed out");
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

const cid = encodeURIComponent;

export function feedUrl(cameraId: string, bust?: number): string {
  const u = `${getApiUrl()}/cameras/${cid(cameraId)}/feed${bust ? `?t=${bust}` : ""}`;
  return withKey(u);
}

export function evidenceUrl(pathOrName: string): string {
  const name = pathOrName.split("/").pop() || pathOrName;
  return withKey(`${getApiUrl()}/evidence/${cid(name)}`);
}

export type ModelThresholds = {
  spike_threshold: number;
  alert_threshold: number;
  score_history_len: number;
};

export type RootResponse = {
  watcher_last_cycle_age_ms?: number | null;
  model_version?: string | null;
  model_name?: string | null;
  preprocessing?: string | null;
  thresholds?: ModelThresholds;
  thresholds_from_sidecar?: boolean;
  auth_enabled?: boolean;
  [k: string]: unknown;
};

export type StatusResponse = {
  camera_id: string;
  label: string;
  backend_online: boolean;
  camera_active: boolean;
  camera_receiving: boolean;
  camera_kind?: "phone" | "webcam" | "none";
  last_frame_age_ms: number | null;
  stream_url: string | null;
  wifi_name: string;
  ip: string;
  port: number;
  path: string;
  buffer_fill: string;
  total_frames: number;
  latest_raw_pct: number | null;
  smoothed_avg_pct: number | null;
  alert_threshold: number;
  open_error: string | null;
  target_fps?: number;
  score_warmup?: string;
  framing_suitable_ratio?: number | null;
  actual_fps?: number | null;
  degraded?: boolean;
  frozen?: boolean;
  offline_since?: string | number | null;
  coverage_pct?: number | null;
  generation?: number;
  via_bridge?: boolean;
  capture_url?: string | null;
  scene?: {
    suitable_for_model?: boolean;
    subject_area_frac?: number;
    method?: string;
    note?: string;
    input_width?: number;
    input_height?: number;
  };
};

export type CamerasResponse = {
  total: number;
  cameras: StatusResponse[];
};

export type AlertItem = {
  timestamp: string;
  confidence: number;
  evidence_file: string;
  evidence_url?: string | null;
  camera_id?: string;
  status?: "recording" | "processing" | "ready" | "failed";
  trigger?: string;
  peak?: number;
  n_triggers?: number;
  end_timestamp?: string;
  model_version?: string;
  scene?: StatusResponse["scene"];
};

export type LogItem = {
  timestamp: string;
  level: string;
  message: string;
  kind: string;
};

export type ConnectPayload = {
  ip: string;
  port?: number;
  path?: string;
  wifi_name?: string;
};

export async function fetchRoot(signal?: AbortSignal, timeoutMs = 4000): Promise<RootResponse> {
  return (await apiFetch("/", { signal, timeoutMs })).json();
}

export async function fetchCameras(signal?: AbortSignal, timeoutMs = 4000): Promise<CamerasResponse> {
  return (await apiFetch("/cameras", { signal, timeoutMs })).json();
}

export async function fetchStatus(cameraId: string, signal?: AbortSignal): Promise<StatusResponse> {
  return (await apiFetch(`/cameras/${cid(cameraId)}/status`, { signal })).json();
}

export async function fetchAlerts(
  cameraId?: string,
  signal?: AbortSignal,
  timeoutMs = 4000
): Promise<{ total_alerts: number; recent: AlertItem[] }> {
  const q = cameraId ? `?camera_id=${cid(cameraId)}` : "";
  return (await apiFetch(`/alerts${q}`, { signal, timeoutMs })).json();
}

export async function fetchLogs(signal?: AbortSignal, timeoutMs = 4000): Promise<{ total: number; recent: LogItem[] }> {
  return (await apiFetch("/logs", { signal, timeoutMs })).json();
}

const JSON_HEADERS = { "Content-Type": "application/json" };

export async function createCamera(cameraId: string, label: string): Promise<{ ok: boolean; camera_id: string }> {
  return (
    await apiFetch("/cameras", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ camera_id: cameraId, label }),
    })
  ).json();
}

export async function deleteCamera(cameraId: string): Promise<unknown> {
  return (await apiFetch(`/cameras/${cid(cameraId)}`, { method: "DELETE" })).json();
}

export async function connectCamera(cameraId: string, payload: ConnectPayload): Promise<unknown> {
  return (
    await apiFetch(`/cameras/${cid(cameraId)}/connect`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    })
  ).json();
}

export async function connectWebcam(cameraId: string): Promise<unknown> {
  return (
    await apiFetch(`/cameras/${cid(cameraId)}/connect-webcam`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({}),
    })
  ).json();
}

export async function disconnectCamera(cameraId: string): Promise<unknown> {
  return (await apiFetch(`/cameras/${cid(cameraId)}/disconnect`, { method: "POST" })).json();
}

export type AnalyzeResult = {
  ok: boolean;
  verdict: "would_alert" | "safe";
  would_alert: boolean;
  peak_raw_pct: number;
  peak_smoothed_pct: number;
  alert_threshold_pct: number;
  fps: number;
  frames_analyzed: number;
  duration_sec: number;
  explanation: string;
  samples: Array<{
    t_sec: number;
    frame: number;
    raw_pct: number;
    smoothed_pct: number | null;
    above_threshold: boolean;
  }>;
  saved_as?: string;
  evidence_file?: string;
};

export async function analyzeVideoFile(file: File): Promise<AnalyzeResult> {
  const body = new FormData();
  body.append("file", file);
  return (await apiFetch("/analyze/video", { method: "POST", body, timeoutMs: 300000 })).json();
}

export async function analyzeEvidence(filename: string): Promise<AnalyzeResult> {
  const name = filename.split("/").pop() || filename;
  return (await apiFetch(`/analyze/evidence/${cid(name)}`, { method: "POST", timeoutMs: 300000 })).json();
}

export async function labelEvidence(
  filename: string,
  label: "safe" | "unsafe"
): Promise<{ ok: boolean; copied_to: string; hint: string }> {
  const name = filename.split("/").pop() || filename;
  return (
    await apiFetch(`/review/label-evidence/${cid(name)}`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ label }),
    })
  ).json();
}

// ---------------------------------------------------------------- calibration
export const CALIBRATION_TIMEOUT_MS = 30 * 60 * 1000;

export type CalibrationSettings = {
  target_fps: number;
  spike_threshold: number;
  alert_threshold: number;
  history_len: number;
  pad_seconds: number;
  merge_gap_seconds: number;
};

export type CalibrationEvent = {
  start_sec: number;
  end_sec: number;
  peak_score: number;
  trigger: string;
  url: string;
  clip_available?: boolean;
};

export type CalibrationSample = {
  time_sec: number;
  raw_score: number;
  smoothed_avg: number;
  triggered?: boolean;
  trigger_reason?: string;
  suitable_for_model?: boolean;
  global_motion?: boolean;
};

export type CalibrationModel = {
  version: string | null;
  name: string | null;
  sha256_12: string | null;
  preprocessing: string | null;
  legacy_model_type?: string;
};

/** A saved run: schema 2 records carry model + settings; schema 1 (older runs) do not. */
export type CalibrationRun = {
  schema: number;
  run_id: string;
  created_at: string;
  created_ts: number;
  video_name: string;
  video_url: string | null;
  video_available: boolean;
  source?: { kind: string; url?: string; original_name?: string; rerun_of?: string | null } | null;
  model: CalibrationModel;
  settings: CalibrationSettings | null;
  summary: { duration_sec: number; n_windows: number; n_events: number; peak_raw: number; framing_suitable_ratio: number | null };
  native_fps: number;
  events: CalibrationEvent[];
  per_sample_log: CalibrationSample[];
};

export type CalibrationRunSummary = {
  run_id: string;
  schema: number;
  created_at: string;
  created_ts: number;
  video_name: string;
  video_available: boolean;
  source: string | null;
  model_name: string | null;
  model_version: string | null;
  preprocessing: string | null;
  spike_threshold: number | null;
  alert_threshold: number | null;
  history_len: number | null;
  n_events: number;
  duration_sec: number | null;
  n_windows: number | null;
  peak_raw: number | null;
};

function appendSettings(body: FormData, s: CalibrationSettings): void {
  body.append("target_fps", String(s.target_fps));
  body.append("spike_threshold", String(s.spike_threshold));
  body.append("alert_threshold", String(s.alert_threshold));
  body.append("history_len", String(s.history_len));
  body.append("pad_seconds", String(s.pad_seconds));
  body.append("merge_gap_seconds", String(s.merge_gap_seconds));
}

/** Calibrate a new video (file upload or YouTube URL) with the backend's loaded model. */
export async function runCalibration(
  source: { file: File } | { youtubeUrl: string },
  settings: CalibrationSettings,
  signal?: AbortSignal
): Promise<CalibrationRun> {
  const body = new FormData();
  if ("file" in source) body.append("video_file", source.file);
  else body.append("youtube_url", source.youtubeUrl);
  appendSettings(body, settings);
  return (await apiFetch("/calibration/run", { method: "POST", body, signal, timeoutMs: CALIBRATION_TIMEOUT_MS })).json();
}

/** Re-calibrate a video already on the server, with the loaded model and new settings. */
export async function rerunCalibration(
  videoName: string,
  settings: CalibrationSettings,
  rerunOf?: string,
  signal?: AbortSignal
): Promise<CalibrationRun> {
  const body = new FormData();
  body.append("video_name", videoName);
  if (rerunOf) body.append("rerun_of", rerunOf);
  appendSettings(body, settings);
  return (await apiFetch("/calibration/rerun", { method: "POST", body, signal, timeoutMs: CALIBRATION_TIMEOUT_MS })).json();
}

export async function fetchCalibrationRuns(signal?: AbortSignal): Promise<CalibrationRunSummary[]> {
  const data = await (await apiFetch("/calibration/runs", { signal, timeoutMs: 15000 })).json();
  return (data.runs || []) as CalibrationRunSummary[];
}

export async function fetchCalibrationRun(runId: string, signal?: AbortSignal): Promise<CalibrationRun> {
  return (await apiFetch(`/calibration/runs/${cid(runId)}`, { signal, timeoutMs: 30000 })).json();
}

export function calibrationMediaUrl(path: string): string {
  return withKey(`${getApiUrl()}${path}`);
}

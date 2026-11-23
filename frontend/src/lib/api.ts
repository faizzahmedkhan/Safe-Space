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

export type RootResponse = {
  watcher_last_cycle_age_ms?: number | null;
  model_version?: string | null;
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

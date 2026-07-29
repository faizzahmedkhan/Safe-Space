const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function getApiUrl(): string {
  return API_URL.replace(/\/$/, "");
}

export function feedUrl(): string {
  return `${getApiUrl()}/feed`;
}

export function evidenceUrl(pathOrName: string): string {
  const name = pathOrName.split("/").pop() || pathOrName;
  return `${getApiUrl()}/evidence/${name}`;
}

export type StatusResponse = {
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

export type AlertItem = {
  timestamp: string;
  confidence: number;
  evidence_file: string;
  evidence_url?: string | null;
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

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((d: { msg?: string }) => d.msg || JSON.stringify(d))
        .join("; ");
    }
    return JSON.stringify(data);
  } catch {
    return res.statusText || "Request failed";
  }
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch(`${getApiUrl()}/status`, { cache: "no-store" });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchAlerts(): Promise<{
  total_alerts: number;
  recent: AlertItem[];
}> {
  const res = await fetch(`${getApiUrl()}/alerts`, { cache: "no-store" });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchLogs(): Promise<{ total: number; recent: LogItem[] }> {
  const res = await fetch(`${getApiUrl()}/logs`, { cache: "no-store" });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function connectCamera(payload: ConnectPayload): Promise<unknown> {
  const res = await fetch(`${getApiUrl()}/camera/connect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function connectWebcam(): Promise<unknown> {
  const res = await fetch(`${getApiUrl()}/camera/connect-webcam`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function disconnectCamera(): Promise<unknown> {
  const res = await fetch(`${getApiUrl()}/camera/disconnect`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
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
  const res = await fetch(`${getApiUrl()}/analyze/video`, {
    method: "POST",
    body,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function analyzeEvidence(
  filename: string
): Promise<AnalyzeResult> {
  const name = filename.split("/").pop() || filename;
  const res = await fetch(`${getApiUrl()}/analyze/evidence/${name}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function labelEvidence(
  filename: string,
  label: "safe" | "unsafe"
): Promise<{ ok: boolean; copied_to: string; hint: string }> {
  const name = filename.split("/").pop() || filename;
  const res = await fetch(`${getApiUrl()}/review/label-evidence/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

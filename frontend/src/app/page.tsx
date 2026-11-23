"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Play, Plus, Upload, X } from "lucide-react";
import {
  AlertItem,
  AnalyzeResult,
  LogItem,
  StatusResponse,
  CamerasResponse,
  analyzeEvidence,
  analyzeVideoFile,
  connectCamera,
  connectWebcam,
  disconnectCamera,
  createCamera,
  deleteCamera,
  fetchAlerts,
  fetchLogs,
  fetchCameras,
  fetchRoot,
  isAuthError,
  labelEvidence,
  RootResponse,
} from "@/lib/api";
import AddCameraDialog from "@/components/AddCameraDialog";
import AlertFeed, { type Scope, type Tab, type Verdict } from "@/components/AlertFeed";
import AppBar from "@/components/AppBar";
import CameraWall from "@/components/CameraWall";
import MonitorPanel from "@/components/MonitorPanel";
import StatBand from "@/components/StatBand";
import styles from "./page.module.css";

const POLL_MS = 1000;
const REQ_TIMEOUT_MS = 4000;
const FEED_RETRY_MS = 5000;

type EndpointErrors = {
  root: string | null;
  cameras: string | null;
  alerts: string | null;
  logs: string | null;
};
const NO_ERRORS: EndpointErrors = { root: null, cameras: null, alerts: null, logs: null };

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : "Request failed";
}

export default function DashboardPage() {
  const [cameras, setCameras] = useState<StatusResponse[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const hasLoaded = useRef(false);
  const [errors, setErrors] = useState<EndpointErrors>(NO_ERRORS);
  const [authFailed, setAuthFailed] = useState(false);
  const [root, setRoot] = useState<RootResponse | null>(null);
  const [feedLost, setFeedLost] = useState(false);
  const [feedBust, setFeedBust] = useState(0);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [tab, setTab] = useState<Tab>("alerts");
  const [scope, setScope] = useState<Scope>("selected");
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [addOpen, setAddOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Add camera form state
  const [newCameraId, setNewCameraId] = useState("");
  const [newCameraLabel, setNewCameraLabel] = useState("");
  const [ip, setIp] = useState("");
  const [port, setPort] = useState("8080");
  const [path, setPath] = useState("/video");
  const [wifiName, setWifiName] = useState("");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  
  const [feedKey, setFeedKey] = useState(0);
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewResult, setReviewResult] = useState<AnalyzeResult | null>(null);
  const [labelMsg, setLabelMsg] = useState<string | null>(null);

  const backendOnline = errors.cameras === null && cameras.length >= 0 && !authFailed && hasLoaded.current;

  const selectedRef = useRef<string | null>(null);
  const scopeRef = useRef<Scope>("selected");
  const reqIdRef = useRef(0);
  const ctrlRef = useRef<AbortController | null>(null);
  const inFlightRef = useRef(false);

  useEffect(() => {
    selectedRef.current = selectedCameraId;
  }, [selectedCameraId]);

  useEffect(() => {
    scopeRef.current = scope;
  }, [scope]);

  // Runs one poll. A newer call (e.g. camera switch) aborts and supersedes an older one.
  const refresh = useCallback(async (supersede = false) => {
    if (inFlightRef.current && !supersede) return;
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    const reqId = ++reqIdRef.current;
    const camAtStart = selectedRef.current;
    const scopeAtStart = scopeRef.current;
    inFlightRef.current = true;
    try {
      const [r, c, a, l] = await Promise.allSettled([
        fetchRoot(ctrl.signal, REQ_TIMEOUT_MS),
        fetchCameras(ctrl.signal, REQ_TIMEOUT_MS),
        fetchAlerts(scopeAtStart === "all" ? undefined : camAtStart || undefined, ctrl.signal, REQ_TIMEOUT_MS),
        fetchLogs(ctrl.signal, REQ_TIMEOUT_MS),
      ]);
      // Drop stale responses (superseded request or camera switched meanwhile).
      if (reqId !== reqIdRef.current) return;
      hasLoaded.current = true;
      const next: EndpointErrors = {
        root: r.status === "rejected" ? errMsg(r.reason) : null,
        cameras: c.status === "rejected" ? errMsg(c.reason) : null,
        alerts: a.status === "rejected" ? errMsg(a.reason) : null,
        logs: l.status === "rejected" ? errMsg(l.reason) : null,
      };
      setErrors(next);
      setAuthFailed([r, c, a, l].some((x) => x.status === "rejected" && isAuthError(x.reason)));
      if (r.status === "fulfilled") setRoot(r.value);
      if (l.status === "fulfilled") setLogs(l.value.recent);
      if (a.status === "fulfilled" && camAtStart === selectedRef.current && scopeAtStart === scopeRef.current) {
        setAlerts(a.value.recent);
      }
      if (c.status === "fulfilled") {
        const list = c.value.cameras;
        setCameras(list);
        const cur = selectedRef.current;
        if (!cur || !list.some((x) => x.camera_id === cur)) {
          const nextId = list.length > 0 ? list[0].camera_id : null;
          if (nextId !== cur) {
            selectedRef.current = nextId;
            setSelectedCameraId(nextId);
          }
        }
      }
    } finally {
      if (reqId === reqIdRef.current) inFlightRef.current = false;
    }
  }, []);

  // Poll loop: sequential (no overlap), paused while the tab is hidden.
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      if (stopped) return;
      if (!document.hidden) await refresh(true);
      if (!stopped) timer = setTimeout(tick, POLL_MS);
    };
    const onVis = () => {
      if (!document.hidden) {
        clearTimeout(timer);
        void tick();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    void tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
      ctrlRef.current?.abort();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [refresh]);

  // Camera switch: clear stale alerts, reset feed state, refetch immediately.
  useEffect(() => {
    setAlerts([]);
    setSelectedEvidence(null);
    setFeedLost(false);
    void refresh(true);
  }, [selectedCameraId, scope, refresh]);

  // Feed lost: retry periodically with a cache-busting param.
  useEffect(() => {
    if (!feedLost) return;
    const id = setInterval(() => setFeedBust((b) => b + 1), FEED_RETRY_MS);
    return () => clearInterval(id);
  }, [feedLost]);

  async function onAddCamera(e: FormEvent) {
    e.preventDefault();
    setActionError(null);
    setBusy(true);
    try {
      const cid = newCameraId.trim();
      if (!cid) throw new Error("Camera ID is required");
      
      // 1. Create slot
      await createCamera(cid, newCameraLabel.trim());

      // 2. Connect. If this fails, roll back the slot we just created so the
      // user can resubmit the same ID instead of hitting "already exists".
      if (ip.trim()) {
        try {
          await connectCamera(cid, {
            ip: ip.trim(),
            port: Number(port) || 8080,
            path: path.trim() || "/video",
            wifi_name: wifiName.trim(),
          });
        } catch (connErr) {
          try {
            await deleteCamera(cid);
          } catch {
            /* best effort; slot may remain and can be removed from the list */
          }
          throw new Error(`Connect failed (camera slot removed): ${errMsg(connErr)}`);
        }
      }
      setSelectedCameraId(cid);
      setNewCameraId("");
      setNewCameraLabel("");
      setIp("");
      setFeedKey((k) => k + 1);
      setAddOpen(false);
      await refresh(true);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Add camera failed");
    } finally {
      setBusy(false);
    }
  }

  async function onAddWebcam() {
    setActionError(null);
    setBusy(true);
    try {
      const cid = "webcam-" + Math.floor(Math.random() * 1000);
      await createCamera(cid, "Laptop Webcam");
      try {
        await connectWebcam(cid);
      } catch (connErr) {
        try {
          await deleteCamera(cid);
        } catch {
          /* best effort */
        }
        throw new Error(`Webcam connect failed (slot removed): ${errMsg(connErr)}`);
      }
      setSelectedCameraId(cid);
      setFeedKey((k) => k + 1);
      setAddOpen(false);
      await refresh(true);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Webcam connect failed");
    } finally {
      setBusy(false);
    }
  }

  async function onDisconnect(cameraId: string) {
    setActionError(null);
    setBusy(true);
    try {
      await disconnectCamera(cameraId);
      setFeedKey((k) => k + 1);
      await refresh(true);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Disconnect failed");
    } finally {
      setBusy(false);
    }
  }

  async function onRemoveCamera(cameraId: string) {
    if (!window.confirm("Remove this camera completely?")) return;
    setActionError(null);
    setBusy(true);
    try {
      await deleteCamera(cameraId);
      if (selectedCameraId === cameraId) {
        selectedRef.current = null;
        setSelectedCameraId(null);
      }
      await refresh(true);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Remove failed");
    } finally {
      setBusy(false);
    }
  }

  async function onReviewFile(file: File | null) {
    if (!file) return;
    setReviewError(null);
    setLabelMsg(null);
    setReviewBusy(true);
    try {
      const result = await analyzeVideoFile(file);
      setReviewResult(result);
    } catch (err) {
      setReviewResult(null);
      setReviewError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setReviewBusy(false);
    }
  }

  async function onReanalyze(evidenceFile: string) {
    setReviewError(null);
    setLabelMsg(null);
    setReviewBusy(true);
    try {
      const result = await analyzeEvidence(evidenceFile);
      setReviewResult(result);
    } catch (err) {
      setReviewResult(null);
      setReviewError(err instanceof Error ? err.message : "Re-analyze failed");
    } finally {
      setReviewBusy(false);
    }
  }

  async function onLabel(evidenceFile: string, label: Verdict) {
    setLabelMsg(null);
    try {
      const res = await labelEvidence(evidenceFile, label);
      setVerdicts((v) => ({ ...v, [evidenceFile]: label }));
      setLabelMsg(
        label === "safe"
          ? `Marked false alarm → ${res.copied_to}. ${res.hint}`
          : `Marked true incident → ${res.copied_to}. ${res.hint}`
      );
    } catch (err) {
      setLabelMsg(err instanceof Error ? err.message : "Label failed");
    }
  }

  const selectedCamStatus = cameras.find(c => c.camera_id === selectedCameraId);

  const cameraConnected = Boolean(
    selectedCamStatus?.camera_active && selectedCamStatus?.camera_receiving
  );
  
  const cameraKind = selectedCamStatus?.camera_kind || "none";
  let cameraLabelMsg = "Unknown";
  if (authFailed) cameraLabelMsg = "API key missing/invalid";
  else if (!backendOnline) cameraLabelMsg = "Start the FastAPI backend";
  else if (selectedCamStatus?.frozen) cameraLabelMsg = "Feed frozen";
  else if (cameraConnected) {
    if (cameraKind === "webcam") cameraLabelMsg = "Laptop webcam connected";
    else if (selectedCamStatus?.wifi_name) cameraLabelMsg = `Connected on ${selectedCamStatus.wifi_name}`;
    else cameraLabelMsg = "Connected — frames arriving";
  } else if (selectedCamStatus?.camera_active) {
    cameraLabelMsg = "Connecting — waiting for frames";
  } else {
    cameraLabelMsg = "Disconnected";
  }

  const backendState = authFailed ? "unauthorized" : backendOnline ? "online" : "offline";
  const onlineCount = cameras.filter((c) => c.camera_receiving).length;
  const hotCameras = cameras.filter(
    (c) => c.camera_receiving && c.smoothed_avg_pct != null && c.smoothed_avg_pct >= c.alert_threshold
  );
  const hotCam = hotCameras[0];
  const reviewVisible = reviewBusy || reviewError || reviewResult;

  return (
    <>
      <AppBar
        active="monitor"
        backend={{ state: backendState }}
        watcherAgeMs={root?.watcher_last_cycle_age_ms}
        modelVersion={root?.model_version}
      />

      <div className="gutter">
        <div className="wrap">
          <div className={styles.head}>
            <div>
              <h1>Live monitoring</h1>
              <p className={styles.sub}>
                {cameras.length} {cameras.length === 1 ? "camera" : "cameras"}, {hotCameras.length} above the alert
                threshold.
              </p>
            </div>
            <div className={styles.headActions}>
              <input
                ref={fileInputRef}
                type="file"
                hidden
                accept="video/mp4,video/*,.mp4,.mov,.avi,.mkv,.webm"
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  void onReviewFile(f);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                className="btn"
                disabled={reviewBusy || !backendOnline}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload aria-hidden="true" />
                {reviewBusy ? "Analyzing…" : "Upload video"}
              </button>
              <button type="button" className="btn btn-primary" onClick={() => setAddOpen(true)}>
                <Plus aria-hidden="true" />
                Add camera
              </button>
            </div>
          </div>

          {authFailed && (
            <p className={styles.error} role="alert">
              API key missing or invalid. Set NEXT_PUBLIC_SAFESPACE_KEY (see .env.example) and restart the frontend.
            </p>
          )}
          {!authFailed && backendOnline && (errors.alerts || errors.logs || errors.root) && (
            <p className={styles.notice} role="status">
              {[
                errors.alerts && `Alerts: ${errors.alerts}`,
                errors.logs && `Logs: ${errors.logs}`,
                errors.root && `Status: ${errors.root}`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
          )}
          {!authFailed && !backendOnline && hasLoaded.current && errors.cameras && (
            <p className={styles.error} role="alert">
              Cameras: {errors.cameras}
            </p>
          )}

          {hotCam && (
            <div className={styles.alertbar} role="alert">
              <i className="dot dot-crit" />
              <b>Possible fight on {hotCam.label || hotCam.camera_id}</b>
              <span className="mono">
                {hotCam.smoothed_avg_pct?.toFixed(0)}% confidence, above alert threshold
              </span>
              <button
                type="button"
                className="btn btn-danger btn-sm"
                onClick={() => setSelectedCameraId(hotCam.camera_id)}
              >
                <Play aria-hidden="true" />
                Watch live
              </button>
            </div>
          )}

          <StatBand
            online={onlineCount}
            total={cameras.length}
            active={hotCameras.length}
            alerts={alerts.length}
            latestConfidence={selectedCamStatus?.smoothed_avg_pct ?? null}
          />

          <main className={styles.grid}>
            <div className={styles.stack}>
              <section className={styles.panel} aria-labelledby="cameras-heading">
                <div className={styles.panelHead}>
                  <h2 id="cameras-heading">Cameras</h2>
                  <span className="label">Red = above alert threshold</span>
                </div>
                <CameraWall
                  cameras={cameras}
                  selectedId={selectedCameraId}
                  onSelect={setSelectedCameraId}
                  onAdd={() => setAddOpen(true)}
                />
              </section>

              {selectedCamStatus && (
                <div className={styles.panel}>
                  <MonitorPanel
                    cam={selectedCamStatus}
                    connected={cameraConnected}
                    stateMessage={cameraLabelMsg}
                    busy={busy}
                    feedKey={feedKey}
                    feedBust={feedBust}
                    feedLost={feedLost}
                    onFeedError={() => setFeedLost(true)}
                    onFeedLoad={() => setFeedLost(false)}
                    onDisconnect={() => onDisconnect(selectedCamStatus.camera_id)}
                    onRemove={() => onRemoveCamera(selectedCamStatus.camera_id)}
                  />
                </div>
              )}

              {reviewVisible && (
                <section className={styles.panel} aria-labelledby="review-heading">
                  <div className={styles.panelHead}>
                    <h2 id="review-heading">Recording review</h2>
                    {!reviewBusy && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => {
                          setReviewResult(null);
                          setReviewError(null);
                        }}
                      >
                        <X aria-hidden="true" />
                        Dismiss
                      </button>
                    )}
                  </div>
                  {reviewBusy && <p className={styles.sub}>Analyzing…</p>}
                  {reviewError && <p className={styles.error}>{reviewError}</p>}
                  {reviewResult && (
                    <div className={styles.result}>
                      <p className={reviewResult.would_alert ? styles.verdictHot : styles.verdictOk}>
                        {reviewResult.would_alert
                          ? "Would ALERT (model thinks this is violence)"
                          : "Would stay SAFE (model does not alert)"}
                      </p>
                      <p className={`${styles.sub} mono`}>
                        Peak smoothed {reviewResult.peak_smoothed_pct}% · Peak raw {reviewResult.peak_raw_pct}% ·
                        Threshold {reviewResult.alert_threshold_pct}% · {reviewResult.duration_sec}s analyzed
                      </p>
                      <p className={styles.sub}>{reviewResult.explanation}</p>
                    </div>
                  )}
                </section>
              )}
            </div>

            <aside className={styles.panel}>
              <AlertFeed
                tab={tab}
                onTab={setTab}
                scope={scope}
                onScope={setScope}
                hasSelectedCamera={Boolean(selectedCameraId)}
                alerts={alerts}
                logs={logs}
                openEvidence={selectedEvidence}
                onOpen={setSelectedEvidence}
                verdicts={verdicts}
                reviewBusy={reviewBusy}
                backendOnline={backendOnline}
                onReanalyze={(f) => void onReanalyze(f)}
                onLabel={(f, l) => void onLabel(f, l)}
                labelMsg={labelMsg}
              />
            </aside>
          </main>
        </div>
      </div>

      <AddCameraDialog
        open={addOpen}
        onClose={() => {
          setAddOpen(false);
          setActionError(null);
        }}
        onSubmit={onAddCamera}
        onAddWebcam={onAddWebcam}
        busy={busy}
        backendOnline={backendOnline}
        error={actionError}
        id={newCameraId}
        setId={setNewCameraId}
        label={newCameraLabel}
        setLabel={setNewCameraLabel}
        ip={ip}
        setIp={setIp}
        port={port}
        setPort={setPort}
        path={path}
        setPath={setPath}
      />
    </>
  );
}

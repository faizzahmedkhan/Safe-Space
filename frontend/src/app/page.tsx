"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
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
  evidenceUrl,
  feedUrl,
  fetchAlerts,
  fetchLogs,
  fetchStatus,
  fetchCameras,
  labelEvidence,
} from "@/lib/api";
import styles from "./page.module.css";

type Tab = "alerts" | "logs";

export default function DashboardPage() {
  const [cameras, setCameras] = useState<StatusResponse[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [tab, setTab] = useState<Tab>("alerts");

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

  const refresh = useCallback(async () => {
    try {
      const [c, a, l] = await Promise.all([
        fetchCameras(),
        fetchAlerts(selectedCameraId || undefined),
        fetchLogs(),
      ]);
      setCameras(c.cameras);
      setBackendOnline(true);
      setAlerts(a.recent);
      setLogs(l.recent);

      if (!selectedCameraId && c.cameras.length > 0) {
        setSelectedCameraId(c.cameras[0].camera_id);
      }
    } catch {
      setBackendOnline(false);
    }
  }, [selectedCameraId]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, [refresh]);

  async function onAddCamera(e: FormEvent) {
    e.preventDefault();
    setActionError(null);
    setBusy(true);
    try {
      const cid = newCameraId.trim();
      if (!cid) throw new Error("Camera ID is required");
      
      // 1. Create slot
      await createCamera(cid, newCameraLabel.trim());
      
      // 2. Connect
      if (ip.trim()) {
        await connectCamera(cid, {
          ip: ip.trim(),
          port: Number(port) || 8080,
          path: path.trim() || "/video",
          wifi_name: wifiName.trim(),
        });
      }
      setSelectedCameraId(cid);
      setNewCameraId("");
      setNewCameraLabel("");
      setIp("");
      setFeedKey((k) => k + 1);
      await refresh();
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
      await connectWebcam(cid);
      setSelectedCameraId(cid);
      setFeedKey((k) => k + 1);
      await refresh();
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
      await refresh();
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
        setSelectedCameraId(null);
      }
      await refresh();
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

  async function onLabel(evidenceFile: string, label: "safe" | "unsafe") {
    setLabelMsg(null);
    try {
      const res = await labelEvidence(evidenceFile, label);
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
  if (!backendOnline) cameraLabelMsg = "Start the FastAPI backend";
  else if (cameraConnected) {
    if (cameraKind === "webcam") cameraLabelMsg = "Laptop webcam connected";
    else if (selectedCamStatus?.wifi_name) cameraLabelMsg = `Connected on ${selectedCamStatus.wifi_name}`;
    else cameraLabelMsg = "Connected — frames arriving";
  } else if (selectedCamStatus?.camera_active) {
    cameraLabelMsg = "Connecting — waiting for frames";
  } else {
    cameraLabelMsg = "Disconnected";
  }

  const raw = selectedCamStatus?.latest_raw_pct;
  const smoothed = selectedCamStatus?.smoothed_avg_pct;
  const threshold = selectedCamStatus?.alert_threshold ?? 60;
  const scoreHot = smoothed != null && smoothed >= threshold;

  return (
    <div className={styles.layoutContainer}>
      {/* Sidebar for Cameras */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarSection}>
          <h2 className={styles.sidebarTitle}>Cameras ({cameras.length})</h2>
          {cameras.length === 0 && (
            <p className={styles.hint}>No cameras configured yet.</p>
          )}
          <div className={styles.cameraList}>
            {cameras.map((c) => (
              <div 
                key={c.camera_id} 
                className={`${styles.cameraCard} ${selectedCameraId === c.camera_id ? styles.cameraCardActive : ''}`}
                onClick={() => setSelectedCameraId(c.camera_id)}
              >
                <div className={styles.cameraCardHeader}>
                  <h3 className={styles.cameraName}>{c.label || c.camera_id}</h3>
                  <div className={`${styles.cameraStatus} ${c.camera_receiving ? styles.active : styles.offline}`}>
                    <span className={styles.dot} />
                    {c.camera_receiving ? "Live" : "Offline"}
                  </div>
                </div>
                <p className={styles.cameraMeta}>
                  {c.camera_kind === 'webcam' ? 'Webcam' : (c.ip ? c.ip : 'Disconnected')}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className={styles.sidebarSection}>
          <h2 className={styles.sidebarTitle}>Add Camera</h2>
          <form className={styles.addCameraForm} onSubmit={onAddCamera}>
            <label className={styles.field}>
              <span>ID (e.g. cam-1)</span>
              <input value={newCameraId} onChange={e => setNewCameraId(e.target.value)} required />
            </label>
            <label className={styles.field}>
              <span>Label</span>
              <input value={newCameraLabel} onChange={e => setNewCameraLabel(e.target.value)} placeholder="Main Hallway" />
            </label>
            <label className={styles.field}>
              <span>Phone IP (Optional)</span>
              <input value={ip} onChange={e => setIp(e.target.value)} placeholder="192.168.1.5" />
            </label>
            {ip && (
              <>
                <label className={styles.field}>
                  <span>Port</span>
                  <input value={port} onChange={e => setPort(e.target.value)} />
                </label>
                <label className={styles.field}>
                  <span>Stream path</span>
                  <input value={path} onChange={e => setPath(e.target.value)} />
                </label>
              </>
            )}
            <button type="submit" className={styles.primary} disabled={busy || !backendOnline}>
              Add & Connect
            </button>
            <button type="button" className={styles.secondary} onClick={onAddWebcam} disabled={busy || !backendOnline}>
              Add Webcam
            </button>
          </form>
          {actionError && <p className={styles.error}>{actionError}</p>}
        </div>
      </aside>

      {/* Main Content Area */}
      <main className={styles.mainContent}>
        <header className={styles.hero} style={{ minHeight: 'auto', paddingTop: '1rem' }}>
          <div className={styles.brandBlock}>
            <p className={styles.eyebrow}>School safety monitor</p>
            <h1 className={styles.brand}>SafeSpace</h1>
            <p className={styles.tagline}>
              Monitor multiple camera feeds, view live confidence scores, and review incident evidence.
            </p>
          </div>

          <div className={styles.statusRow} aria-live="polite">
            <div className={`${styles.statusPill} ${backendOnline ? styles.ok : styles.bad}`}>
              <span className={styles.dot} />
              Backend: {backendOnline ? "Online" : "Offline"}
            </div>
            {selectedCamStatus && (
              <div className={`${styles.statusPill} ${cameraConnected ? styles.ok : styles.bad}`}>
                <span className={styles.dot} />
                {selectedCamStatus.label || selectedCamStatus.camera_id}: {cameraLabelMsg}
              </div>
            )}
          </div>
        </header>

        {selectedCamStatus ? (
          <section className={styles.monitor} aria-labelledby="monitor-heading">
            <div className={styles.sectionHead} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h2 id="monitor-heading">{selectedCamStatus.label || selectedCamStatus.camera_id}</h2>
                <p>Live view and smoothed violence confidence.</p>
              </div>
              <div className={styles.actions}>
                <button 
                  type="button" 
                  className={styles.secondary} 
                  onClick={() => onDisconnect(selectedCamStatus.camera_id)} 
                  disabled={!selectedCamStatus.camera_active || busy}
                >
                  Disconnect
                </button>
                <button 
                  type="button" 
                  className={styles.secondary} 
                  onClick={() => onRemoveCamera(selectedCamStatus.camera_id)}
                  disabled={busy}
                >
                  Remove
                </button>
              </div>
            </div>
            <div className={styles.monitorGrid}>
              <div className={styles.previewFrame}>
                {cameraConnected ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    key={`${selectedCameraId}-${feedKey}`}
                    src={feedUrl(selectedCamStatus.camera_id)}
                    alt={`Live feed from ${selectedCamStatus.label || selectedCamStatus.camera_id}`}
                  />
                ) : (
                  <div className={styles.previewEmpty}>
                    No live feed. Connect the camera to start monitoring.
                  </div>
                )}
              </div>
              <div className={styles.scores}>
                <div className={styles.scoreBlock}>
                  <span className={styles.scoreLabel}>Raw spike</span>
                  <strong className={styles.scoreValue}>
                    {raw != null ? `${raw.toFixed(1)}%` : "—"}
                  </strong>
                </div>
                <div className={styles.scoreBlock}>
                  <span className={styles.scoreLabel}>Smoothed average</span>
                  <strong
                    className={`${styles.scoreValue} ${
                      scoreHot ? styles.scoreAlert : ""
                    }`}
                  >
                    {smoothed != null ? `${smoothed.toFixed(1)}%` : "—"}
                  </strong>
                </div>
                <div className={styles.meter}>
                  <div className={styles.meterTrack}>
                    <div
                      className={styles.meterFill}
                      style={{
                        width: `${Math.min(100, Math.max(0, smoothed ?? 0))}%`,
                      }}
                    />
                    <div
                      className={styles.meterThreshold}
                      style={{ left: `${threshold}%` }}
                      title={`Alert threshold ${threshold}%`}
                    />
                  </div>
                  <p className={styles.meterCaption}>
                    Alert threshold {threshold}% · Buffer{" "}
                    {selectedCamStatus?.buffer_fill ?? "—"} · Frames{" "}
                    {selectedCamStatus?.total_frames ?? 0}
                  </p>
                  {selectedCamStatus?.scene?.note && (
                    <p
                      className={
                        selectedCamStatus.scene.suitable_for_model
                          ? styles.hint
                          : styles.error
                      }
                    >
                      Framing: {selectedCamStatus.scene.method} · subject{" "}
                      {((selectedCamStatus.scene.subject_area_frac ?? 0) * 100).toFixed(1)}%
                      {" — "}
                      {selectedCamStatus.scene.note}
                    </p>
                  )}
                </div>
              </div>
            </div>
          </section>
        ) : (
          <section className={styles.monitor}>
            <div className={styles.previewEmpty} style={{ border: '1px dashed var(--line)' }}>
              Select or add a camera to view its live monitor.
            </div>
          </section>
        )}

        <section className={styles.feedSection} aria-labelledby="review-heading">
          <div className={styles.sectionHead}>
            <h2 id="review-heading">Review a recording</h2>
            <p>
              Upload a clip (or re-analyze evidence) to see if the model would alert —
              useful for checking false positives.
            </p>
          </div>
          <div className={styles.reviewRow}>
            <label className={styles.fileBtn}>
              <input
                type="file"
                accept="video/mp4,video/*,.mp4,.mov,.avi,.mkv,.webm"
                disabled={reviewBusy || !backendOnline}
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  void onReviewFile(f);
                  e.target.value = "";
                }}
              />
              {reviewBusy ? "Analyzing…" : "Upload video to analyze"}
            </label>
          </div>
          {reviewError && <p className={styles.error}>{reviewError}</p>}
          {labelMsg && <p className={styles.hint}>{labelMsg}</p>}
          {reviewResult && (
            <div className={styles.reviewResult}>
              <p
                className={
                  reviewResult.would_alert ? styles.verdictHot : styles.verdictOk
                }
              >
                {reviewResult.would_alert
                  ? "Would ALERT (model thinks this is violence)"
                  : "Would stay SAFE (model does not alert)"}
              </p>
              <p className={styles.itemMeta}>
                Peak smoothed {reviewResult.peak_smoothed_pct}% · Peak raw{" "}
                {reviewResult.peak_raw_pct}% · Threshold{" "}
                {reviewResult.alert_threshold_pct}% ·{" "}
                {reviewResult.duration_sec}s analyzed
              </p>
              <p className={styles.hint}>{reviewResult.explanation}</p>
            </div>
          )}
        </section>

        <section className={styles.feedSection} aria-labelledby="feed-heading">
          <div className={styles.sectionHead}>
            <h2 id="feed-heading">Alerts & logs</h2>
            <p>Violence incidents with evidence, plus system connection events.</p>
          </div>
          <div className={styles.tabs}>
            <button
              type="button"
              className={tab === "alerts" ? styles.tabActive : styles.tab}
              onClick={() => setTab("alerts")}
            >
              Alerts ({alerts.length})
            </button>
            <button
              type="button"
              className={tab === "logs" ? styles.tabActive : styles.tab}
              onClick={() => setTab("logs")}
            >
              Logs ({logs.length})
            </button>
          </div>

          {tab === "alerts" ? (
            <div className={styles.list}>
              {alerts.length === 0 && (
                <p className={styles.empty}>
                  {selectedCameraId ? "No alerts for this camera yet." : "No alerts yet."}
                </p>
              )}
              {alerts.map((alert) => {
                const videoSrc = evidenceUrl(
                  alert.evidence_url || alert.evidence_file
                );
                const isOpen = selectedEvidence === alert.evidence_file;
                return (
                  <article key={alert.timestamp + alert.evidence_file} className={styles.item}>
                    <div className={styles.itemMain}>
                      <div>
                        <p className={styles.itemTitle}>
                          Incident · {alert.confidence}% confidence
                          {alert.camera_id && ` · [${alert.camera_id}]`}
                        </p>
                        <p className={styles.itemMeta}>{alert.timestamp}</p>
                      </div>
                      <div className={styles.itemActions}>
                        <button
                          type="button"
                          className={styles.secondary}
                          onClick={() =>
                            setSelectedEvidence(isOpen ? null : alert.evidence_file)
                          }
                        >
                          {isOpen ? "Hide clip" : "Play clip"}
                        </button>
                        <button
                          type="button"
                          className={styles.secondary}
                          disabled={reviewBusy || !backendOnline}
                          onClick={() => void onReanalyze(alert.evidence_file)}
                        >
                          Re-analyze
                        </button>
                        <button
                          type="button"
                          className={styles.secondary}
                          onClick={() => void onLabel(alert.evidence_file, "safe")}
                        >
                          False alarm
                        </button>
                        <button
                          type="button"
                          className={styles.secondary}
                          onClick={() => void onLabel(alert.evidence_file, "unsafe")}
                        >
                          True fight
                        </button>
                        <a
                          className={styles.linkBtn}
                          href={videoSrc}
                          download
                          target="_blank"
                          rel="noreferrer"
                        >
                          Download
                        </a>
                      </div>
                    </div>
                    {isOpen && (
                      <video
                        className={styles.evidenceVideo}
                        src={videoSrc}
                        controls
                        playsInline
                        preload="metadata"
                      />
                    )}
                  </article>
                );
              })}
            </div>
          ) : (
            <div className={styles.list}>
              {logs.length === 0 && (
                <p className={styles.empty}>No system logs yet.</p>
              )}
              {logs.map((log, i) => (
                <article
                  key={`${log.timestamp}-${i}`}
                  className={styles.logItem}
                >
                  <span
                    className={`${styles.logLevel} ${
                      log.level === "error" || log.level === "alert"
                        ? styles.logHot
                        : log.level === "warning"
                          ? styles.logWarn
                          : ""
                    }`}
                  >
                    {log.kind}/{log.level}
                  </span>
                  <span className={styles.logTime}>{log.timestamp}</span>
                  <p className={styles.logMsg}>{log.message}</p>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

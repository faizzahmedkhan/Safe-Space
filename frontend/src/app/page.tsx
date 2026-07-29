"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  AlertItem,
  AnalyzeResult,
  LogItem,
  StatusResponse,
  analyzeEvidence,
  analyzeVideoFile,
  connectCamera,
  connectWebcam,
  disconnectCamera,
  evidenceUrl,
  feedUrl,
  fetchAlerts,
  fetchLogs,
  fetchStatus,
  labelEvidence,
} from "@/lib/api";
import styles from "./page.module.css";

type Tab = "alerts" | "logs";

export default function DashboardPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [tab, setTab] = useState<Tab>("alerts");
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
  const formHydrated = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const [s, a, l] = await Promise.all([
        fetchStatus(),
        fetchAlerts(),
        fetchLogs(),
      ]);
      setStatus(s);
      setBackendOnline(true);
      setAlerts(a.recent);
      setLogs(l.recent);
      if (!formHydrated.current) {
        formHydrated.current = true;
        if (s.ip) setIp(s.ip);
        if (s.port) setPort(String(s.port));
        if (s.path) setPath(s.path);
        if (s.wifi_name) setWifiName(s.wifi_name);
      }
    } catch {
      setBackendOnline(false);
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, [refresh]);

  async function onConnect(e: FormEvent) {
    e.preventDefault();
    setActionError(null);
    setBusy(true);
    try {
      await connectCamera({
        ip: ip.trim(),
        port: Number(port) || 8080,
        path: path.trim() || "/video",
        wifi_name: wifiName.trim(),
      });
      setFeedKey((k) => k + 1);
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Connect failed");
    } finally {
      setBusy(false);
    }
  }

  async function onDisconnect() {
    setActionError(null);
    setBusy(true);
    try {
      await disconnectCamera();
      setFeedKey((k) => k + 1);
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Disconnect failed");
    } finally {
      setBusy(false);
    }
  }

  async function onWebcam() {
    setActionError(null);
    setBusy(true);
    try {
      await connectWebcam();
      setFeedKey((k) => k + 1);
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Webcam connect failed");
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

  const cameraConnected = Boolean(
    status?.camera_active && status?.camera_receiving
  );
  const cameraKind = status?.camera_kind || "none";
  const cameraLabel = !backendOnline
    ? "Unknown — start the FastAPI backend"
    : cameraConnected
      ? cameraKind === "webcam"
        ? "Laptop webcam connected"
        : status?.wifi_name
          ? `Connected on ${status.wifi_name}`
          : "Connected — frames arriving"
      : status?.camera_active
        ? "Connecting — waiting for frames"
        : "Disconnected — close Safari tabs to the phone, then Connect";

  const raw = status?.latest_raw_pct;
  const smoothed = status?.smoothed_avg_pct;
  const threshold = status?.alert_threshold ?? 60;
  const scoreHot = smoothed != null && smoothed >= threshold;

  return (
    <main className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.brandBlock}>
          <p className={styles.eyebrow}>School safety monitor</p>
          <h1 className={styles.brand}>SafeSpace</h1>
          <p className={styles.tagline}>
            Connect your phone camera, watch live detection scores, and review
            incident evidence from one operator view.
          </p>
        </div>

        <div className={styles.statusRow} aria-live="polite">
          <div
            className={`${styles.statusPill} ${
              backendOnline ? styles.ok : styles.bad
            }`}
          >
            <span className={styles.dot} />
            Laptop backend: {backendOnline ? "Connected" : "Offline"}
          </div>
          <div
            className={`${styles.statusPill} ${
              cameraConnected ? styles.ok : styles.bad
            }`}
          >
            <span className={styles.dot} />
            Phone camera: {cameraLabel}
          </div>
        </div>

        <form className={styles.connectForm} onSubmit={onConnect}>
          <label className={styles.field}>
            <span>Phone IP address</span>
            <input
              value={ip}
              onChange={(e) => setIp(e.target.value)}
              placeholder="192.168.1.5"
              required
              autoComplete="off"
            />
          </label>
          <label className={styles.field}>
            <span>Port</span>
            <input
              value={port}
              onChange={(e) => setPort(e.target.value)}
              placeholder="8080"
              inputMode="numeric"
            />
          </label>
          <label className={styles.field}>
            <span>Stream path</span>
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="/video"
            />
          </label>
          <label className={styles.fieldWide}>
            <span>Wi‑Fi name (label)</span>
            <input
              value={wifiName}
              onChange={(e) => setWifiName(e.target.value)}
              placeholder="e.g. Home-5G"
              autoComplete="off"
            />
          </label>
          <div className={styles.actions}>
            <button type="submit" className={styles.primary} disabled={busy}>
              {busy ? "Working…" : "Connect phone"}
            </button>
            <button
              type="button"
              className={styles.secondary}
              onClick={onWebcam}
              disabled={busy || !backendOnline}
            >
              Use laptop webcam
            </button>
            <button
              type="button"
              className={styles.secondary}
              onClick={onDisconnect}
              disabled={busy || !backendOnline}
            >
              Disconnect
            </button>
          </div>
        </form>
        <p className={styles.hint}>
          You can run the backend inside Cursor. On first phone Connect, macOS may open a
          Terminal window for the LAN bridge — keep that window open. Or enable Cursor under
          System Settings → Privacy & Security → Local Network.
        </p>
        {actionError && <p className={styles.error}>{actionError}</p>}
        {status?.stream_url && (
          <p className={styles.hint}>Stream: {status.stream_url}</p>
        )}
      </header>

      <section className={styles.monitor} aria-labelledby="monitor-heading">
        <div className={styles.sectionHead}>
          <h2 id="monitor-heading">Live monitor</h2>
          <p>What the model sees right now, with smoothed violence confidence.</p>
        </div>
        <div className={styles.monitorGrid}>
          <div className={styles.previewFrame}>
            {cameraConnected ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={feedKey}
                src={feedUrl()}
                alt="Live camera feed from phone"
              />
            ) : (
              <div className={styles.previewEmpty}>
                No live feed. Connect the phone camera to start monitoring.
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
                {status?.buffer_fill ?? "—"} · Frames{" "}
                {status?.total_frames ?? 0}
              </p>
              {status?.scene?.note && (
                <p
                  className={
                    status.scene.suitable_for_model
                      ? styles.hint
                      : styles.error
                  }
                >
                  Framing: {status.scene.method} · subject{" "}
                  {((status.scene.subject_area_frac ?? 0) * 100).toFixed(1)}%
                  {" — "}
                  {status.scene.note}
                </p>
              )}
            </div>
          </div>
        </div>
      </section>

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
              <p className={styles.empty}>No alerts yet.</p>
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
  );
}

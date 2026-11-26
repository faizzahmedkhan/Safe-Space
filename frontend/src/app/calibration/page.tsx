"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { History, PlayCircle, RefreshCw, RotateCcw, Upload, Video } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  CalibrationRun,
  CalibrationRunSummary,
  CalibrationSettings,
  RootResponse,
  calibrationMediaUrl,
  fetchCalibrationRun,
  fetchCalibrationRuns,
  fetchRoot,
  rerunCalibration,
  runCalibration,
} from "@/lib/api";
import AppBar from "@/components/AppBar";
import styles from "./page.module.css";

type Tab = "upload" | "youtube" | "history";

const FALLBACK_SETTINGS: CalibrationSettings = {
  target_fps: 15,
  spike_threshold: 0.9,
  alert_threshold: 0.67,
  history_len: 4,
  pad_seconds: 2.0,
  merge_gap_seconds: 1.5,
};

const formatTime = (seconds: number) => {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

const formatDate = (ts: number) => (ts ? new Date(ts * 1000).toLocaleString() : "unknown date");

const shortModel = (name: string | null | undefined, version?: string | null) => {
  const n = (name || "unknown model").replace(/^safespace_bullying_model_/, "");
  const sha = version?.split("@")[1];
  return sha ? `${n} (${sha.slice(0, 8)})` : n;
};

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "–" : v.toFixed(2));

export default function CalibrationPage() {
  const [tab, setTab] = useState<Tab>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");

  // The backend's loaded model; sliders start from ITS thresholds.
  const [modelInfo, setModelInfo] = useState<RootResponse | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [settings, setSettings] = useState<CalibrationSettings>(FALLBACK_SETTINGS);
  const settingsTouched = useRef(false);

  const [runs, setRuns] = useState<CalibrationRunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);

  const [busy, setBusy] = useState<string | null>(null); // what is running, for the loader text
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CalibrationRun | null>(null);
  const [compareId, setCompareId] = useState<string>("");
  const [compareRun, setCompareRun] = useState<CalibrationRun | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const mainVideoRef = useRef<HTMLVideoElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const [currentSec, setCurrentSec] = useState<number | null>(null);

  const modelDefaults = useCallback(
    (info: RootResponse | null): CalibrationSettings => ({
      ...FALLBACK_SETTINGS,
      ...(info?.thresholds
        ? {
            spike_threshold: info.thresholds.spike_threshold,
            alert_threshold: info.thresholds.alert_threshold,
            history_len: info.thresholds.score_history_len,
          }
        : {}),
    }),
    []
  );

  useEffect(() => {
    const ctrl = new AbortController();
    fetchRoot(ctrl.signal)
      .then((info) => {
        setModelInfo(info);
        if (!settingsTouched.current) setSettings(modelDefaults(info));
      })
      .catch((err) => {
        if (!ctrl.signal.aborted) setModelError(err instanceof Error ? err.message : "Backend unreachable");
      });
    return () => ctrl.abort();
  }, [modelDefaults]);

  const loadRuns = useCallback(async () => {
    setRunsLoading(true);
    setRunsError(null);
    try {
      setRuns(await fetchCalibrationRuns());
    } catch (err) {
      setRunsError(err instanceof Error ? err.message : "Could not load past runs");
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (!busy) return;
    const start = Date.now();
    setElapsed(0);
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [busy]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const update = (patch: Partial<CalibrationSettings>) => {
    settingsTouched.current = true;
    setSettings((s) => ({ ...s, ...patch }));
  };

  const showRun = (run: CalibrationRun) => {
    setResult(run);
    setCompareId("");
    setCompareRun(null);
    setCurrentSec(null);
    setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  /** Run something long (calibration / re-run / open) with the shared loader + cancel. */
  const withBusy = async (label: string, job: (signal: AbortSignal) => Promise<void>) => {
    setBusy(label);
    setError(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await job(ctrl.signal);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") setError("Cancelled");
      else setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setBusy(null);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (tab === "upload" && !file) return;
    if (tab === "youtube" && !youtubeUrl) return;
    setResult(null);
    await withBusy("Analyzing video frames", async (signal) => {
      const run = await runCalibration(tab === "upload" && file ? { file } : { youtubeUrl }, settings, signal);
      showRun(run);
      loadRuns();
    });
  };

  const rerun = (videoName: string, rerunOf?: string) =>
    withBusy("Re-running calibration on the saved video", async (signal) => {
      const run = await rerunCalibration(videoName, settings, rerunOf, signal);
      showRun(run);
      loadRuns();
    });

  const openRun = (runId: string) =>
    withBusy("Opening saved run", async (signal) => {
      showRun(await fetchCalibrationRun(runId, signal));
    });

  useEffect(() => {
    if (!compareId) {
      setCompareRun(null);
      return;
    }
    const ctrl = new AbortController();
    fetchCalibrationRun(compareId, ctrl.signal)
      .then(setCompareRun)
      .catch((err) => {
        if (!ctrl.signal.aborted) setError(err instanceof Error ? err.message : "Could not load comparison run");
      });
    return () => ctrl.abort();
  }, [compareId]);

  // Past runs grouped by source video, newest video first.
  const grouped = useMemo(() => {
    const m = new Map<string, CalibrationRunSummary[]>();
    for (const r of runs) {
      const k = r.video_name || "(unknown video)";
      m.set(k, [...(m.get(k) || []), r]);
    }
    return [...m.entries()];
  }, [runs]);

  const siblings = useMemo(
    () => (result ? runs.filter((r) => r.video_name === result.video_name && r.run_id !== result.run_id) : []),
    [runs, result]
  );

  // Thresholds the displayed run actually used (old runs did not record them).
  const runSpike = result?.settings?.spike_threshold ?? null;
  const runAlert = result?.settings?.alert_threshold ?? null;

  const chartData = useMemo(() => {
    if (!result) return [];
    if (!compareRun) return result.per_sample_log;
    const key = (t: number) => t.toFixed(2);
    const other = new Map(compareRun.per_sample_log.map((s) => [key(s.time_sec), s.raw_score]));
    return result.per_sample_log.map((s) => ({ ...s, compare_raw: other.get(key(s.time_sec)) }));
  }, [result, compareRun]);

  const compareMatched = useMemo(
    () => (compareRun ? chartData.filter((d) => "compare_raw" in d && d.compare_raw !== undefined).length : 0),
    [chartData, compareRun]
  );

  const activeSample =
    currentSec !== null && result?.per_sample_log && result.per_sample_log.length > 0
      ? result.per_sample_log.reduce((prev, curr) =>
          Math.abs(curr.time_sec - currentSec) < Math.abs(prev.time_sec - currentSec) ? curr : prev
        )
      : null;

  const defaults = modelDefaults(modelInfo);
  const differsFromModel =
    !!modelInfo?.thresholds &&
    (settings.spike_threshold !== defaults.spike_threshold ||
      settings.alert_threshold !== defaults.alert_threshold ||
      settings.history_len !== defaults.history_len);

  return (
    <>
      <AppBar active="calibration" />
      <div className={`${styles.layoutContainer} gutter`}>
        <main className={`${styles.mainContent} wrap`}>
          <div className={styles.header}>
            <h1 className={styles.title}>Model calibration</h1>
            <p className={styles.lede}>
              Run a recording through the model to tune the spike and sustained-alert thresholds. Every run is saved,
              so past results can be reopened, re-run with new settings and compared.
            </p>
          </div>

          <form className={styles.formCard} onSubmit={handleSubmit}>
            <div className={styles.modelBanner} role="status">
              {modelInfo ? (
                <>
                  <span>
                    Runs use the loaded model <strong>{shortModel(modelInfo.model_name, modelInfo.model_version)}</strong>
                    {" · "}preprocessing <strong>{modelInfo.preprocessing ?? "v1"}</strong>
                    {" · "}its thresholds: spike <strong>{pct(defaults.spike_threshold)}</strong>, alert{" "}
                    <strong>{pct(defaults.alert_threshold)}</strong>
                    {modelInfo.thresholds_from_sidecar ? " (from its sidecar)" : " (defaults)"}
                  </span>
                  {differsFromModel && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => {
                        settingsTouched.current = false;
                        setSettings(defaults);
                      }}
                    >
                      <RotateCcw size={14} /> Reset to model thresholds
                    </button>
                  )}
                </>
              ) : (
                <span>{modelError ? `Backend unreachable: ${modelError}` : "Checking which model is loaded…"}</span>
              )}
            </div>

            <div className={styles.tabs}>
              <button
                type="button"
                className={`${styles.tabBtn} ${tab === "upload" ? styles.tabBtnActive : ""}`}
                onClick={() => setTab("upload")}
              >
                <Upload size={16} />
                Upload Local Video
              </button>
              <button
                type="button"
                className={`${styles.tabBtn} ${tab === "youtube" ? styles.tabBtnActive : ""}`}
                onClick={() => setTab("youtube")}
              >
                <Video size={16} />
                YouTube URL
              </button>
              <button
                type="button"
                className={`${styles.tabBtn} ${tab === "history" ? styles.tabBtnActive : ""}`}
                onClick={() => {
                  setTab("history");
                  loadRuns();
                }}
              >
                <History size={16} />
                Past runs ({runs.length})
              </button>
            </div>

            {tab === "upload" && (
              <label className={styles.field}>
                <span>Select Video File</span>
                <input
                  type="file"
                  className={styles.input}
                  accept="video/*"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  required
                />
              </label>
            )}
            {tab === "youtube" && (
              <label className={styles.field}>
                <span>YouTube Video URL</span>
                <input
                  type="url"
                  className={styles.input}
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  required
                />
              </label>
            )}
            {tab === "history" && (
              <div className={styles.historyList}>
                <div className={styles.historyToolbar}>
                  <span>
                    Newest first, grouped by video. The sliders below apply when you re-run a video with the loaded
                    model.
                  </span>
                  <button type="button" className="btn btn-sm" onClick={loadRuns} disabled={runsLoading}>
                    <RefreshCw size={14} /> {runsLoading ? "Loading…" : "Refresh"}
                  </button>
                </div>
                {runsError && <div className={styles.error}>{runsError}</div>}
                {!runsError && !runsLoading && grouped.length === 0 && (
                  <p className={styles.lede}>No calibration runs saved yet.</p>
                )}
                {grouped.map(([videoName, videoRuns]) => (
                  <div key={videoName} className={styles.historyGroup}>
                    <div className={styles.historyGroupHeader}>
                      <span className={styles.historyVideo}>{videoName}</span>
                      {videoRuns[0].video_available ? (
                        <button
                          type="button"
                          className="btn btn-sm"
                          disabled={!!busy}
                          onClick={() => rerun(videoName, videoRuns[0].run_id)}
                        >
                          <RefreshCw size={14} /> Re-run with current settings
                        </button>
                      ) : (
                        <span className={styles.muted}>source video no longer on the server</span>
                      )}
                    </div>
                    <table className={styles.runTable}>
                      <thead>
                        <tr>
                          <th>Run</th>
                          <th>Model</th>
                          <th>Spike / alert</th>
                          <th>Events</th>
                          <th>Peak</th>
                          <th />
                        </tr>
                      </thead>
                      <tbody>
                        {videoRuns.map((r) => (
                          <tr key={r.run_id} className={result?.run_id === r.run_id ? styles.runRowActive : undefined}>
                            <td>{formatDate(r.created_ts)}</td>
                            <td>
                              {shortModel(r.model_name, r.model_version)}
                              {r.preprocessing ? ` · ${r.preprocessing}` : ""}
                            </td>
                            <td>
                              {r.spike_threshold === null ? (
                                <span className={styles.muted}>not recorded</span>
                              ) : (
                                `${pct(r.spike_threshold)} / ${pct(r.alert_threshold)}`
                              )}
                            </td>
                            <td>{r.n_events}</td>
                            <td>{r.peak_raw === null ? "–" : `${(r.peak_raw * 100).toFixed(0)}%`}</td>
                            <td>
                              <button type="button" className="btn btn-sm" disabled={!!busy} onClick={() => openRun(r.run_id)}>
                                Open
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ))}
              </div>
            )}

            <div className={styles.sliders}>
              <div className={styles.sliderRow}>
                <label>
                  <span>Spike Threshold</span>
                  <span>{settings.spike_threshold.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="1.0"
                  step="0.01"
                  value={settings.spike_threshold}
                  onChange={(e) => update({ spike_threshold: parseFloat(e.target.value) })}
                />
              </div>
              <div className={styles.sliderRow}>
                <label>
                  <span>Sustained Alert Threshold</span>
                  <span>{settings.alert_threshold.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0.4"
                  max="1.0"
                  step="0.01"
                  value={settings.alert_threshold}
                  onChange={(e) => update({ alert_threshold: parseFloat(e.target.value) })}
                />
              </div>
              <div className={styles.sliderRow}>
                <label>
                  <span>Smoothing History Length (clips)</span>
                  <span>{settings.history_len}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={settings.history_len}
                  onChange={(e) => update({ history_len: parseInt(e.target.value, 10) })}
                />
              </div>
              <div className={styles.sliderRow}>
                <label>
                  <span>Padding Seconds</span>
                  <span>{settings.pad_seconds.toFixed(1)}s</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="5"
                  step="0.5"
                  value={settings.pad_seconds}
                  onChange={(e) => update({ pad_seconds: parseFloat(e.target.value) })}
                />
              </div>
            </div>

            {error && (
              <div className={styles.error} role="alert">
                {error}
              </div>
            )}

            {tab !== "history" && (
              <button
                type="submit"
                className={`btn btn-primary ${styles.submitBtn}`}
                disabled={!!busy || (tab === "upload" && !file) || (tab === "youtube" && !youtubeUrl)}
              >
                {busy ? "Processing Video..." : "Run Calibration"}
              </button>
            )}
          </form>

          {busy && (
            <div className={styles.loader}>
              <div className={styles.spinner} />
              <p>
                {busy}... elapsed {formatTime(elapsed)}. Longer videos can take a few minutes.
              </p>
              <button type="button" className="btn btn-sm" onClick={() => abortRef.current?.abort()}>
                Cancel
              </button>
            </div>
          )}

          {result && (
            <div className={styles.resultsArea} ref={resultsRef}>
              {/* What produced this run */}
              <div className={styles.runMeta}>
                <div className={styles.runMetaMain}>
                  <h3>{result.video_name}</h3>
                  <p>
                    {formatDate(result.created_ts)} · model{" "}
                    <strong>{shortModel(result.model.name, result.model.version)}</strong>
                    {result.model.preprocessing ? ` · preprocessing ${result.model.preprocessing}` : ""}
                    {result.settings ? (
                      <>
                        {" · "}spike <strong>{pct(result.settings.spike_threshold)}</strong>, alert{" "}
                        <strong>{pct(result.settings.alert_threshold)}</strong>, history {result.settings.history_len}
                      </>
                    ) : (
                      " · older run: thresholds were not recorded"
                    )}
                    {" · "}
                    {result.events.length} event(s), peak {(result.summary.peak_raw * 100).toFixed(0)}%
                  </p>
                </div>
                <div className={styles.runMetaActions}>
                  {result.video_available && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={!!busy}
                      onClick={() => rerun(result.video_name, result.run_id)}
                    >
                      <RefreshCw size={14} /> Re-run with current settings
                    </button>
                  )}
                  {siblings.length > 0 && (
                    <label className={styles.compareSelect}>
                      <span>Compare with</span>
                      <select value={compareId} onChange={(e) => setCompareId(e.target.value)}>
                        <option value="">no comparison</option>
                        {siblings.map((r) => (
                          <option key={r.run_id} value={r.run_id}>
                            {formatDate(r.created_ts)} · {shortModel(r.model_name, r.model_version)}
                            {r.spike_threshold !== null ? ` · spike ${pct(r.spike_threshold)}` : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                </div>
              </div>

              {/* Synchronized Uncut Video Player */}
              <div className={styles.uncutCard}>
                <div className={styles.uncutHeader}>
                  <h3>Full video with synchronized scores</h3>
                  <span className={styles.timeBadge}>{formatTime(currentSec || 0)}</span>
                </div>
                <div className={styles.uncutBody}>
                  <div className={styles.mainVideoWrapper}>
                    {result.video_available && result.video_url ? (
                      <video
                        ref={mainVideoRef}
                        src={calibrationMediaUrl(result.video_url)}
                        controls
                        onTimeUpdate={(e) => setCurrentSec(e.currentTarget.currentTime)}
                      />
                    ) : (
                      <p className={styles.unavailable}>
                        The source video is no longer on the server, so it cannot be played. The scores and any
                        remaining clips are still shown.
                      </p>
                    )}
                  </div>
                  <div className={styles.metricsPanel}>
                    <div className={styles.metricRow}>
                      <div className={styles.metricHeader}>
                        <span>Raw Confidence Score</span>
                        <span className={styles.metricValue}>
                          {activeSample ? `${(activeSample.raw_score * 100).toFixed(1)}%` : "--"}
                        </span>
                      </div>
                      <div className={styles.progressBarBg}>
                        <div
                          className={styles.progressBarFillRaw}
                          style={{ width: activeSample ? `${Math.min(100, activeSample.raw_score * 100)}%` : "0%" }}
                        />
                      </div>
                    </div>

                    <div className={styles.metricRow}>
                      <div className={styles.metricHeader}>
                        <span>Smoothed Average</span>
                        <span className={styles.metricValue}>
                          {activeSample ? `${(activeSample.smoothed_avg * 100).toFixed(1)}%` : "--"}
                        </span>
                      </div>
                      <div className={styles.progressBarBg}>
                        <div
                          className={styles.progressBarFillAvg}
                          style={{ width: activeSample ? `${Math.min(100, activeSample.smoothed_avg * 100)}%` : "0%" }}
                        />
                      </div>
                    </div>

                    <div className={styles.statusBox}>
                      <span>Detection Status</span>
                      <div
                        className={`${styles.statusIndicator} ${
                          activeSample?.triggered
                            ? activeSample.trigger_reason === "spike"
                              ? styles.spikeActive
                              : styles.sustainedActive
                            : styles.normalActive
                        }`}
                      >
                        {activeSample?.triggered
                          ? activeSample.trigger_reason === "spike"
                            ? "Spike alert triggered"
                            : "Sustained alert triggered"
                          : "Normal, below thresholds"}
                      </div>
                    </div>
                    <p className={styles.seekHint}>
                      Click the timeline chart below, or an incident clip, to jump the video to that moment.
                    </p>
                  </div>
                </div>
              </div>

              {/* Confidence Score Timeline */}
              <div className={styles.chartCard}>
                <h3>Confidence timeline</h3>
                {compareRun && compareMatched === 0 && (
                  <p className={styles.muted}>
                    The comparison run&apos;s timestamps do not line up with this run (different frame rate), so it is
                    not drawn.
                  </p>
                )}
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={chartData}
                    margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
                    onClick={(e) => {
                      if (e && e.activeLabel !== undefined && mainVideoRef.current) {
                        mainVideoRef.current.currentTime = Number(e.activeLabel);
                        mainVideoRef.current.play();
                      }
                    }}
                    style={{ cursor: "pointer" }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e3e9f3" />
                    <XAxis
                      dataKey="time_sec"
                      stroke="#5a6a82"
                      tickFormatter={(val) => formatTime(Number(val))}
                      label={{ value: "Time (MM:SS)", position: "bottom", fill: "#5a6a82" }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      stroke="#5a6a82"
                      label={{ value: "Confidence", angle: -90, position: "insideLeft", fill: "#5a6a82" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#ffffff",
                        border: "none",
                        color: "#0e1b2e",
                        borderRadius: 8,
                        boxShadow: "0 6px 20px rgba(10,27,51,0.16)",
                      }}
                      labelFormatter={(val) => `Time: ${formatTime(Number(val))} (${Number(val).toFixed(1)}s)`}
                    />
                    <Legend verticalAlign="top" height={36} />
                    {runSpike !== null && (
                      <ReferenceLine y={runSpike} label="Spike Threshold" stroke="#d62f3b" strokeDasharray="3 3" />
                    )}
                    {runAlert !== null && (
                      <ReferenceLine y={runAlert} label="Alert Threshold" stroke="#b86e00" strokeDasharray="3 3" />
                    )}
                    {currentSec !== null && (
                      <ReferenceLine
                        x={currentSec}
                        stroke="#0a1b33"
                        strokeWidth={2}
                        label={{ value: "Playback", fill: "#0a1b33", position: "top" }}
                      />
                    )}
                    <Line type="monotone" dataKey="raw_score" name="Raw Score" stroke="#7d93b8" strokeWidth={1.5} dot={false} />
                    <Line
                      type="monotone"
                      dataKey="smoothed_avg"
                      name="Smoothed Average"
                      stroke="#1b5cf0"
                      strokeWidth={2}
                      dot={false}
                    />
                    {compareRun && compareMatched > 0 && (
                      <Line
                        type="monotone"
                        dataKey="compare_raw"
                        name={`Raw Score · ${shortModel(compareRun.model.name, compareRun.model.version)}, ${formatDate(
                          compareRun.created_ts
                        )}`}
                        stroke="#b86e00"
                        strokeWidth={1.5}
                        dot={false}
                        connectNulls
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Isolated Clips Grid */}
              <div>
                <h3 className={styles.sectionTitle} style={{ marginBottom: 16 }}>
                  Isolated incident clips ({result.events.length})
                </h3>
                {result.events.length === 0 ? (
                  <p className={styles.lede}>No segments crossed the threshold.</p>
                ) : (
                  <div className={styles.clipsGrid}>
                    {result.events.map((ev, i) => (
                      <div key={i} className={styles.clipCard}>
                        <div className={styles.videoContainer}>
                          {ev.clip_available === false ? (
                            <p className={styles.unavailable}>Clip removed by storage cleanup.</p>
                          ) : (
                            <video src={calibrationMediaUrl(ev.url)} controls preload="metadata" />
                          )}
                        </div>
                        <div className={styles.clipInfo}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span className={styles.clipTime}>
                              {formatTime(ev.start_sec)} - {formatTime(ev.end_sec)} ({ev.start_sec.toFixed(1)}s -{" "}
                              {ev.end_sec.toFixed(1)}s)
                            </span>
                            <span className={`${styles.triggerBadge} ${ev.trigger === "sustained" ? styles.sustained : ""}`}>
                              {ev.trigger}
                            </span>
                          </div>
                          <div className={styles.clipMeta}>
                            <span>Peak Score:</span>
                            <strong>{(ev.peak_score * 100).toFixed(1)}%</strong>
                          </div>
                          <button
                            type="button"
                            className={`btn btn-sm ${styles.jumpBtn}`}
                            disabled={!result.video_available}
                            onClick={() => {
                              if (mainVideoRef.current) {
                                mainVideoRef.current.currentTime = ev.start_sec;
                                mainVideoRef.current.play();
                                mainVideoRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
                              }
                            }}
                          >
                            <PlayCircle size={14} />
                            Seek video to {formatTime(ev.start_sec)}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </>
  );
}

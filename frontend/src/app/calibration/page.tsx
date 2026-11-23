"use client";

import { FormEvent, useEffect, useState, useRef } from "react";
import { Upload, Video, PlayCircle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { apiFetch, getApiUrl, withKey } from "@/lib/api";
import AppBar from "@/components/AppBar";
import styles from "./page.module.css";

const CALIBRATION_TIMEOUT_MS = 30 * 60 * 1000;

type CalibrationEvent = {
  start_sec: number;
  end_sec: number;
  peak_score: number;
  trigger: string;
  url: string;
};

type CalibrationResult = {
  ok: boolean;
  video_path: string;
  native_fps: number;
  events: CalibrationEvent[];
  per_sample_log: Array<{
    time_sec: number;
    raw_score: number;
    smoothed_avg: number;
    triggered?: boolean;
    trigger_reason?: string;
  }>;
};

const formatTime = (seconds: number) => {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

export default function CalibrationPage() {
  const [tab, setTab] = useState<"upload" | "youtube">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  
  // Settings
  const [targetFps, setTargetFps] = useState(15);
  const [spikeThreshold, setSpikeThreshold] = useState(0.9);
  const [alertThreshold, setAlertThreshold] = useState(0.75);
  const [historyLen, setHistoryLen] = useState(4);
  const [padSeconds, setPadSeconds] = useState(2.0);
  const [mergeGapSeconds, setMergeGapSeconds] = useState(1.5);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CalibrationResult | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!loading) return;
    const start = Date.now();
    setElapsed(0);
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [loading]);

  useEffect(() => () => abortRef.current?.abort(), []);

  // Playback sync
  const mainVideoRef = useRef<HTMLVideoElement | null>(null);
  const [currentSec, setCurrentSec] = useState<number | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (tab === "upload" && !file) return;
    if (tab === "youtube" && !youtubeUrl) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentSec(null);

    try {
      const formData = new FormData();
      if (tab === "upload" && file) {
        formData.append("video_file", file);
      } else if (tab === "youtube") {
        formData.append("youtube_url", youtubeUrl);
      }
      
      formData.append("target_fps", targetFps.toString());
      formData.append("spike_threshold", spikeThreshold.toString());
      formData.append("alert_threshold", alertThreshold.toString());
      formData.append("history_len", historyLen.toString());
      formData.append("pad_seconds", padSeconds.toString());
      formData.append("merge_gap_seconds", mergeGapSeconds.toString());

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      const res = await apiFetch("/calibration/run", {
        method: "POST",
        body: formData,
        signal: ctrl.signal,
        timeoutMs: CALIBRATION_TIMEOUT_MS,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Calibration cancelled");
      } else {
        setError(err instanceof Error ? err.message : "An error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  const getAPIUrl = (path: string) => withKey(`${getApiUrl()}${path}`);

  const activeSample =
    currentSec !== null && result?.per_sample_log && result.per_sample_log.length > 0
      ? result.per_sample_log.reduce((prev, curr) =>
          Math.abs(curr.time_sec - currentSec) < Math.abs(prev.time_sec - currentSec) ? curr : prev
        )
      : null;

  return (
    <>
      <AppBar active="calibration" />
      <div className={`${styles.layoutContainer} gutter`}>
      <main className={`${styles.mainContent} wrap`}>
        <div className={styles.header}>
          <h1 className={styles.title}>Model calibration</h1>
          <p className={styles.lede}>
            Run a recording through the model to tune the spike and sustained-alert thresholds.
          </p>
        </div>

        <form className={styles.formCard} onSubmit={handleSubmit}>
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
          </div>

          {tab === "upload" ? (
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
          ) : (
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

          <div className={styles.sliders}>
            <div className={styles.sliderRow}>
              <label>
                <span>Spike Threshold</span>
                <span>{spikeThreshold.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.01"
                value={spikeThreshold}
                onChange={(e) => setSpikeThreshold(parseFloat(e.target.value))}
              />
            </div>
            <div className={styles.sliderRow}>
              <label>
                <span>Sustained Alert Threshold</span>
                <span>{alertThreshold.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.4"
                max="1.0"
                step="0.01"
                value={alertThreshold}
                onChange={(e) => setAlertThreshold(parseFloat(e.target.value))}
              />
            </div>
            <div className={styles.sliderRow}>
              <label>
                <span>Smoothing History Length (clips)</span>
                <span>{historyLen}</span>
              </label>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={historyLen}
                onChange={(e) => setHistoryLen(parseInt(e.target.value))}
              />
            </div>
            <div className={styles.sliderRow}>
              <label>
                <span>Padding Seconds</span>
                <span>{padSeconds.toFixed(1)}s</span>
              </label>
              <input
                type="range"
                min="0"
                max="5"
                step="0.5"
                value={padSeconds}
                onChange={(e) => setPadSeconds(parseFloat(e.target.value))}
              />
            </div>
          </div>

          {error && <div className={styles.error} role="alert">{error}</div>}

          <button type="submit" className={`btn btn-primary ${styles.submitBtn}`} disabled={loading || (tab === "upload" && !file) || (tab === "youtube" && !youtubeUrl)}>
            {loading ? "Processing Video..." : "Run Calibration"}
          </button>
        </form>

        {loading && (
          <div className={styles.loader}>
            <div className={styles.spinner} />
            <p>
              Analyzing video frames... elapsed {formatTime(elapsed)}. This may take a few minutes for longer videos.
            </p>
            <button type="button" className="btn btn-sm" onClick={() => abortRef.current?.abort()}>
              Cancel
            </button>
          </div>
        )}

        {result && (
          <div className={styles.resultsArea}>
            {/* Synchronized Uncut Video Player */}
            <div className={styles.uncutCard}>
              <div className={styles.uncutHeader}>
                <h3>Full video with synchronized scores</h3>
                <span className={styles.timeBadge}>
                  {formatTime(currentSec || 0)}
                </span>
              </div>
              <div className={styles.uncutBody}>
                <div className={styles.mainVideoWrapper}>
                  <video
                    ref={mainVideoRef}
                    src={getAPIUrl(`/calibration/clip/${encodeURIComponent(result.video_path.split(/[/\\]/).pop() || '')}`)}
                    controls
                    onTimeUpdate={(e) => setCurrentSec(e.currentTarget.currentTime)}
                  />
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
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={result.per_sample_log}
                  margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
                  onClick={(e) => {
                    if (e && e.activeLabel !== undefined && mainVideoRef.current) {
                      const sec = Number(e.activeLabel);
                      mainVideoRef.current.currentTime = sec;
                      mainVideoRef.current.play();
                    }
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e3e9f3" />
                  <XAxis 
                    dataKey="time_sec" 
                    stroke="#5a6a82" 
                    tickFormatter={(val) => formatTime(Number(val))}
                    label={{ value: 'Time (MM:SS)', position: 'bottom', fill: '#5a6a82' }} 
                  />
                  <YAxis 
                    domain={[0, 1]} 
                    stroke="#5a6a82" 
                    label={{ value: 'Confidence', angle: -90, position: 'insideLeft', fill: '#5a6a82' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', border: 'none', color: '#0e1b2e', borderRadius: 8, boxShadow: '0 6px 20px rgba(10,27,51,0.16)' }}
                    labelFormatter={(val) => `Time: ${formatTime(Number(val))} (${Number(val).toFixed(1)}s)`}
                  />
                  <Legend verticalAlign="top" height={36}/>
                  <ReferenceLine y={spikeThreshold} label="Spike Threshold" stroke="#d62f3b" strokeDasharray="3 3" />
                  <ReferenceLine y={alertThreshold} label="Alert Threshold" stroke="#b86e00" strokeDasharray="3 3" />
                  {currentSec !== null && (
                    <ReferenceLine x={currentSec} stroke="#0a1b33" strokeWidth={2} label={{ value: 'Playback', fill: '#0a1b33', position: 'top' }} />
                  )}
                  <Line type="monotone" dataKey="raw_score" name="Raw Score" stroke="#7d93b8" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="smoothed_avg" name="Smoothed Average" stroke="#1b5cf0" strokeWidth={2} dot={false} />
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
                        <video src={getAPIUrl(ev.url)} controls preload="metadata" />
                      </div>
                      <div className={styles.clipInfo}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span className={styles.clipTime}>{formatTime(ev.start_sec)} - {formatTime(ev.end_sec)} ({ev.start_sec.toFixed(1)}s - {ev.end_sec.toFixed(1)}s)</span>
                          <span className={`${styles.triggerBadge} ${ev.trigger === 'sustained' ? styles.sustained : ''}`}>
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
                          onClick={() => {
                            if (mainVideoRef.current) {
                              mainVideoRef.current.currentTime = ev.start_sec;
                              mainVideoRef.current.play();
                              mainVideoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
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

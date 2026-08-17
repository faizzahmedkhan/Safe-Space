"use client";

import { FormEvent, useState, useRef, useEffect, useMemo } from "react";
import Link from "next/link";
import { ArrowLeft, Upload, Video, PlayCircle, History } from "lucide-react";
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
import styles from "./page.module.css";

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
  per_sample_log: any[];
  model_type: string;
};

const formatTime = (seconds: number) => {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

type HistoryItem = { id: string; video_path: string; events_count: number; time: number; model_type: string; };

export default function CalibrationPage() {
  const [tab, setTab] = useState<"upload" | "youtube" | "history">("upload");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [modelType, setModelType] = useState<"original" | "smart_crop" | "compare">("original");
  const [clipFilter, setClipFilter] = useState<"all" | "original" | "smart_crop">("all");
  
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
  const [compareResult, setCompareResult] = useState<CalibrationResult | null>(null);

  // Playback sync
  const mainVideoRef = useRef<HTMLVideoElement | null>(null);
  const [currentSec, setCurrentSec] = useState<number | null>(null);

  const groupedHistory = useMemo(() => {
    return history.reduce((acc, run) => {
      if (!acc[run.video_path]) acc[run.video_path] = [];
      acc[run.video_path].push(run);
      return acc;
    }, {} as Record<string, HistoryItem[]>);
  }, [history]);

  useEffect(() => {
    if (tab === "history") {
      fetch("http://127.0.0.1:8000/calibration/runs")
        .then((res) => res.json())
        .then((data) => setHistory(data.runs || []))
        .catch(console.error);
    }
  }, [tab]);

  const loadRun = async (runId: string, compareRunId?: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setCompareResult(null);
    setCurrentSec(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/calibration/runs/${runId}`);
      if (!res.ok) throw new Error("Failed to load run");
      setResult(await res.json());

      if (compareRunId) {
        const res2 = await fetch(`http://127.0.0.1:8000/calibration/runs/${compareRunId}`);
        if (!res2.ok) throw new Error("Failed to load compare run");
        setCompareResult(await res2.json());
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error loading run");
    } finally {
      setLoading(false);
    }
  };

  const handleRunExisting = async (videoPath: string, targetModel: string, currentRunId: string) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("video_path", videoPath);
      formData.append("model_type", targetModel);
      formData.append("target_fps", targetFps.toString());
      formData.append("spike_threshold", spikeThreshold.toString());
      formData.append("alert_threshold", alertThreshold.toString());
      formData.append("history_len", historyLen.toString());
      formData.append("pad_seconds", padSeconds.toString());
      formData.append("merge_gap_seconds", mergeGapSeconds.toString());
      
      const res = await fetch("http://127.0.0.1:8000/calibration/run_existing", { method: "POST", body: formData });
      if (!res.ok) throw new Error("Failed to re-run calibration");
      const newData = await res.json();
      
      const resOrig = await fetch(`http://127.0.0.1:8000/calibration/runs/${currentRunId}`);
      const origData = await resOrig.json();
      
      if (origData.model_type === "original" || !origData.model_type) {
        setResult(origData);
        setCompareResult(newData);
      } else {
        setResult(newData);
        setCompareResult(origData);
      }
      
      const histRes = await fetch("http://127.0.0.1:8000/calibration/runs");
      const histData = await histRes.json();
      setHistory(histData.runs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error running existing video");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (tab === "upload" && !file) return;
    if (tab === "youtube" && !youtubeUrl) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setCompareResult(null);
    setCurrentSec(null);

    try {
      const formData = new FormData();
      if (tab === "upload" && file) formData.append("video_file", file);
      else if (tab === "youtube") formData.append("youtube_url", youtubeUrl);
      
      formData.append("model_type", modelType === "compare" ? "original" : modelType);
      formData.append("target_fps", targetFps.toString());
      formData.append("spike_threshold", spikeThreshold.toString());
      formData.append("alert_threshold", alertThreshold.toString());
      formData.append("history_len", historyLen.toString());
      formData.append("pad_seconds", padSeconds.toString());
      formData.append("merge_gap_seconds", mergeGapSeconds.toString());

      const res = await fetch("http://127.0.0.1:8000/calibration/run", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Calibration failed");
      }

      const data = await res.json();
      setResult(data);

      if (modelType === "compare") {
        const secondForm = new FormData();
        secondForm.append("video_path", data.video_path);
        secondForm.append("model_type", "smart_crop");
        secondForm.append("target_fps", targetFps.toString());
        secondForm.append("spike_threshold", spikeThreshold.toString());
        secondForm.append("alert_threshold", alertThreshold.toString());
        secondForm.append("history_len", historyLen.toString());
        secondForm.append("pad_seconds", padSeconds.toString());
        secondForm.append("merge_gap_seconds", mergeGapSeconds.toString());
        
        const res2 = await fetch("http://127.0.0.1:8000/calibration/run_existing", {
          method: "POST",
          body: secondForm
        });
        if (res2.ok) {
           setCompareResult(await res2.json());
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const getAPIUrl = (path: string) => `http://127.0.0.1:8000${path}`;

  const activeSample =
    currentSec !== null && result?.per_sample_log && result.per_sample_log.length > 0
      ? result.per_sample_log.reduce((prev, curr) =>
          Math.abs(curr.time_sec - currentSec) < Math.abs(prev.time_sec - currentSec) ? curr : prev
        )
      : null;

  const activeCompareSample =
    currentSec !== null && compareResult?.per_sample_log && compareResult.per_sample_log.length > 0
      ? compareResult.per_sample_log.reduce((prev, curr) =>
          Math.abs(curr.time_sec - currentSec) < Math.abs(prev.time_sec - currentSec) ? curr : prev
        )
      : null;

  const mergedLogs = useMemo(() => {
    if (!result) return [];
    if (!compareResult) return result.per_sample_log;
    
    const map = new Map();
    result.per_sample_log.forEach(item => {
      map.set(item.time_sec, { ...item, raw_score: item.raw_score, smoothed_avg: item.smoothed_avg });
    });
    compareResult.per_sample_log.forEach(item => {
      if (map.has(item.time_sec)) {
        const existing = map.get(item.time_sec);
        map.set(item.time_sec, { 
          ...existing, 
          raw_score_smart: item.raw_score, 
          smoothed_avg_smart: item.smoothed_avg 
        });
      } else {
        map.set(item.time_sec, { 
          time_sec: item.time_sec, 
          raw_score_smart: item.raw_score, 
          smoothed_avg_smart: item.smoothed_avg 
        });
      }
    });
    return Array.from(map.values()).sort((a, b) => a.time_sec - b.time_sec);
  }, [result, compareResult]);

  return (
    <div className={styles.layoutContainer}>
      <main className={styles.mainContent}>
        <div className={styles.header}>
          <h1 className={styles.title}>Model Calibration</h1>
          <Link href="/" className={styles.backLink}>
            <ArrowLeft size={20} /> Back to Dashboard
          </Link>
        </div>

        <form className={styles.formCard} onSubmit={handleSubmit}>
          <div className={styles.tabs}>
            <button
              type="button"
              className={`${styles.tabBtn} ${tab === "upload" ? styles.tabBtnActive : ""}`}
              onClick={() => setTab("upload")}
            >
              <Upload size={16} style={{ display: 'inline', marginRight: 6, verticalAlign: 'text-bottom' }} /> 
              Upload Local Video
            </button>
            <button
              type="button"
              className={`${styles.tabBtn} ${tab === "youtube" ? styles.tabBtnActive : ""}`}
              onClick={() => setTab("youtube")}
            >
              <Video size={16} style={{ display: 'inline', marginRight: 6, verticalAlign: 'text-bottom' }} /> 
              YouTube URL
            </button>
            <button
              type="button"
              className={`${styles.tabBtn} ${tab === "history" ? styles.tabBtnActive : ""}`}
              onClick={() => setTab("history")}
            >
              <History size={16} style={{ display: 'inline', marginRight: 6, verticalAlign: 'text-bottom' }} /> 
              Previous Runs
            </button>
          </div>

          {tab === "history" ? (
            <div className={styles.historyList}>
              {Object.keys(groupedHistory).length === 0 ? (
                <p style={{ color: "var(--muted)" }}>No previous runs found.</p>
              ) : (
                Object.entries(groupedHistory).map(([videoPath, runs]) => {
                  const origRun = runs.find(r => r.model_type === "original" || !r.model_type);
                  const smartRun = runs.find(r => r.model_type === "smart_crop");
                  const displayName = videoPath.split(/[/\\]/).pop();
                  
                  return (
                    <div key={videoPath} className={styles.historyItem} style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                      <div className={styles.historyMeta} style={{ marginBottom: '1rem' }}>
                        <span className={styles.historyTitle}>{displayName}</span>
                        <span className={styles.historyDate}>{new Date(runs[0].time * 1000).toLocaleString()}</span>
                      </div>
                      
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {origRun && !smartRun && (
                          <>
                            <button onClick={() => loadRun(origRun.id)} type="button" className={styles.jumpBtn}>View Original</button>
                            <button onClick={() => handleRunExisting(videoPath, "smart_crop", origRun.id)} type="button" className={styles.jumpBtn}>Run Smart Crop & Compare</button>
                          </>
                        )}
                        {!origRun && smartRun && (
                          <>
                            <button onClick={() => loadRun(smartRun.id)} type="button" className={styles.jumpBtn}>View Smart Crop</button>
                            <button onClick={() => handleRunExisting(videoPath, "original", smartRun.id)} type="button" className={styles.jumpBtn}>Run Original & Compare</button>
                          </>
                        )}
                        {origRun && smartRun && (
                          <button onClick={() => loadRun(origRun.id, smartRun.id)} type="button" className={styles.jumpBtn} style={{ background: 'rgba(59, 130, 246, 0.1)', borderColor: 'rgba(59, 130, 246, 0.3)', color: '#60a5fa' }}>
                            View Comparison Side-by-Side
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          ) : (
            <>
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

              <label className={styles.field} style={{ marginTop: '0.5rem' }}>
                <span>Model Selection</span>
                <select className={styles.input} value={modelType} onChange={e => setModelType(e.target.value as any)}>
                  <option value="original">Original Model</option>
                  <option value="smart_crop">Smart Crop Model</option>
                  <option value="compare">Run Both & Compare</option>
                </select>
              </label>

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

              {error && <div style={{ color: '#ef4444', background: 'rgba(239, 68, 68, 0.1)', padding: '1rem', borderRadius: '8px' }}>{error}</div>}

              <button type="submit" className={styles.submitBtn} disabled={loading || (tab === "upload" && !file) || (tab === "youtube" && !youtubeUrl)}>
                {loading ? "Processing Video (This may take a while)..." : "Run Calibration"}
              </button>
            </>
          )}
        </form>

        {loading && (
          <div className={styles.loader}>
            <div className={styles.spinner} />
            <p>Analyzing video frames... This may take a few minutes for longer videos.</p>
          </div>
        )}

        {result && (
          <div className={styles.resultsArea}>
            <div className={styles.uncutCard}>
              <div className={styles.uncutHeader}>
                <h3>Full Uncut Video Playback & Synchronized Scores {compareResult && "(Comparison Mode)"}</h3>
                <span className={styles.timeBadge}>
                  {formatTime(currentSec || 0)}
                </span>
              </div>
              <div className={styles.uncutBody} style={compareResult ? { gridTemplateColumns: '1fr 320px 320px' } : undefined}>
                <div className={styles.mainVideoWrapper}>
                  <video
                    ref={mainVideoRef}
                    src={getAPIUrl(`/calibration/clip/${encodeURIComponent(result.video_path.split(/[/\\]/).pop() || '')}`)}
                    controls
                    onTimeUpdate={(e) => setCurrentSec(e.currentTarget.currentTime)}
                  />
                </div>
                
                {/* Original Model Metrics */}
                <div className={styles.metricsPanel}>
                  <h4 style={{ margin: 0, color: '#8b5cf6' }}>Original Model</h4>
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
                          ? "⚡ Spike Alert Triggered"
                          : "⚠️ Sustained Alert Triggered"
                        : "✓ Normal / Below Thresholds"}
                    </div>
                  </div>
                </div>

                {/* Smart Crop Model Metrics */}
                {compareResult && (
                  <div className={styles.metricsPanel} style={{ borderColor: 'rgba(239, 68, 68, 0.3)' }}>
                    <h4 style={{ margin: 0, color: '#ef4444' }}>Smart Crop Model</h4>
                    <div className={styles.metricRow}>
                      <div className={styles.metricHeader}>
                        <span>Raw Confidence Score</span>
                        <span className={styles.metricValue}>
                          {activeCompareSample && activeCompareSample.raw_score !== undefined ? `${(activeCompareSample.raw_score * 100).toFixed(1)}%` : "--"}
                        </span>
                      </div>
                      <div className={styles.progressBarBg}>
                        <div
                          className={styles.progressBarFillRaw}
                          style={{ background: '#ef4444', width: activeCompareSample && activeCompareSample.raw_score !== undefined ? `${Math.min(100, activeCompareSample.raw_score * 100)}%` : "0%" }}
                        />
                      </div>
                    </div>

                    <div className={styles.metricRow}>
                      <div className={styles.metricHeader}>
                        <span>Smoothed Average</span>
                        <span className={styles.metricValue}>
                          {activeCompareSample && activeCompareSample.smoothed_avg !== undefined ? `${(activeCompareSample.smoothed_avg * 100).toFixed(1)}%` : "--"}
                        </span>
                      </div>
                      <div className={styles.progressBarBg}>
                        <div
                          className={styles.progressBarFillAvg}
                          style={{ background: '#f59e0b', width: activeCompareSample && activeCompareSample.smoothed_avg !== undefined ? `${Math.min(100, activeCompareSample.smoothed_avg * 100)}%` : "0%" }}
                        />
                      </div>
                    </div>

                    <div className={styles.statusBox}>
                      <span>Detection Status</span>
                      <div
                        className={`${styles.statusIndicator} ${
                          activeCompareSample?.triggered
                            ? activeCompareSample.trigger_reason === "spike"
                              ? styles.spikeActive
                              : styles.sustainedActive
                            : styles.normalActive
                        }`}
                      >
                        {activeCompareSample?.triggered
                          ? activeCompareSample.trigger_reason === "spike"
                            ? "⚡ Spike Alert Triggered"
                            : "⚠️ Sustained Alert Triggered"
                          : "✓ Normal / Below Thresholds"}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className={styles.chartCard}>
              <h3>Confidence Score Timeline (Click timeline to seek video)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={compareResult ? mergedLogs : result.per_sample_log}
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
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="time_sec" 
                    stroke="rgba(255,255,255,0.5)" 
                    tickFormatter={(val) => formatTime(Number(val))}
                    label={{ value: 'Time (MM:SS)', position: 'bottom', fill: 'rgba(255,255,255,0.5)' }} 
                  />
                  <YAxis 
                    domain={[0, 1]} 
                    stroke="rgba(255,255,255,0.5)" 
                    label={{ value: 'Confidence', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.5)' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(18, 26, 43, 0.9)', borderColor: 'var(--line)', color: '#fff', borderRadius: '8px' }}
                    labelFormatter={(val) => `Time: ${formatTime(Number(val))} (${Number(val).toFixed(1)}s)`}
                  />
                  <Legend verticalAlign="top" height={36}/>
                  <ReferenceLine y={spikeThreshold} label="Spike Threshold" stroke="#ef4444" strokeDasharray="3 3" />
                  <ReferenceLine y={alertThreshold} label="Alert Threshold" stroke="#f59e0b" strokeDasharray="3 3" />
                  {currentSec !== null && (
                    <ReferenceLine x={currentSec} stroke="#3b82f6" strokeWidth={2} label={{ value: 'Playback', fill: '#3b82f6', position: 'top' }} />
                  )}
                  <Line type="monotone" dataKey="raw_score" name="Original Raw" stroke="#8b5cf6" strokeWidth={1} dot={false} />
                  <Line type="monotone" dataKey="smoothed_avg" name="Original Avg" stroke="#2dd4bf" strokeWidth={2} dot={false} />
                  {compareResult && (
                    <>
                      <Line type="monotone" dataKey="raw_score_smart" name="Smart Crop Raw" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                      <Line type="monotone" dataKey="smoothed_avg_smart" name="Smart Crop Avg" stroke="#f59e0b" strokeWidth={2} dot={false} />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, color: 'var(--fg)', fontSize: '1.25rem', fontWeight: 500 }}>
                  Isolated Incident Clips 
                  {compareResult 
                    ? ` (Original: ${result.events.length}, Smart Crop: ${compareResult.events.length})` 
                    : ` (${result.events.length})`
                  }
                </h3>
                {compareResult && (
                  <select 
                    className={styles.input} 
                    style={{ width: 'auto', padding: '0.25rem 0.5rem', height: 'auto', minHeight: 'unset' }}
                    value={clipFilter}
                    onChange={(e) => setClipFilter(e.target.value as any)}
                  >
                    <option value="all">Show All Clips</option>
                    <option value="original">Original Only</option>
                    <option value="smart_crop">Smart Crop Only</option>
                  </select>
                )}
              </div>
              
              <div className={styles.clipsGrid}>
                {/* Render Original Events */}
                {(clipFilter === 'all' || clipFilter === 'original') && result.events.map((ev, i) => (
                  <div key={`orig-${i}`} className={styles.clipCard}>
                    <div className={styles.videoContainer}>
                      <video src={getAPIUrl(ev.url)} controls preload="metadata" />
                    </div>
                    <div className={styles.clipInfo}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className={styles.clipTime}>{formatTime(ev.start_sec)} - {formatTime(ev.end_sec)}</span>
                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                          {compareResult && <span className={styles.triggerBadge} style={{ background: '#8b5cf622', color: '#a78bfa' }}>Orig</span>}
                          <span className={`${styles.triggerBadge} ${ev.trigger === 'sustained' ? styles.sustained : ''}`}>
                            {ev.trigger}
                          </span>
                        </div>
                      </div>
                      <div className={styles.clipMeta}>
                        <span>Peak Score:</span>
                        <span style={{ color: '#fff', fontWeight: 500 }}>{(ev.peak_score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Render Smart Crop Events */}
                {compareResult && (clipFilter === 'all' || clipFilter === 'smart_crop') && compareResult.events.map((ev, i) => (
                  <div key={`smart-${i}`} className={styles.clipCard} style={{ borderColor: 'rgba(239, 68, 68, 0.3)' }}>
                    <div className={styles.videoContainer}>
                      <video src={getAPIUrl(ev.url)} controls preload="metadata" />
                    </div>
                    <div className={styles.clipInfo}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className={styles.clipTime}>{formatTime(ev.start_sec)} - {formatTime(ev.end_sec)}</span>
                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                          <span className={styles.triggerBadge} style={{ background: '#ef444422', color: '#f87171' }}>Smart</span>
                          <span className={`${styles.triggerBadge} ${ev.trigger === 'sustained' ? styles.sustained : ''}`}>
                            {ev.trigger}
                          </span>
                        </div>
                      </div>
                      <div className={styles.clipMeta}>
                        <span>Peak Score:</span>
                        <span style={{ color: '#fff', fontWeight: 500 }}>{(ev.peak_score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
                
                {result.events.length === 0 && (!compareResult || compareResult.events.length === 0) && (
                  <p style={{ color: 'var(--muted)' }}>No segments crossed the threshold.</p>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

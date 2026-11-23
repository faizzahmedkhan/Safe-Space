import { Check, Download, Play, RefreshCw, ShieldAlert } from "lucide-react";
import { evidenceUrl, type AlertItem, type LogItem } from "@/lib/api";
import { HIGH_CONFIDENCE, formatAlertTime } from "@/lib/format";
import styles from "./AlertFeed.module.css";

export type Tab = "alerts" | "logs";
export type Scope = "selected" | "all";
export type Verdict = "safe" | "unsafe";

type Props = {
  tab: Tab;
  onTab: (t: Tab) => void;
  scope: Scope;
  onScope: (s: Scope) => void;
  hasSelectedCamera: boolean;
  alerts: AlertItem[];
  logs: LogItem[];
  openEvidence: string | null;
  onOpen: (evidenceFile: string | null) => void;
  verdicts: Record<string, Verdict>;
  reviewBusy: boolean;
  backendOnline: boolean;
  onReanalyze: (evidenceFile: string) => void;
  onLabel: (evidenceFile: string, label: Verdict) => void;
  labelMsg: string | null;
};

export default function AlertFeed({
  tab,
  onTab,
  scope,
  onScope,
  hasSelectedCamera,
  alerts,
  logs,
  openEvidence,
  onOpen,
  verdicts,
  reviewBusy,
  backendOnline,
  onReanalyze,
  onLabel,
  labelMsg,
}: Props) {
  return (
    <section aria-labelledby="feed-heading">
      <div className={styles.head}>
        <h2 id="feed-heading">Alerts &amp; logs</h2>
      </div>

      <div className={styles.chips} role="group" aria-label="Feed view">
        <button
          type="button"
          className={`${styles.chip} ${tab === "alerts" ? styles.chipOn : ""}`}
          aria-pressed={tab === "alerts"}
          onClick={() => onTab("alerts")}
        >
          Alerts ({alerts.length})
        </button>
        <button
          type="button"
          className={`${styles.chip} ${tab === "logs" ? styles.chipOn : ""}`}
          aria-pressed={tab === "logs"}
          onClick={() => onTab("logs")}
        >
          Logs ({logs.length})
        </button>
      </div>

      {tab === "alerts" && (
        <div className={styles.chips} role="group" aria-label="Camera scope">
          <button
            type="button"
            className={`${styles.chip} ${scope === "selected" ? styles.chipOn : ""}`}
            aria-pressed={scope === "selected"}
            onClick={() => onScope("selected")}
            disabled={!hasSelectedCamera}
          >
            Selected camera
          </button>
          <button
            type="button"
            className={`${styles.chip} ${scope === "all" ? styles.chipOn : ""}`}
            aria-pressed={scope === "all"}
            onClick={() => onScope("all")}
          >
            All cameras
          </button>
        </div>
      )}

      {labelMsg && <p className={styles.msg}>{labelMsg}</p>}

      {tab === "alerts" ? (
        <div className={styles.list}>
          {alerts.length === 0 && (
            <p className={styles.empty}>
              {scope === "selected" && hasSelectedCamera ? "No alerts for this camera yet." : "No alerts yet."}
            </p>
          )}
          {alerts.map((alert) => {
            const key = alert.evidence_file;
            const isOpen = openEvidence === key;
            const ready = !alert.status || alert.status === "ready";
            const high = alert.confidence >= HIGH_CONFIDENCE;
            const verdict = verdicts[key];
            const videoSrc = evidenceUrl(alert.evidence_url || alert.evidence_file);
            return (
              <article
                key={alert.timestamp + key}
                className={`${styles.item} ${isOpen ? styles.itemOpen : ""} ${high ? styles.high : ""}`}
              >
                <button
                  type="button"
                  className={styles.row}
                  aria-expanded={isOpen}
                  disabled={!ready}
                  onClick={() => onOpen(isOpen ? null : key)}
                >
                  <span className={styles.thumb} aria-hidden="true">
                    <Play fill="currentColor" />
                  </span>
                  <span style={{ minWidth: 0 }}>
                    <span className={styles.title}>
                      <i className={styles.sev} />
                      <span>{alert.camera_id || "Unknown camera"}</span>
                      {alert.status && alert.status !== "ready" && (
                        <span className={`${styles.badge} ${alert.status === "failed" ? styles.badgeBad : ""}`}>
                          {alert.status}
                        </span>
                      )}
                    </span>
                    <span className={`${styles.time} mono`} style={{ display: "block" }}>
                      {formatAlertTime(alert.timestamp)}
                    </span>
                  </span>
                  <span className={styles.right}>
                    <span className={`${styles.conf} mono`} style={{ display: "block" }}>
                      {alert.confidence}%
                    </span>
                    {verdict && (
                      <span className={`${styles.verdict} ${verdict === "unsafe" ? styles.vFight : styles.vFalse}`}>
                        {verdict === "unsafe" ? "Confirmed" : "False alarm"}
                      </span>
                    )}
                  </span>
                </button>

                {isOpen && ready && (
                  <div className={styles.detail}>
                    <video className={styles.video} src={videoSrc} controls playsInline preload="metadata" />
                    <p className={`${styles.meta} mono`}>
                      {alert.timestamp}
                      {alert.peak != null && ` · peak ${alert.peak}%`}
                      {alert.n_triggers != null && ` · ${alert.n_triggers} triggers`}
                      {alert.trigger && ` · ${alert.trigger}`}
                    </p>
                    <div className={styles.seg}>
                      <button
                        type="button"
                        className={`btn btn-sm ${verdict === "unsafe" ? styles.segOn : ""}`}
                        onClick={() => onLabel(key, "unsafe")}
                      >
                        <ShieldAlert aria-hidden="true" />
                        True fight
                      </button>
                      <button
                        type="button"
                        className={`btn btn-sm ${verdict === "safe" ? styles.segOn : ""}`}
                        onClick={() => onLabel(key, "safe")}
                      >
                        <Check aria-hidden="true" />
                        False alarm
                      </button>
                    </div>
                    <div className={styles.more}>
                      <button
                        type="button"
                        className="btn btn-sm"
                        disabled={reviewBusy || !backendOnline}
                        onClick={() => onReanalyze(key)}
                      >
                        <RefreshCw aria-hidden="true" />
                        Re-analyze
                      </button>
                      <a className="btn btn-sm" href={videoSrc} download target="_blank" rel="noreferrer">
                        <Download aria-hidden="true" />
                        Download
                      </a>
                    </div>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      ) : (
        <div className={styles.list}>
          {logs.length === 0 && <p className={styles.empty}>No system logs yet.</p>}
          {logs.map((log, i) => (
            <article key={`${log.timestamp}-${i}`} className={styles.item} style={{ padding: "8px" }}>
              <p className={`${styles.time} mono`} style={{ marginTop: 0 }}>
                {formatAlertTime(log.timestamp)} · {log.kind}/{log.level}
              </p>
              <p
                style={{
                  marginTop: 2,
                  color: log.level === "error" || log.level === "alert" ? "var(--crit-ink)" : "var(--ink)",
                }}
              >
                {log.message}
              </p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

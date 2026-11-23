import { Unplug, Trash2 } from "lucide-react";
import { feedUrl, type StatusResponse } from "@/lib/api";
import styles from "./MonitorPanel.module.css";

type Props = {
  cam: StatusResponse;
  connected: boolean;
  stateMessage: string;
  busy: boolean;
  feedKey: number;
  feedBust: number;
  feedLost: boolean;
  onFeedError: () => void;
  onFeedLoad: () => void;
  onDisconnect: () => void;
  onRemove: () => void;
};

export default function MonitorPanel({
  cam,
  connected,
  stateMessage,
  busy,
  feedKey,
  feedBust,
  feedLost,
  onFeedError,
  onFeedLoad,
  onDisconnect,
  onRemove,
}: Props) {
  const name = cam.label || cam.camera_id;
  const raw = cam.latest_raw_pct;
  const smoothed = cam.smoothed_avg_pct;
  const threshold = cam.alert_threshold ?? 60;
  const hot = smoothed != null && smoothed >= threshold;

  return (
    <section aria-labelledby="monitor-heading">
      <div className={styles.head}>
        <div>
          <h2 id="monitor-heading">{name}</h2>
          <p className={styles.state}>
            <i className={`dot ${connected ? "" : "dot-crit"}`} />
            {stateMessage}
          </p>
        </div>
        <div className={styles.actions}>
          <button
            type="button"
            className="btn btn-sm"
            onClick={onDisconnect}
            disabled={!cam.camera_active || busy}
          >
            <Unplug aria-hidden="true" />
            Disconnect
          </button>
          <button type="button" className="btn btn-sm" onClick={onRemove} disabled={busy}>
            <Trash2 aria-hidden="true" />
            Remove
          </button>
        </div>
      </div>

      <div className={styles.grid}>
        <div className={styles.frame}>
          {connected ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                key={`${cam.camera_id}-${feedKey}`}
                src={feedUrl(cam.camera_id, feedBust)}
                alt={`Live feed from ${name}`}
                onError={onFeedError}
                onLoad={onFeedLoad}
              />
              {feedLost && (
                <div className={styles.overlay} role="status">
                  Feed lost, retrying…
                </div>
              )}
            </>
          ) : (
            <div className={styles.empty}>No live feed. Connect the camera to start monitoring.</div>
          )}
        </div>

        <div className={styles.scores}>
          <div className={styles.scoreRow}>
            <div className={styles.score}>
              <span className="label">Raw spike</span>
              <strong className={styles.scoreValue}>{raw != null ? `${raw.toFixed(1)}%` : "—"}</strong>
            </div>
            <div className={styles.score}>
              <span className="label">Smoothed</span>
              <strong className={`${styles.scoreValue} ${hot ? styles.scoreHot : ""}`}>
                {smoothed != null ? `${smoothed.toFixed(1)}%` : "—"}
              </strong>
            </div>
          </div>

          <div>
            <div className={styles.track}>
              <div
                className={`${styles.fill} ${hot ? styles.fillHot : ""}`}
                style={{ width: `${Math.min(100, Math.max(0, smoothed ?? 0))}%` }}
              />
              <div
                className={styles.threshold}
                style={{ left: `${threshold}%` }}
                title={`Alert threshold ${threshold}%`}
              />
            </div>
            <p className={`${styles.caption} mono`}>
              Threshold {threshold}% · Buffer {cam.buffer_fill ?? "—"} · Frames {cam.total_frames ?? 0}
              {cam.score_warmup ? ` · Warmup ${cam.score_warmup}` : ""}
            </p>
            <p className={`${styles.caption} mono`}>
              {cam.actual_fps != null &&
                `FPS ${cam.actual_fps.toFixed(1)}${cam.target_fps ? ` / ${cam.target_fps}` : ""}`}
              {cam.coverage_pct != null && ` · Coverage ${cam.coverage_pct.toFixed(0)}%`}
              {cam.framing_suitable_ratio != null &&
                ` · Framing OK ${(cam.framing_suitable_ratio * 100).toFixed(0)}%`}
            </p>
          </div>

          {(cam.frozen || cam.degraded) && (
            <span className={`${styles.badge} ${cam.frozen ? styles.badgeBad : ""}`}>
              {cam.frozen ? "Feed frozen" : "Degraded"}
            </span>
          )}
          {cam.scene?.note && (
            <p className={`${styles.note} ${cam.scene.suitable_for_model ? "" : styles.noteBad}`}>
              Framing: {cam.scene.method} · subject {((cam.scene.subject_area_frac ?? 0) * 100).toFixed(1)}% —{" "}
              {cam.scene.note}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

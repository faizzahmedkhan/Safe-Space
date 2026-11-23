import { Plus } from "lucide-react";
import type { StatusResponse } from "@/lib/api";
import styles from "./CameraWall.module.css";

type Props = {
  cameras: StatusResponse[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAdd: () => void;
};

function stateOf(c: StatusResponse): "hot" | "live" | "off" {
  if (!c.camera_receiving) return "off";
  const s = c.smoothed_avg_pct;
  return s != null && s >= c.alert_threshold ? "hot" : "live";
}

export default function CameraWall({ cameras, selectedId, onSelect, onAdd }: Props) {
  return (
    <div className={styles.wall}>
      {cameras.map((c) => {
        const state = stateOf(c);
        const pct = c.smoothed_avg_pct;
        const chipLabel = c.frozen
          ? "Frozen"
          : state === "hot"
            ? "Possible fight"
            : state === "live"
              ? "Live"
              : "Offline";
        const sub = c.camera_kind === "webcam" ? "Webcam" : c.ip || "Not connected";
        return (
          <button
            key={c.camera_id}
            type="button"
            className={`${styles.tile} ${state === "hot" ? styles.hot : ""} ${
              selectedId === c.camera_id ? styles.selected : ""
            }`}
            onClick={() => onSelect(c.camera_id)}
            aria-pressed={selectedId === c.camera_id}
          >
            <span className={styles.chip}>
              <i className={`dot ${state === "off" ? "dot-off" : ""}`} />
              {chipLabel}
            </span>
            <span className={styles.meta}>
              <span className={styles.name}>
                {c.label || c.camera_id}
                <span className={`${styles.sub} mono`}>{sub}</span>
              </span>
              <span className={`${styles.conf} mono`}>
                {state === "off" || pct == null ? "—" : `${pct.toFixed(0)}%`}
              </span>
            </span>
            <span className={styles.meter}>
              <i style={{ width: `${Math.min(100, Math.max(0, state === "off" ? 0 : (pct ?? 0)))}%` }} />
            </span>
          </button>
        );
      })}
      <button type="button" className={styles.add} onClick={onAdd}>
        <span>
          <Plus aria-hidden="true" />
          Add camera
        </span>
      </button>
    </div>
  );
}

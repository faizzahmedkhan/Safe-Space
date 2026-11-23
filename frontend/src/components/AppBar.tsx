import Link from "next/link";
import styles from "./AppBar.module.css";

type Props = {
  active: "monitor" | "calibration";
  backend?: { state: "online" | "offline" | "unauthorized" };
  watcherAgeMs?: number | null;
  modelVersion?: string | null;
};

export default function AppBar({ active, backend, watcherAgeMs, modelVersion }: Props) {
  const online = backend?.state === "online";
  return (
    <header className={`${styles.bar} gutter`}>
      <div className={`${styles.inner} wrap`}>
        <Link href="/" className={styles.logo}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 2.5 4 5.5v6c0 5 3.3 8.6 8 10 4.7-1.4 8-5 8-10v-6l-8-3Z" fill="#4d86ff" />
            <circle cx="12" cy="11.5" r="3" fill="#0a1b33" />
          </svg>
          SafeSpace
        </Link>
        <nav className={styles.nav} aria-label="Main">
          <Link href="/" aria-current={active === "monitor" ? "page" : undefined}>
            Monitor
          </Link>
          <Link href="/calibration" aria-current={active === "calibration" ? "page" : undefined}>
            Calibration
          </Link>
        </nav>
        {backend && (
          <div className={styles.status} aria-live="polite">
            <span>
              <i className={`dot ${online ? "" : "dot-crit"}`} />
              {backend.state === "unauthorized"
                ? "Unauthorized"
                : online
                  ? "Backend online"
                  : "Backend offline"}
            </span>
            {watcherAgeMs != null && (
              <span title="Time since the camera watcher last completed a cycle">
                <i className={`dot ${watcherAgeMs > 15000 ? "dot-crit" : ""}`} />
                Watcher <span className="mono">{(watcherAgeMs / 1000).toFixed(1)}s</span>
              </span>
            )}
            {modelVersion && <span className="mono">{modelVersion}</span>}
          </div>
        )}
      </div>
    </header>
  );
}

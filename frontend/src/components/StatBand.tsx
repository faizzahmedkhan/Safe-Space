import styles from "./StatBand.module.css";

type Props = {
  online: number;
  total: number;
  active: number;
  alerts: number;
  latestConfidence: number | null;
};

export default function StatBand({ online, total, active, alerts, latestConfidence }: Props) {
  return (
    <section className={styles.stats} aria-label="Summary">
      <div className={styles.stat}>
        <div className="label">Cameras online</div>
        <div className={styles.value}>
          {online}
          <span className={styles.of}>/{total}</span>
        </div>
      </div>
      <div className={`${styles.stat} ${active > 0 ? styles.crit : ""}`}>
        <div className="label">Above threshold</div>
        <div className={styles.value}>{active}</div>
      </div>
      <div className={styles.stat}>
        <div className="label">Alerts</div>
        <div className={styles.value}>{alerts}</div>
      </div>
      <div className={styles.stat}>
        <div className="label">Live confidence</div>
        <div className={styles.value}>
          {latestConfidence != null ? `${latestConfidence.toFixed(1)}%` : "—"}
        </div>
      </div>
    </section>
  );
}

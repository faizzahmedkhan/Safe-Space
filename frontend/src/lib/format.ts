const TS_RE = /(\d{4})-(\d{2})-(\d{2})[-_ T](\d{2})[-_:](\d{2})[-_:](\d{2})/;

/** "2026-08-06_05-32-48" -> "Today 05:32" / "Yesterday 05:32" / "Aug 6, 05:32". Falls back to the raw string. */
export function formatAlertTime(raw: string, now: Date = new Date()): string {
  const m = TS_RE.exec(raw);
  if (!m) return raw;
  const [, y, mo, d, hh, mm] = m;
  const when = new Date(Number(y), Number(mo) - 1, Number(d));
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const days = Math.round((startOfToday.getTime() - when.getTime()) / 86_400_000);
  const clock = `${hh}:${mm}`;
  if (days === 0) return `Today ${clock}`;
  if (days === 1) return `Yesterday ${clock}`;
  const month = when.toLocaleString("en-US", { month: "short" });
  return `${month} ${Number(d)}, ${clock}`;
}

/** Confidence at or above this is shown as high severity. */
export const HIGH_CONFIDENCE = 75;

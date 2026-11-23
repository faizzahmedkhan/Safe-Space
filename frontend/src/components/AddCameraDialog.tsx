import { useEffect, useRef, type FormEvent } from "react";
import { Plus, Video } from "lucide-react";
import styles from "./AddCameraDialog.module.css";

type Props = {
  open: boolean;
  onClose: () => void;
  onSubmit: (e: FormEvent) => void;
  onAddWebcam: () => void;
  busy: boolean;
  backendOnline: boolean;
  error: string | null;
  id: string;
  setId: (v: string) => void;
  label: string;
  setLabel: (v: string) => void;
  ip: string;
  setIp: (v: string) => void;
  port: string;
  setPort: (v: string) => void;
  path: string;
  setPath: (v: string) => void;
};

export default function AddCameraDialog(p: Props) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (p.open && !d.open) d.showModal();
    if (!p.open && d.open) d.close();
  }, [p.open]);

  return (
    <dialog ref={ref} className={styles.dialog} onClose={p.onClose} aria-labelledby="add-camera-title">
      <form className={styles.form} onSubmit={p.onSubmit}>
        <h2 id="add-camera-title" className={styles.title}>
          Add camera
        </h2>
        <p className={styles.lede}>Connect a phone camera by IP, or use this computer&apos;s webcam.</p>

        <label className={styles.field}>
          <span>Camera ID</span>
          <input value={p.id} onChange={(e) => p.setId(e.target.value)} placeholder="cam-1" required />
        </label>
        <label className={styles.field}>
          <span>Label</span>
          <input value={p.label} onChange={(e) => p.setLabel(e.target.value)} placeholder="Main Hallway" />
        </label>
        <label className={styles.field}>
          <span>Phone IP (optional)</span>
          <input value={p.ip} onChange={(e) => p.setIp(e.target.value)} placeholder="192.168.1.5" />
        </label>
        {p.ip && (
          <div className={styles.row}>
            <label className={styles.field}>
              <span>Port</span>
              <input value={p.port} onChange={(e) => p.setPort(e.target.value)} />
            </label>
            <label className={styles.field}>
              <span>Stream path</span>
              <input value={p.path} onChange={(e) => p.setPath(e.target.value)} />
            </label>
          </div>
        )}

        {p.error && (
          <p className={styles.error} role="alert">
            {p.error}
          </p>
        )}

        <div className={styles.footer}>
          <button type="button" className="btn" onClick={p.onAddWebcam} disabled={p.busy || !p.backendOnline}>
            <Video aria-hidden="true" />
            Use webcam
          </button>
          <div className={styles.group}>
            <button type="button" className="btn" onClick={p.onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={p.busy || !p.backendOnline}>
              <Plus aria-hidden="true" />
              Add &amp; connect
            </button>
          </div>
        </div>
      </form>
    </dialog>
  );
}

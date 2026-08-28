import React, { useState, FormEvent } from "react";
import { useSafeSpace } from "./SafeSpaceProvider";
import { createCamera, connectCamera, connectWebcam, connectYouTube } from "@/lib/api";

type Props = {
  onClose: () => void;
};

export function AddCameraModal({ onClose }: Props) {
  const { setSelectedCameraId, backendOnline, refresh } = useSafeSpace();

  const [newCameraId, setNewCameraId] = useState("");
  const [newCameraLabel, setNewCameraLabel] = useState("");
  const [ip, setIp] = useState("");
  const [port, setPort] = useState("8080");
  const [path, setPath] = useState("/video");
  const [wifiName, setWifiName] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  async function onAddCamera(e: FormEvent) {
    e.preventDefault();
    setActionError(null);
    setBusy(true);
    try {
      const cid = newCameraId.trim();
      if (!cid) throw new Error("Camera ID is required");
      
      await createCamera(cid, newCameraLabel.trim());
      
      if (ip.trim()) {
        await connectCamera(cid, {
          ip: ip.trim(),
          port: Number(port) || 8080,
          path: path.trim() || "/video",
          wifi_name: wifiName.trim(),
        });
      } else if (youtubeUrl.trim()) {
        await connectYouTube(cid, youtubeUrl.trim());
      }
      setSelectedCameraId(cid);
      await refresh();
      onClose();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Add camera failed");
    } finally {
      setBusy(false);
    }
  }

  async function onAddWebcam() {
    setActionError(null);
    setBusy(true);
    try {
      const cid = "webcam-" + Math.floor(Math.random() * 1000);
      await createCamera(cid, "Laptop Webcam");
      await connectWebcam(cid);
      setSelectedCameraId(cid);
      await refresh();
      onClose();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Webcam connect failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">Add camera</div>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <form className="add-cam" onSubmit={onAddCamera} style={{ border: "none", padding: 0 }}>
          <div>
            <div className="field-label" style={{ marginBottom: "4px" }}>ID</div>
            <input
              className="field"
              style={{ width: "100%" }}
              placeholder="cam-2"
              value={newCameraId}
              onChange={(e) => setNewCameraId(e.target.value)}
              required
            />
          </div>
          <div>
            <div className="field-label" style={{ marginBottom: "4px" }}>Label</div>
            <input
              className="field"
              style={{ width: "100%" }}
              placeholder="East stairwell"
              value={newCameraLabel}
              onChange={(e) => setNewCameraLabel(e.target.value)}
            />
          </div>
          <div>
            <div className="field-label" style={{ marginBottom: "4px" }}>Phone IP (Optional)</div>
            <input
              className="field"
              style={{ width: "100%" }}
              placeholder="192.168.1.5"
              value={ip}
              onChange={(e) => setIp(e.target.value)}
            />
          </div>
          {ip && (
            <>
              <div>
                <div className="field-label" style={{ marginBottom: "4px" }}>Port</div>
                <input
                  className="field"
                  style={{ width: "100%" }}
                  value={port}
                  onChange={(e) => setPort(e.target.value)}
                />
              </div>
              <div>
                <div className="field-label" style={{ marginBottom: "4px" }}>Stream path</div>
                <input
                  className="field"
                  style={{ width: "100%" }}
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                />
              </div>
            </>
          )}
          {!ip && (
            <div>
              <div className="field-label" style={{ marginBottom: "4px" }}>YouTube Livestream URL (Optional)</div>
              <input
                className="field"
                style={{ width: "100%" }}
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
              />
            </div>
          )}
          <div style={{ marginTop: "8px", display: "flex", gap: "8px", flexDirection: "column" }}>
            <button type="submit" className="btn btn-action" disabled={busy || !backendOnline}>
              Connect
            </button>
            <button type="button" className="btn btn-ghost" onClick={onAddWebcam} disabled={busy || !backendOnline}>
              Add Webcam
            </button>
          </div>
          {actionError && <div className="error-text" style={{ fontSize: "11px", marginTop: "4px" }}>{actionError}</div>}
        </form>
      </div>
    </div>
  );
}

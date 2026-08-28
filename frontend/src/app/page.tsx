"use client";

import React, { useState } from "react";
import { IconVideo } from "@tabler/icons-react";
import { useSafeSpace } from "@/components/SafeSpaceProvider";
import { Topbar } from "@/components/Topbar";
import { AddCameraModal } from "@/components/AddCameraModal";
import { ConfirmModal } from "@/components/ConfirmModal";
import { disconnectCamera, connectCamera, connectWebcam, connectYouTube, deleteCamera, feedUrl } from "@/lib/api";

export default function MonitorPage() {
  const { cameras, backendOnline, selectedCameraId, refresh } = useSafeSpace();
  const [showAddModal, setShowAddModal] = useState(false);
  const [showConfirmRemove, setShowConfirmRemove] = useState(false);

  const selectedCam = cameras.find((c) => c.camera_id === selectedCameraId);

  const isActive = selectedCam?.camera_active;
  const isReceiving = selectedCam?.camera_receiving;
  const raw = selectedCam?.latest_raw_pct ?? 0;
  const smoothed = selectedCam?.smoothed_avg_pct ?? 0;
  const threshold = selectedCam?.alert_threshold ?? 75;

  const handleDisconnect = async () => {
    if (!selectedCam) return;
    try {
      await disconnectCamera(selectedCam.camera_id);
      await refresh();
    } catch (e) {
      console.error("Disconnect failed", e);
    }
  };

  const handleReconnect = async () => {
    if (!selectedCam) return;
    try {
      if (selectedCam.source === "youtube" && selectedCam.youtube_url) {
        await connectYouTube(selectedCam.camera_id, selectedCam.youtube_url);
      } else if (selectedCam.source === "webcam") {
        await connectWebcam(selectedCam.camera_id);
      } else if (selectedCam.ip) {
        await connectCamera(selectedCam.camera_id, {
          ip: selectedCam.ip,
          port: selectedCam.port,
          path: selectedCam.path,
        });
      }
      await refresh();
    } catch (e) {
      console.error("Reconnect failed", e);
    }
  };

  const handleRemove = async () => {
    if (!selectedCam) return;
    try {
      await deleteCamera(selectedCam.camera_id);
      setShowConfirmRemove(false);
      await refresh();
    } catch (e) {
      console.error("Remove failed", e);
    }
  };

  return (
    <main className="main">
      <Topbar />

      <div className="status-row" style={{ alignItems: "center" }}>
        <div className={`status-chip ${!backendOnline ? "offline" : ""}`}>
          <span className="dot"></span>
          Backend {backendOnline ? "online" : "offline"}
        </div>
        {selectedCam && (
          <div className={`status-chip ${!isReceiving ? "offline" : ""}`}>
            <span className="dot"></span>
            {selectedCam.camera_id} {isReceiving ? "connected" : "disconnected"}
          </div>
        )}
        {cameras.length > 0 && (
          <div style={{ marginLeft: "auto" }}>
            <button className="btn btn-action" onClick={() => setShowAddModal(true)}>
              + Add camera
            </button>
          </div>
        )}
      </div>

      <section className="monitor">
        {selectedCam ? (
          <div className="monitor-grid">
            <div className={`feed-card ${!isReceiving ? "offline" : ""}`}>
              <div className="feed-head">
                <div>
                  <div className="feed-title">{selectedCam.label || selectedCam.camera_id}</div>
                  <div className="feed-sub">Live view — smoothed violence confidence</div>
                </div>
                <div className="feed-actions">
                  {isActive ? (
                    <button className="btn btn-ghost" onClick={handleDisconnect}>
                      Disconnect
                    </button>
                  ) : (
                    <button className="btn btn-action" onClick={handleReconnect}>
                      Reconnect
                    </button>
                  )}
                  <button className="btn btn-ghost" onClick={() => setShowConfirmRemove(true)}>
                    Remove
                  </button>
                </div>
              </div>
              <div className="feed-visual">
                {isReceiving && (
                  <div className="live-badge">
                    <span className="dot"></span>LIVE
                  </div>
                )}
                {isReceiving ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={feedUrl(selectedCam.camera_id)} alt="Live feed" />
                ) : (
                  <div className="feed-placeholder">
                    <IconVideo size={14} /> camera feed offline
                  </div>
                )}
              </div>
            </div>

            <div className="stat-panel">
              <div>
                <div className="stat-block-label">Raw spike</div>
                <div className="stat-value">{raw.toFixed(1)}%</div>
              </div>
              <div>
                <div className="stat-block-label">Smoothed average</div>
                <div className="stat-value small">{smoothed.toFixed(1)}%</div>
                <div className="threshold-bar">
                  <div
                    className={`threshold-fill ${smoothed >= threshold ? "alert" : ""}`}
                    style={{ width: `${Math.min(100, Math.max(0, smoothed))}%` }}
                  ></div>
                  <div className="threshold-marker" style={{ left: `${threshold}%` }}></div>
                </div>
              </div>
              <div>
                <div className="meta-line">
                  <span>Alert threshold</span>
                  <span>{threshold}%</span>
                </div>
                <div className="meta-line">
                  <span>Buffer</span>
                  <span>{selectedCam.buffer_fill || "—"}</span>
                </div>
                <div className="meta-line">
                  <span>Frames</span>
                  <span>{selectedCam.total_frames || 0}</span>
                </div>
                <div className="meta-line">
                  <span>Framing</span>
                  <span>{selectedCam.scene?.method || "—"}</span>
                </div>
                <div className="meta-line">
                  <span>Subject area</span>
                  <span>{selectedCam.scene?.subject_area_frac ? `${(selectedCam.scene.subject_area_frac * 100).toFixed(1)}%` : "—"}</span>
                </div>
              </div>
              {selectedCam.scene?.note && (
                <div className={selectedCam.scene.suitable_for_model ? "hint-text" : "error-text"} style={{ fontSize: "11px" }}>
                  {selectedCam.scene.note}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: "40px",
            textAlign: "center",
            color: "var(--ink-dim)"
          }}>
            Select or add a camera to view its live monitor.
            <div style={{ marginTop: "16px" }}>
              <button className="btn btn-action" onClick={() => setShowAddModal(true)}>
                + Add camera
              </button>
            </div>
          </div>
        )}
      </section>
      
      {showAddModal && <AddCameraModal onClose={() => setShowAddModal(false)} />}
      {showConfirmRemove && (
        <ConfirmModal
          title="Remove camera"
          message={`Are you sure you want to completely remove "${selectedCam?.label || selectedCam?.camera_id}" from SafeSpace?`}
          confirmText="Remove"
          onConfirm={handleRemove}
          onCancel={() => setShowConfirmRemove(false)}
        />
      )}
    </main>
  );
}

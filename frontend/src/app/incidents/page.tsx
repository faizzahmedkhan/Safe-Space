"use client";

import React, { useState, useMemo } from "react";
import { Topbar } from "@/components/Topbar";
import { useSafeSpace } from "@/components/SafeSpaceProvider";
import { updateAlertReview, evidenceUrl, AlertItem } from "@/lib/api";
import { IconPlayerPlayFilled } from "@tabler/icons-react";

export default function IncidentsPage() {
  const { alerts, cameras, refresh } = useSafeSpace();
  const [activeFilter, setActiveFilter] = useState<"all" | "spike" | "sustained" | "unreviewed">("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reviewBusy, setReviewBusy] = useState(false);

  // We use evidence_file as the unique identifier
  const filteredIncidents = useMemo(() => {
    if (activeFilter === "all") return alerts;
    if (activeFilter === "unreviewed") return alerts.filter((i) => i.review_status === "unreviewed");
    return alerts.filter((i) => i.trigger === activeFilter);
  }, [alerts, activeFilter]);

  const selectedIncident = useMemo(() => {
    if (!selectedId && filteredIncidents.length > 0) return filteredIncidents[0];
    return filteredIncidents.find((i) => i.evidence_file === selectedId) || null;
  }, [selectedId, filteredIncidents]);

  const handleReview = async (status: "confirmed" | "dismissed") => {
    if (!selectedIncident) return;
    setReviewBusy(true);
    try {
      await updateAlertReview(selectedIncident.evidence_file, status);
      await refresh();
    } catch (e) {
      console.error("Failed to update review status", e);
    } finally {
      setReviewBusy(false);
    }
  };

  const getCameraName = (cid?: string) => {
    if (!cid) return "Unknown";
    const cam = cameras.find((c) => c.camera_id === cid);
    return cam?.label || cid;
  };

  const formatTime = (ts: string) => {
    const parts = ts.split("_");
    if (parts.length < 2) return ts;
    const dateParts = parts[0].split("-");
    const timeParts = parts[1].split("-");
    return `${dateParts[1]}/${dateParts[2]} ${timeParts[0]}:${timeParts[1]}`;
  };

  return (
    <main className="main">
      <Topbar />
      <section className="incidents active">
        <div className="filter-row">
          <div
            className={`filter-chip ${activeFilter === "all" ? "active" : ""}`}
            onClick={() => setActiveFilter("all")}
          >
            All cameras
          </div>
          <div
            className={`filter-chip ${activeFilter === "spike" ? "active" : ""}`}
            onClick={() => setActiveFilter("spike")}
          >
            Spike only
          </div>
          <div
            className={`filter-chip ${activeFilter === "sustained" ? "active" : ""}`}
            onClick={() => setActiveFilter("sustained")}
          >
            Sustained only
          </div>
          <div
            className={`filter-chip ${activeFilter === "unreviewed" ? "active" : ""}`}
            onClick={() => setActiveFilter("unreviewed")}
          >
            Unreviewed
          </div>
        </div>

        <div className="incidents-grid">
          <div className="incident-list">
            {filteredIncidents.length === 0 && (
              <div style={{ padding: "20px", color: "var(--ink-dim)", textAlign: "center" }}>
                No incidents match the filter.
              </div>
            )}
            {filteredIncidents.map((i) => {
              const isSelected = (selectedId || filteredIncidents[0]?.evidence_file) === i.evidence_file;
              return (
                <div
                  key={i.evidence_file}
                  className={`incident-row ${isSelected ? "selected" : ""}`}
                  onClick={() => setSelectedId(i.evidence_file)}
                >
                  <span
                    className="row-dot"
                    style={{ background: i.trigger === "spike" ? "var(--alert)" : "var(--watch)" }}
                  ></span>
                  <span className={`trig-badge trig-${i.trigger || "spike"}`}>{i.trigger || "spike"}</span>
                  <div>
                    <div className="incident-cam">{getCameraName(i.camera_id)}</div>
                    <div className="incident-time">{formatTime(i.timestamp)}</div>
                  </div>
                  <div className="incident-conf">{i.confidence}%</div>
                  <div className={`review-badge ${i.review_status}`}>{i.review_status}</div>
                </div>
              );
            })}
          </div>

          <div className="detail-panel">
            {selectedIncident ? (
              <>
                <div className="detail-video">
                  <video
                    src={evidenceUrl(selectedIncident.evidence_file)}
                    controls
                    playsInline
                    autoPlay
                    muted
                  />
                  {!selectedIncident.evidence_url && (
                    <div className="feed-placeholder" style={{ position: "absolute", bottom: "10px", left: "12px", zIndex: 2 }}>
                      {selectedIncident.evidence_file}
                    </div>
                  )}
                </div>
                <div className="detail-body">
                  <div className="detail-title">{getCameraName(selectedIncident.camera_id)}</div>
                  <div className="detail-sub">
                    {formatTime(selectedIncident.timestamp)} · triggered by {selectedIncident.trigger || "spike"} ·{" "}
                    {selectedIncident.confidence}% confidence
                  </div>
                  <div className="meta-line">
                    <span>Trigger</span>
                    <span>{selectedIncident.trigger || "spike"}</span>
                  </div>
                  <div className="meta-line">
                    <span>Confidence</span>
                    <span>{selectedIncident.confidence}%</span>
                  </div>
                  <div className="meta-line">
                    <span>Review status</span>
                    <span>{selectedIncident.review_status}</span>
                  </div>
                  {selectedIncident.review_status === "unreviewed" && (
                    <div className="detail-actions">
                      <button
                        className="btn btn-confirm"
                        onClick={() => handleReview("confirmed")}
                        disabled={reviewBusy}
                      >
                        Confirm violent
                      </button>
                      <button
                        className="btn btn-dismiss"
                        onClick={() => handleReview("dismissed")}
                        disabled={reviewBusy}
                      >
                        False positive
                      </button>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div style={{ padding: "40px", textAlign: "center", color: "var(--ink-dim)" }}>
                Select an incident to view details.
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

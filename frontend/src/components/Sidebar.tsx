"use client";

import React from "react";
import { IconShieldCheck } from "@tabler/icons-react";
import { useSafeSpace } from "./SafeSpaceProvider";
import { usePathname } from "next/navigation";

export function Sidebar() {
  const { cameras, selectedCameraId, setSelectedCameraId } = useSafeSpace();
  const pathname = usePathname();



  const activeCount = cameras.filter(c => c.camera_receiving).length;

  return (
    <aside className="sidebar">
      <div className="brand" style={{ paddingBottom: "12px" }}>
        <IconShieldCheck size={34} color="var(--action)" />
        <span className="brand-name">SafeSpace</span>
      </div>

      <div>
        <div className="side-label" style={{ marginBottom: "8px" }}>
          Cameras — {activeCount} active
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {cameras.map((c) => {
            const isActive = c.camera_receiving;
            const isSelected = selectedCameraId === c.camera_id;
            return (
              <div
                key={c.camera_id}
                className={`cam-item ${!isActive ? "offline" : ""}`}
                style={isSelected ? { background: "var(--bg-raised)", borderColor: "var(--action)" } : {}}
                onClick={() => setSelectedCameraId(c.camera_id)}
              >
                <span className="cam-dot"></span>
                <div>
                  <div className="cam-item-name">{c.label || c.camera_id}</div>
                  <div className="cam-item-sub">
                    {c.camera_id} · {isActive ? "live" : "offline"}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {pathname !== "/" && (
        <div className="hint-text" style={{ fontSize: "11px", textAlign: "center", marginTop: "auto", paddingBottom: "16px" }}>
          Camera setup happens on the Monitor tab.
        </div>
      )}
    </aside>
  );
}

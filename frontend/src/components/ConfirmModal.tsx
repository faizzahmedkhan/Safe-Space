import React from "react";

type Props = {
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel: () => void;
};

export function ConfirmModal({
  title,
  message,
  confirmText = "Confirm",
  cancelText = "Cancel",
  onConfirm,
  onCancel,
}: Props) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ width: "320px", gap: "24px" }}>
        <div className="modal-header" style={{ marginBottom: "-12px" }}>
          <div className="modal-title">{title}</div>
        </div>
        <div style={{ color: "var(--ink-dim)", fontSize: "14px" }}>{message}</div>
        <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
          <button className="btn btn-ghost" onClick={onCancel}>
            {cancelText}
          </button>
          <button className="btn btn-action" onClick={onConfirm} style={{ background: "var(--alert)", color: "#2A0D08" }}>
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import {
  StatusResponse,
  AlertItem,
  LogItem,
  fetchCameras,
  fetchAlerts,
  fetchLogs,
} from "@/lib/api";

type SafeSpaceContextType = {
  cameras: StatusResponse[];
  alerts: AlertItem[];
  logs: LogItem[];
  backendOnline: boolean;
  selectedCameraId: string | null;
  setSelectedCameraId: (id: string | null) => void;
  refresh: () => Promise<void>;
};

const SafeSpaceContext = createContext<SafeSpaceContextType | null>(null);

export function SafeSpaceProvider({ children }: { children: ReactNode }) {
  const [cameras, setCameras] = useState<StatusResponse[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [backendOnline, setBackendOnline] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [c, a, l] = await Promise.all([
        fetchCameras(),
        fetchAlerts(),
        fetchLogs(),
      ]);
      setCameras(c.cameras);
      setAlerts(a.recent);
      setLogs(l.recent);
      setBackendOnline(true);

      // Auto-select the first camera if none is selected
      if (!selectedCameraId && c.cameras.length > 0) {
        setSelectedCameraId(c.cameras[0].camera_id);
      }
    } catch {
      setBackendOnline(false);
    }
  }, [selectedCameraId]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <SafeSpaceContext.Provider
      value={{
        cameras,
        alerts,
        logs,
        backendOnline,
        selectedCameraId,
        setSelectedCameraId,
        refresh,
      }}
    >
      {children}
    </SafeSpaceContext.Provider>
  );
}

export function useSafeSpace() {
  const context = useContext(SafeSpaceContext);
  if (!context) {
    throw new Error("useSafeSpace must be used within a SafeSpaceProvider");
  }
  return context;
}

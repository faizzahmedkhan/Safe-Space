"use client";

import React, { useMemo, useState } from "react";
import { Topbar } from "@/components/Topbar";
import { useSafeSpace } from "@/components/SafeSpaceProvider";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip);

const alertColor = "#C1503D";
const watchColor = "#C89550";
const inkDim = "#8B93A1";
const border = "#272E37";

export default function DashboardPage() {
  const { alerts, cameras } = useSafeSpace();
  const [timespan, setTimespan] = useState<number>(14);

  const {
    last7DaysCount,
    spikeCount,
    sustainedCount,
    avgConfidence,
    alertsByCamera,
    alertsPerDay,
  } = useMemo(() => {
    const now = new Date();
    const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const historicDaysAgo = new Date(now.getTime() - timespan * 24 * 60 * 60 * 1000);

    let l7Count = 0;
    let sCount = 0;
    let susCount = 0;
    let confSum = 0;

    const byCam: Record<string, { count: number; confSum: number; lastAlert: Date }> = {};
    const perDayMap: Record<string, number> = {};

    // Initialize historic days map
    for (let i = timespan - 1; i >= 0; i--) {
      const d = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      perDayMap[d.toISOString().split("T")[0]] = 0;
    }

    alerts.forEach((a) => {
      // Backend timestamp format: "YYYY-MM-DD_HH-MM-SS"
      const dateParts = a.timestamp.split("_")[0].split("-");
      const timeParts = a.timestamp.split("_")[1].split("-");
      const ts = new Date(
        parseInt(dateParts[0]),
        parseInt(dateParts[1]) - 1,
        parseInt(dateParts[2]),
        parseInt(timeParts[0]),
        parseInt(timeParts[1]),
        parseInt(timeParts[2])
      );

      const isL7 = ts >= sevenDaysAgo;
      const isHistoric = ts >= historicDaysAgo;

      if (isL7) {
        l7Count++;
        confSum += a.confidence;
        if (a.trigger === "spike") sCount++;
        else susCount++;

        const cid = a.camera_id || "Unknown";
        if (!byCam[cid]) byCam[cid] = { count: 0, confSum: 0, lastAlert: ts };
        byCam[cid].count++;
        byCam[cid].confSum += a.confidence;
        if (ts > byCam[cid].lastAlert) byCam[cid].lastAlert = ts;
      }

      if (isHistoric) {
        const dayKey = ts.toISOString().split("T")[0];
        if (perDayMap[dayKey] !== undefined) {
          perDayMap[dayKey]++;
        }
      }
    });

    return {
      last7DaysCount: l7Count,
      spikeCount: sCount,
      sustainedCount: susCount,
      avgConfidence: l7Count > 0 ? Math.round(confSum / l7Count) : 0,
      alertsByCamera: Object.entries(byCam).map(([id, data]) => ({
        id,
        count: data.count,
        avgConf: Math.round(data.confSum / data.count),
        lastAlert: data.lastAlert,
        name: cameras.find((c) => c.camera_id === id)?.label || id,
        statusDot: cameras.find((c) => c.camera_id === id)?.camera_receiving ? "var(--calm)" : "var(--alert)",
      })),
      alertsPerDay: Object.entries(perDayMap).map(([date, count]) => ({ date, count })),
    };
  }, [alerts, cameras, timespan]);

  const timeAgo = (date: Date) => {
    const diff = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  const lineChartData = {
    labels: alertsPerDay.map((d) => {
      const parts = d.date.split("-");
      return `${parts[1]}/${parts[2]}`;
    }),
    datasets: [
      {
        data: alertsPerDay.map((d) => d.count),
        borderColor: alertColor,
        backgroundColor: "rgba(193,80,61,0.1)",
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  };

  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: inkDim, font: { family: "IBM Plex Mono", size: 10 }, maxRotation: 0 } },
      y: { beginAtZero: true, grid: { color: border }, ticks: { color: inkDim, font: { family: "IBM Plex Mono", size: 10 }, stepSize: 1 } },
    },
  };

  const barChartData = {
    labels: ["Spike", "Sustained"],
    datasets: [
      {
        data: [spikeCount, sustainedCount],
        backgroundColor: [alertColor, watchColor],
        borderRadius: 4,
        maxBarThickness: 40,
      },
    ],
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: "y" as const,
    plugins: { legend: { display: false } },
    scales: {
      x: { beginAtZero: true, grid: { color: border }, ticks: { color: inkDim, font: { family: "IBM Plex Mono", size: 10 } } },
      y: { grid: { display: false }, ticks: { color: inkDim, font: { family: "IBM Plex Sans", size: 12 } } },
    },
  };

  return (
    <main className="main">
      <Topbar />
      <section className="dash active">
        <div className="kpi-row">
          <div className="kpi">
            <div className="kpi-label">Alerts, last 7 days</div>
            <div className="kpi-value">{last7DaysCount}</div>
            <div className="kpi-delta down">—</div>
          </div>
          <div className="kpi">
            <div className="kpi-label">Avg confidence at alert</div>
            <div className="kpi-value">{avgConfidence}%</div>
            <div className="kpi-delta">—</div>
          </div>
          <div className="kpi">
            <div className="kpi-label">Spike-triggered</div>
            <div className="kpi-value" style={{ color: "var(--alert)" }}>
              {spikeCount}
            </div>
            <div className="kpi-delta">of {last7DaysCount} alerts</div>
          </div>
          <div className="kpi">
            <div className="kpi-label">Sustained-triggered</div>
            <div className="kpi-value" style={{ color: "var(--watch)" }}>
              {sustainedCount}
            </div>
            <div className="kpi-delta">of {last7DaysCount} alerts</div>
          </div>
        </div>

        <div className="chart-row">
          <div className="chart-card" style={{ marginBottom: 0 }}>
            <div className="chart-card-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>Alerts per day</span>
              <select 
                value={timespan} 
                onChange={(e) => setTimespan(Number(e.target.value))}
                style={{
                  background: "var(--bg-base)",
                  color: "var(--ink)",
                  border: "1px solid var(--border)",
                  borderRadius: "4px",
                  padding: "2px 8px",
                  fontSize: "12px",
                  fontFamily: "var(--font-sans)",
                }}
              >
                <option value={7}>Last 7 days</option>
                <option value={14}>Last 14 days</option>
                <option value={30}>Last 1 month</option>
                <option value={90}>Last 3 months</option>
                <option value={180}>Last 6 months</option>
                <option value={365}>Last 1 year</option>
              </select>
            </div>
            <div className="chart-card-sub" style={{ marginTop: "4px" }}>All cameras</div>
            <div style={{ position: "relative", height: "220px" }}>
              <Line data={lineChartData} options={lineChartOptions as any} />
            </div>
          </div>
          <div className="chart-card" style={{ marginBottom: 0 }}>
            <div className="chart-card-title">Trigger type</div>
            <div className="chart-card-sub">Spike vs sustained, this week</div>
            <div className="legend">
              <span>
                <i style={{ background: "var(--alert)" }}></i>Spike
              </span>
              <span>
                <i style={{ background: "var(--watch)" }}></i>Sustained
              </span>
            </div>
            <div style={{ position: "relative", height: "180px" }}>
              <Bar data={barChartData} options={barChartOptions as any} />
            </div>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-card-title">Alerts by camera</div>
          <div className="chart-card-sub">Last 7 days</div>
          {alertsByCamera.length > 0 ? (
            <table className="cam-table">
              <thead>
                <tr>
                  <th>Camera</th>
                  <th>Alerts</th>
                  <th style={{ textAlign: "right" }}>Avg confidence</th>
                  <th style={{ textAlign: "right" }}>Last alert</th>
                </tr>
              </thead>
              <tbody>
                {alertsByCamera.map((cam) => (
                  <tr key={cam.id}>
                    <td>
                      <span className="row-dot" style={{ background: cam.statusDot }}></span>
                      {cam.name}
                    </td>
                    <td className="num">{cam.count}</td>
                    <td className="num">{cam.avgConf}%</td>
                    <td className="num">{timeAgo(cam.lastAlert)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ color: "var(--ink-dim)", fontSize: "12px", padding: "10px 0" }}>
              No alerts in the last 7 days.
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

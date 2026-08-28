"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function Topbar() {
  const pathname = usePathname();

  return (
    <div className="topbar">
      <div>
        <div className="eyebrow">School safety monitor</div>
        <h1>Overview</h1>
      </div>
      <div className="tabs">
        <Link href="/" className={`tab ${pathname === "/" ? "active" : ""}`}>
          Monitor
        </Link>
        <Link href="/dashboard" className={`tab ${pathname === "/dashboard" ? "active" : ""}`}>
          Dashboard
        </Link>
        <Link href="/incidents" className={`tab ${pathname === "/incidents" ? "active" : ""}`}>
          Incidents
        </Link>
      </div>
    </div>
  );
}

import type { Metadata } from "next";
import { IBM_Plex_Sans_Condensed, IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { SafeSpaceProvider } from "@/components/SafeSpaceProvider";
import { Sidebar } from "@/components/Sidebar";

const plexCond = IBM_Plex_Sans_Condensed({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-cond",
  display: "swap",
});

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SafeSpace",
  description: "School safety monitor",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${plexCond.variable} ${plexSans.variable} ${plexMono.variable}`}>
        <SafeSpaceProvider>
          <div className="app">
            <Sidebar />
            {children}
          </div>
        </SafeSpaceProvider>
      </body>
    </html>
  );
}

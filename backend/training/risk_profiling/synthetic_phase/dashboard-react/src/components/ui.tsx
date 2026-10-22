import { motion, animate } from "framer-motion";
import { AlertTriangle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import type { Tier } from "../types";

export const TIER_COLOURS: Record<Tier, string> = {
  "High Performer": "#10b981",
  Progressing: "#f59e0b",
  "Needs Urgent Assistance": "#ef4444",
  "Not enough data": "#94a3b8",
};
export const LEVEL_COLOURS: Record<string, string> = {
  "No concern": "#10b981",
  "Worth monitoring": "#facc15",
  "Closer look": "#fb923c",
  "Closer look (priority)": "#ef4444",
  "Not enough data": "#94a3b8",
};
export const SKILL_LEVEL_COLOURS: Record<string, string> = {
  strength: "#10b981",
  as_expected: "#84cc16",
  needs_practice: "#facc15",
  needs_extra_help: "#ef4444",
  not_assessed: "#cbd5e1",
};
export const SKILL_LEVEL_NAMES: Record<string, string> = {
  strength: "Strength",
  as_expected: "As expected",
  needs_practice: "Needs practice",
  needs_extra_help: "Needs extra help",
  not_assessed: "Not assessed",
};
export const PALETTE = ["#6366f1", "#ec4899", "#14b8a6", "#f59e0b", "#8b5cf6", "#06b6d4"];
export const ACTIVITY_EMOJI: Record<string, string> = {
  counting: "🔢", comparison: "⚖️", ordering: "📶", place_value: "🧱",
  addition: "➕", subtraction: "➖", multiplication: "✖️", division: "➗",
};

const NAMES: Record<string, string> = {
  svm: "SVM", xgboost: "XGBoost", accuracy_rule: "Accuracy rule", random_forest: "Random Forest",
  logistic: "Logistic regression", formula: "Evidence formula", stacking: "Stacking",
};
export const pretty = (s: string) => NAMES[s] ?? s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/** Red -> yellow -> green for a percentile (0-100). */
export function percentileColour(p: number | null): string {
  if (p === null || p === undefined) return "#e2e8f0";
  const stops = [
    [0, [239, 68, 68]], [25, [251, 146, 60]], [50, [250, 204, 21]], [75, [132, 204, 22]], [100, [16, 185, 129]],
  ] as const;
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i];
    const [p0, c0] = stops[i - 1];
    if (p <= p1) {
      const t = (p - p0) / (p1 - p0);
      const c = c0.map((v, k) => Math.round(v + t * (c1[k] - v)));
      return `rgb(${c.join(",")})`;
    }
  }
  return "rgb(16,185,129)";
}

export function Counter({ value, decimals = 0 }: { value: number; decimals?: number }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const controls = animate(0, value, { duration: 0.9, onUpdate: (v) => setShown(v) });
    return () => controls.stop();
  }, [value]);
  return <>{shown.toFixed(decimals)}</>;
}

export function Kpi({ label, value, icon, from, to, decimals = 0, suffix = "" }: {
  label: string; value: number; icon: ReactNode; from: string; to: string; decimals?: number; suffix?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      className="rounded-3xl p-5 text-white shadow-lg"
      style={{ background: `linear-gradient(135deg, ${from}, ${to})` }}
    >
      <div className="flex items-center justify-between opacity-90">{icon}</div>
      <div className="mt-2 text-4xl font-extrabold"><Counter value={value} decimals={decimals} />{suffix}</div>
      <div className="text-sm font-semibold opacity-90">{label}</div>
    </motion.div>
  );
}

export function Badge({ text, colour }: { text: string; colour: string }) {
  return <span className="chip text-white shadow-sm" style={{ background: colour }}>{text}</span>;
}

export function Avatar({ id, colour, size = 44 }: { id: string; colour: string; size?: number }) {
  return (
    <div
      className="flex shrink-0 items-center justify-center rounded-2xl font-extrabold text-white shadow"
      style={{ width: size, height: size, background: colour, fontSize: size / 3.2 }}
    >
      {id.slice(-3)}
    </div>
  );
}

export function Disclaimer({ text }: { text: string }) {
  return (
    <div className="mb-5 flex items-center gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
      <AlertTriangle size={18} className="shrink-0" />
      <span><b>Synthetic demo children.</b> {text} Not yet validated on Sri Lankan children.</span>
    </div>
  );
}

export function Header({ title, subtitle, children }: { title: string; subtitle?: string; children?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-3xl font-extrabold text-slate-800">{title}</h1>
        {subtitle && <p className="text-sm font-semibold text-slate-500">{subtitle}</p>}
      </div>
      <div className="flex flex-wrap items-center gap-3">{children}</div>
    </div>
  );
}

export function Card({ title, icon, children, className = "" }: {
  title?: string; icon?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`card ${className}`}>
      {title && <div className="card-title">{icon}{title}</div>}
      {children}
    </motion.div>
  );
}

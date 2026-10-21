import { Award, Crosshair, FlaskConical, GraduationCap, HeartPulse, Layers, Target } from "lucide-react";
import { Bar, BarChart, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Disclaimer, Header, Kpi, TIER_COLOURS, pretty } from "../components/ui";
import type { Payload } from "../types";

const TIERS = ["High Performer", "Progressing", "Needs Urgent Assistance"] as const;

export default function Evidence({ payload }: { payload: Payload }) {
  const e = payload.evidence;
  const mastery = e.mastery_validation.map((m) => ({ ...m, model: pretty(m.model) + (m.model === e.mastery_selected ? " ★" : "") }));
  const ablation = ["accuracy_only", "performance", "multimodal", "historical"].map((set) => {
    const row: Record<string, number | string> = { set: pretty(set) };
    for (const a of e.ablation.filter((x) => x.set === set)) row[a.model.toUpperCase()] = a.macro_f1;
    return row;
  });
  const risk = e.risk_models.map((m) => ({ ...m, model: pretty(m.model) + (m.model === "formula" ? " ★" : "") }));
  const grades = Object.entries(e.risk_test.by_grade).map(([g, auc]) => ({ grade: `Grade ${g}`, auc }));
  const cmMax = Math.max(...e.confusion_matrix.flat());

  return (
    <div>
      <Header title="📊 Model evidence" subtitle="How well the component works on held-out synthetic children (and 14 real children)" />
      <Disclaimer text="All numbers come from the component's result files; ★ marks the selected model." />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Kpi label="mastery macro-F1 (test)" value={e.mastery_test.macro_f1} decimals={2} icon={<Award />} from="#6366f1" to="#22d3ee" />
        <Kpi label="children needing help caught" value={e.mastery_test.recall_per_class["Needs Urgent Assistance"] * 100} suffix="%" icon={<HeartPulse />} from="#ef4444" to="#f472b6" />
        <Kpi label="risk AUC (synthetic test)" value={e.risk_test.auc.auc} decimals={2} icon={<Target />} from="#8b5cf6" to="#ec4899" />
        <Kpi label="risk AUC (14 real children)" value={e.risk_test.real_auc.auc} decimals={2} icon={<Crosshair />} from="#10b981" to="#84cc16" />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Card title="Mastery models (validation)" icon={<FlaskConical size={18} className="text-indigo-500" />}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mastery}>
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend iconType="circle" />
              <Bar isAnimationActive={false} dataKey="macro_f1" name="Macro-F1" fill="#6366f1" radius={[6, 6, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="urgent_recall" name="Needs-urgent recall" fill="#f97316" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Selected SVM: confusion matrix (test)" icon={<Layers size={18} className="text-fuchsia-500" />}>
          <div className="grid grid-cols-4 gap-2 text-center text-sm">
            <div />
            {TIERS.map((t) => <div key={t} className="text-xs font-extrabold" style={{ color: TIER_COLOURS[t] }}>pred: {t}</div>)}
            {e.confusion_matrix.map((row, i) => (
              <div key={i} className="contents">
                <div className="flex items-center justify-end pr-2 text-xs font-extrabold" style={{ color: TIER_COLOURS[TIERS[i]] }}>true: {TIERS[i]}</div>
                {row.map((v, j) => (
                  <div key={j} className="flex h-16 items-center justify-center rounded-2xl text-xl font-extrabold"
                    style={{ background: i === j ? `rgba(16,185,129,${0.15 + 0.75 * v / cmMax})` : `rgba(239,68,68,${0.08 + 0.6 * v / cmMax})`, color: v / cmMax > 0.5 ? "white" : "#334155" }}>
                    {v}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-slate-500">Green diagonal = correct. Needs-urgent recall {Math.round(e.mastery_test.recall_per_class["Needs Urgent Assistance"] * 100)}% (95% CI {e.mastery_test.needs_urgent_recall_ci95.map((v) => Math.round(v * 100)).join("–")}%), accuracy-rule baseline {Math.round(e.mastery_baseline_test.recall_per_class["Needs Urgent Assistance"] * 100)}%.</p>
        </Card>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Card title="Ablation: adding feature groups" icon={<Layers size={18} className="text-cyan-500" />}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={ablation}>
              <XAxis dataKey="set" />
              <YAxis domain={[0.35, 0.6]} />
              <Tooltip />
              <Legend iconType="circle" />
              <Line isAnimationActive={false} dataKey="SVM" stroke="#ec4899" strokeWidth={3} dot={{ r: 6 }} />
              <Line isAnimationActive={false} dataKey="XGBOOST" name="XGBoost" stroke="#6366f1" strokeWidth={3} dot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Risk models: synthetic validation vs real children (AUC)" icon={<Target size={18} className="text-violet-500" />}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={risk}>
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis domain={[0.5, 1]} />
              <ReferenceLine y={0.5} stroke="#94a3b8" />
              <Tooltip />
              <Legend iconType="circle" />
              <Bar isAnimationActive={false} dataKey="validation_auc" name="Synthetic validation" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
              <Bar isAnimationActive={false} dataKey="real_auc" name="Real Decarli (50)" fill="#14b8a6" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="Risk screening by grade (final test AUC)" icon={<GraduationCap size={18} className="text-amber-500" />} className="mt-5">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={grades}>
            <XAxis dataKey="grade" />
            <YAxis domain={[0, 1]} />
            <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "chance", fontSize: 11 }} />
            <Tooltip formatter={(v: number) => v.toFixed(2)} />
            <Bar isAnimationActive={false} dataKey="auc" name="AUC" radius={[8, 8, 0, 0]} fill="#f59e0b" />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-500">Grade 3 is weak (8 DD children) and is a priority for the pilot study.</p>
      </Card>
    </div>
  );
}

import { motion } from "framer-motion";
import { Activity, Bot, Brain, Copy, Lightbulb, Radar as RadarIcon, Search } from "lucide-react";
import { useMemo, useState } from "react";
import {
  Bar, BarChart, Cell, ComposedChart, Legend, Line, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  RadialBar, RadialBarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  ACTIVITY_EMOJI, Avatar, Badge, Card, Disclaimer, Header, LEVEL_COLOURS, SKILL_LEVEL_COLOURS, SKILL_LEVEL_NAMES,
  TIER_COLOURS, pretty,
} from "../components/ui";
import type { Child, Payload } from "../types";

export default function ChildView({ payload, child, setChild }: {
  payload: Payload; child: Child; setChild: (c: Child) => void;
}) {
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"teacher" | "parent">("teacher");
  const classmates = useMemo(
    () => payload.children.filter((c) => c.class_name === child.class_name).sort((a, b) => a.id.localeCompare(b.id)),
    [payload, child.class_name],
  );
  const found = query.length >= 2 ? payload.children.filter((c) => c.id.toLowerCase().includes(query.toLowerCase())).slice(0, 6) : [];

  const radar = payload.meta.skills.map((s) => ({ skill: s, child: child.skills[s]?.percentile ?? 0, typical: 50,
    assessed: child.skills[s]?.percentile !== null }));
  const contrib = payload.meta.skills
    .filter((s) => child.skills[s]?.contribution !== null && child.skills[s]?.contribution !== undefined)
    .map((s) => ({ skill: s, value: child.skills[s].contribution as number }))
    .sort((a, b) => b.value - a.value);
  const reasons = Object.entries(child.reasons ?? {}).map(([k, v]) => ({ feature: k, value: v })).sort((a, b) => b.value - a.value);
  const sessions = child.sessions.map((s) => ({ session: `Session ${s.session}`, accuracy: Math.round(s.accuracy * 100), time: s.median_rt }));
  const urgent = Math.round((child.p_urgent ?? 0) * 100);
  const report = tab === "teacher" ? child.teacher_report : child.parent_report;

  return (
    <div>
      <Header title="👧 Child view" subtitle={child.class_name}>
        <div className="relative">
          <Search size={16} className="absolute left-3 top-2.5 text-slate-400" />
          <input className="pl-9" placeholder="Search child ID…" value={query} onChange={(e) => setQuery(e.target.value)} />
          {found.length > 0 && (
            <div className="absolute z-20 mt-1 w-full rounded-xl border bg-white shadow-lg">
              {found.map((c) => (
                <button key={c.id} className="block w-full px-3 py-2 text-left text-sm hover:bg-indigo-50"
                  onClick={() => { setChild(c); setQuery(""); }}>{c.id} · {c.class_name}</button>
              ))}
            </div>
          )}
        </div>
        <select value={child.id} onChange={(e) => setChild(classmates.find((c) => c.id === e.target.value)!)}>
          {classmates.map((c) => <option key={c.id} value={c.id}>{c.id} · {c.tier}</option>)}
        </select>
      </Header>
      <Disclaimer text="This profile supports the teacher; it is not a diagnosis." />

      <motion.div key={child.id} initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
        className="flex flex-wrap items-center gap-5 rounded-3xl p-6 text-white shadow-lg"
        style={{ background: `linear-gradient(135deg, ${TIER_COLOURS[child.tier]}, #6366f1)` }}>
        <Avatar id={child.id} colour="rgba(255,255,255,.25)" size={72} />
        <div className="flex-1">
          <div className="text-3xl font-extrabold">{child.id}</div>
          <div className="font-semibold opacity-90">Grade {child.grade} · {Math.floor(child.age_months / 12)} y {child.age_months % 12} m · {child.n_sessions} sessions · {child.n_questions} questions</div>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge text={child.tier} colour="rgba(255,255,255,.25)" />
            <Badge text={`Risk: ${child.level_name}`} colour={LEVEL_COLOURS[child.level_name]} />
            {child.profile && <Badge text={child.profile} colour="#7c3aed" />}
          </div>
        </div>
        {child.p_urgent !== null && (
          <div className="relative h-32 w-32">
            <ResponsiveContainer>
              <RadialBarChart innerRadius="75%" outerRadius="100%" data={[{ v: urgent }]} startAngle={90} endAngle={-270}>
                <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                <RadialBar isAnimationActive={false} dataKey="v" cornerRadius={10} fill="#fff" background={{ fill: "rgba(255,255,255,.25)" }} />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="text-2xl font-extrabold">{urgent}%</div>
              <div className="text-[10px] font-bold uppercase opacity-90">needs help</div>
            </div>
          </div>
        )}
      </motion.div>

      {child.level === "insufficient" ? (
        <Card className="mt-5"><p className="font-semibold">Not enough data yet. Please complete another session with the robot.</p></Card>
      ) : (
        <>
          <div className="mt-5 grid gap-5 lg:grid-cols-2">
            <Card title="Skills vs same-age peers (percentile)" icon={<RadarIcon size={18} className="text-indigo-500" />}>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radar}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="skill" tick={{ fontSize: 12, fontWeight: 700 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} />
                  <Radar isAnimationActive={false} name="Typical (50)" dataKey="typical" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.1} strokeDasharray="4 4" />
                  <Radar isAnimationActive={false} name={child.id} dataKey="child" stroke="#6366f1" fill="#6366f1" fillOpacity={0.45} />
                  <Legend iconType="circle" />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-2">
                {payload.meta.skills.map((s) => {
                  const lv = child.skills[s]?.level ?? "not_assessed";
                  return <Badge key={s} text={`${s}: ${SKILL_LEVEL_NAMES[lv] ?? lv}`} colour={SKILL_LEVEL_COLOURS[lv] ?? "#94a3b8"} />;
                })}
              </div>
            </Card>
            <Card title="Session by session" icon={<Activity size={18} className="text-emerald-500" />}>
              <ResponsiveContainer width="100%" height={340}>
                <ComposedChart data={sessions}>
                  <XAxis dataKey="session" />
                  <YAxis yAxisId="l" domain={[0, 100]} unit="%" />
                  <YAxis yAxisId="r" orientation="right" unit="s" />
                  <Tooltip />
                  <Legend iconType="circle" />
                  <Bar isAnimationActive={false} yAxisId="r" dataKey="time" name="Median time (s)" fill="#f9a8d4" radius={[8, 8, 0, 0]} barSize={50} />
                  <Line isAnimationActive={false} yAxisId="l" dataKey="accuracy" name="Accuracy %" stroke="#10b981" strokeWidth={4} dot={{ r: 8 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div className="mt-5 grid gap-5 lg:grid-cols-2">
            <Card title="Risk score: what each skill adds (exact)" icon={<Lightbulb size={18} className="text-amber-500" />}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={contrib} layout="vertical" margin={{ left: 40 }}>
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="skill" width={140} />
                  <ReferenceLine x={0} stroke="#64748b" />
                  <Tooltip formatter={(v: number) => v.toFixed(2)} />
                  <Bar isAnimationActive={false} dataKey="value" radius={6}>{contrib.map((d) => <Cell key={d.skill} fill={d.value > 0 ? "#ef4444" : "#10b981"} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-slate-500">Red raises the risk score, green lowers it.{child.not_assessed.length > 0 && ` Not assessed at this grade: ${child.not_assessed.join(", ")}.`}</p>
            </Card>
            <Card title="Mastery: why 'needs urgent help'? (SHAP)" icon={<Brain size={18} className="text-fuchsia-500" />}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={reasons} layout="vertical" margin={{ left: 60 }}>
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="feature" width={170} tick={{ fontSize: 11 }} />
                  <ReferenceLine x={0} stroke="#64748b" />
                  <Tooltip formatter={(v: number) => v.toFixed(3)} />
                  <Bar isAnimationActive={false} dataKey="value" radius={6}>{reasons.map((d) => <Cell key={d.feature} fill={d.value > 0 ? "#f97316" : "#6366f1"} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-slate-500">Orange pushes toward "needs urgent help", blue away. Associations, not causes.</p>
            </Card>
          </div>
        </>
      )}

      <div className="mt-5 grid gap-5 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex gap-2">
              {(["teacher", "parent"] as const).map((t) => (
                <button key={t} onClick={() => setTab(t)}
                  className={`chip px-4 py-2 text-sm ${tab === t ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-600"}`}>
                  {t === "teacher" ? "👩‍🏫 Teacher report" : "👪 Parent report"}
                </button>
              ))}
            </div>
            <button className="chip bg-slate-100 text-slate-600" onClick={() => navigator.clipboard.writeText(report.join("\n"))}>
              <Copy size={14} /> Copy
            </button>
          </div>
          <motion.div key={tab} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-2 rounded-2xl bg-indigo-50/60 p-4 text-sm leading-relaxed">
            {report.map((line, i) => <p key={i} className={i === 0 ? "font-extrabold" : ""}>{line}</p>)}
          </motion.div>
        </Card>
        <Card title="Suggested robot activities" icon={<Bot size={18} className="text-cyan-500" />}>
          {child.practise.length === 0 ? (
            <p className="text-sm text-slate-500">No extra practice suggested. Keep playing the normal game.</p>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              {child.practise.map((a, i) => (
                <motion.div key={a} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: i * 0.08 }}
                  className="rounded-2xl p-4 text-center font-extrabold text-white shadow"
                  style={{ background: `linear-gradient(135deg, ${["#6366f1", "#ec4899", "#14b8a6", "#f59e0b"][i % 4]}, #a78bfa)` }}>
                  <div className="text-3xl">{ACTIVITY_EMOJI[a] ?? "🎲"}</div>{pretty(a)}
                </motion.div>
              ))}
            </div>
          )}
          <p className="mt-3 text-xs text-slate-500">From Component 2's activities offered at this grade.</p>
        </Card>
      </div>
    </div>
  );
}

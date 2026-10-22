import { motion } from "framer-motion";
import { AlertOctagon, Grid3x3, PieChart as PieIcon, Search, Sparkles, Star, TrendingUp, Users } from "lucide-react";
import { useMemo } from "react";
import { Bar, BarChart, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  ACTIVITY_EMOJI, Avatar, Badge, Card, Disclaimer, Header, Kpi, LEVEL_COLOURS, SKILL_LEVEL_COLOURS,
  SKILL_LEVEL_NAMES, TIER_COLOURS, percentileColour, pretty,
} from "../components/ui";
import type { Child, Payload } from "../types";

const TIERS = ["High Performer", "Progressing", "Needs Urgent Assistance", "Not enough data"] as const;
const LEVELS = Object.keys(LEVEL_COLOURS);

export default function Overview({ payload, cls, setCls, openChild }: {
  payload: Payload; cls: string; setCls: (c: string) => void; openChild: (c: Child) => void;
}) {
  const kids = useMemo(
    () => payload.children.filter((c) => cls === "All classes" || c.class_name === cls),
    [payload, cls],
  );
  const count = (f: (c: Child) => boolean) => kids.filter(f).length;

  const tierData = TIERS.map((t) => ({ name: t, value: count((c) => c.tier === t) })).filter((d) => d.value > 0);
  const groupKey = (c: Child) => (cls === "All classes" ? c.class_name.replace(" · Demo class ", " ") : `Grade ${c.grade}`);
  const levelData = Object.values(
    kids.reduce<Record<string, Record<string, number | string>>>((acc, c) => {
      const k = groupKey(c);
      acc[k] ??= { group: k };
      acc[k][c.level_name] = ((acc[k][c.level_name] as number) ?? 0) + 1;
      return acc;
    }, {}),
  );
  const skillData = payload.meta.skills.map((s) => {
    const assessed = kids.filter((c) => c.skills[s]?.level && c.skills[s].level !== "not_assessed");
    const row: Record<string, number | string> = { skill: s };
    for (const lv of ["strength", "as_expected", "needs_practice", "needs_extra_help"]) {
      row[SKILL_LEVEL_NAMES[lv]] = assessed.length ? Math.round((100 * assessed.filter((c) => c.skills[s].level === lv).length) / assessed.length) : 0;
    }
    return row;
  });
  const warnings = kids
    .filter((c) => c.tier === "Needs Urgent Assistance" || c.level === "moderate" || c.level === "high")
    .sort((a, b) => (b.p_urgent ?? 0) - (a.p_urgent ?? 0));

  return (
    <div>
      <Header title="🏫 Class overview" subtitle={cls === "All classes" ? "All demo classes" : cls}>
        <select value={cls} onChange={(e) => setCls(e.target.value)}>
          <option>All classes</option>
          {payload.meta.classes.map((c) => <option key={c}>{c}</option>)}
        </select>
      </Header>
      <Disclaimer text="Results are learning-performance insights and screening support, not a diagnosis." />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        <Kpi label="children" value={kids.length} icon={<Users />} from="#6366f1" to="#22d3ee" />
        <Kpi label="High Performers" value={count((c) => c.tier === "High Performer")} icon={<Star />} from="#10b981" to="#84cc16" />
        <Kpi label="Progressing" value={count((c) => c.tier === "Progressing")} icon={<TrendingUp />} from="#f59e0b" to="#facc15" />
        <Kpi label="Need urgent assistance" value={count((c) => c.tier === "Needs Urgent Assistance")} icon={<Sparkles />} from="#ef4444" to="#f472b6" />
        <Kpi label="Closer look (risk screen)" value={count((c) => c.level === "moderate" || c.level === "high")} icon={<Search />} from="#8b5cf6" to="#ec4899" />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-5">
        <Card title="Mastery tiers" icon={<PieIcon size={18} className="text-indigo-500" />} className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie isAnimationActive={false} data={tierData} dataKey="value" nameKey="name" innerRadius={70} outerRadius={110} paddingAngle={3}>
                {tierData.map((d) => <Cell key={d.name} fill={TIER_COLOURS[d.name as keyof typeof TIER_COLOURS]} />)}
              </Pie>
              <Tooltip />
              <Legend iconType="circle" />
            </PieChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Dyscalculia-risk screening levels" icon={<AlertOctagon size={18} className="text-rose-500" />} className="lg:col-span-3">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={levelData}>
              <XAxis dataKey="group" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend iconType="circle" />
              {LEVELS.map((l) => <Bar isAnimationActive={false} key={l} dataKey={l} stackId="a" fill={LEVEL_COLOURS[l]} radius={l === "Not enough data" ? [6, 6, 0, 0] : 0} />)}
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="Where does the class need support? (% of children assessed)" icon={<Sparkles size={18} className="text-amber-500" />} className="mt-5">
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={skillData} layout="vertical" margin={{ left: 40 }}>
            <XAxis type="number" domain={[0, 100]} allowDataOverflow ticks={[0, 25, 50, 75, 100]} unit="%" />
            <YAxis type="category" dataKey="skill" width={140} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Legend iconType="circle" />
            {["strength", "as_expected", "needs_practice", "needs_extra_help"].map((lv) => (
              <Bar isAnimationActive={false} key={lv} dataKey={SKILL_LEVEL_NAMES[lv]} stackId="s" fill={SKILL_LEVEL_COLOURS[lv]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {cls !== "All classes" && (
        <Card title="Skill heat-map: better than …% of same-age peers" icon={<Grid3x3 size={18} className="text-emerald-500" />} className="mt-5">
          <div className="overflow-x-auto">
            <div className="grid gap-1" style={{ gridTemplateColumns: `150px repeat(${kids.length}, minmax(22px, 1fr))` }}>
              {payload.meta.skills.map((s) => (
                <div key={s} className="contents">
                  <div className="pr-2 text-right text-xs font-bold text-slate-500">{s}</div>
                  {kids.map((c) => {
                    const p = c.skills[s]?.percentile ?? null;
                    return (
                      <motion.button
                        whileHover={{ scale: 1.35, zIndex: 10 }}
                        key={c.id + s}
                        onClick={() => openChild(c)}
                        title={`${c.id} · ${s}: ${p === null ? "not assessed" : Math.round(p) + "th percentile"}`}
                        className="h-7 rounded-md"
                        style={{ background: percentileColour(p) }}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
          <p className="mt-2 text-xs text-slate-500">Each column is a child (click to open). Grey = not assessed at this grade.</p>
        </Card>
      )}

      <Card title={`Early warnings (${warnings.length})`} icon={<AlertOctagon size={18} className="text-rose-500" />} className="mt-5">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {warnings.slice(0, 30).map((c) => (
            <motion.button
              key={c.id}
              whileHover={{ y: -3 }}
              onClick={() => openChild(c)}
              className="flex gap-3 rounded-2xl border border-slate-100 bg-gradient-to-br from-white to-rose-50 p-4 text-left shadow-sm"
            >
              <Avatar id={c.id} colour={TIER_COLOURS[c.tier]} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between font-extrabold">
                  <span>{c.id}</span>
                  <span className="text-xs text-slate-400">{c.class_name.replace(" · Demo class ", " · ")}</span>
                </div>
                <div className="mt-1 flex flex-wrap gap-1">
                  <Badge text={c.tier} colour={TIER_COLOURS[c.tier]} />
                  <Badge text={c.level_name} colour={LEVEL_COLOURS[c.level_name]} />
                </div>
                {c.p_urgent !== null && (
                  <div className="mt-2">
                    <div className="h-2 rounded-full bg-slate-100">
                      <div className="h-2 rounded-full bg-gradient-to-r from-amber-400 to-rose-500" style={{ width: `${c.p_urgent * 100}%` }} />
                    </div>
                    <div className="mt-1 text-xs text-slate-500">P(needs urgent help) {Math.round(c.p_urgent * 100)}%</div>
                  </div>
                )}
                <div className="mt-1 text-xs">{c.practise.map((a) => `${ACTIVITY_EMOJI[a] ?? ""} ${pretty(a)}`).join("  ")}</div>
              </div>
            </motion.button>
          ))}
        </div>
      </Card>
    </div>
  );
}

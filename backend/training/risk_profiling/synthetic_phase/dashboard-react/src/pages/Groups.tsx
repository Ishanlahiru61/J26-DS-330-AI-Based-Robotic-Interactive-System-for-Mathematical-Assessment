import { motion } from "framer-motion";
import { BarChart3, Users } from "lucide-react";
import { useMemo } from "react";
import { Bar, BarChart, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ACTIVITY_EMOJI, Avatar, Card, Disclaimer, Header, PALETTE, TIER_COLOURS, pretty } from "../components/ui";
import type { Child, Payload } from "../types";

export default function Groups({ payload, cls, setCls, openChild }: {
  payload: Payload; cls: string; setCls: (c: string) => void; openChild: (c: Child) => void;
}) {
  const current = cls === "All classes" ? payload.meta.classes[0] : cls;
  const flagged = useMemo(() => payload.children.filter((c) => c.class_name === current && c.flagged), [payload, current]);
  const groups = useMemo(() => {
    const g: Record<string, Child[]> = {};
    for (const c of flagged) (g[c.profile ?? "Other"] ??= []).push(c);
    return Object.entries(g).sort((a, b) => b[1].length - a[1].length);
  }, [flagged]);
  const chart = groups.map(([name, kids]) => {
    const row: Record<string, number | string> = { group: name };
    for (const s of payload.meta.skills) {
      const vals = kids.map((k) => k.skills[s]?.percentile).filter((v): v is number => v !== null && v !== undefined);
      row[s] = vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
    }
    return row;
  });

  return (
    <div>
      <Header title="👥 Practice groups" subtitle={`${current} · children flagged at the screening cut-off, grouped by difficulty profile`}>
        <select value={current} onChange={(e) => setCls(e.target.value)}>
          {payload.meta.classes.map((c) => <option key={c}>{c}</option>)}
        </select>
      </Header>
      <Disclaimer text="Groups help the teacher plan small-group practice; they are not diagnostic categories." />

      {flagged.length === 0 ? (
        <Card><p>No flagged children in this class.</p></Card>
      ) : (
        <>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {groups.map(([name, kids], i) => {
              const acts = Array.from(new Set(kids.flatMap((k) => k.practise)));
              return (
                <motion.div key={name} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}
                  className="overflow-hidden rounded-3xl bg-white shadow-sm">
                  <div className="p-5 text-white" style={{ background: `linear-gradient(135deg, ${PALETTE[i % PALETTE.length]}, ${PALETTE[(i + 1) % PALETTE.length]})` }}>
                    <div className="flex items-center gap-2 text-lg font-extrabold"><Users size={18} /> {name}</div>
                    <div className="text-sm font-semibold opacity-90">{kids.length} {kids.length === 1 ? "child" : "children"}</div>
                  </div>
                  <div className="p-5">
                    <div className="flex flex-wrap gap-2">
                      {kids.map((k) => (
                        <button key={k.id} onClick={() => openChild(k)} title={`${k.id} · ${k.tier}`}>
                          <Avatar id={k.id} colour={TIER_COLOURS[k.tier]} size={38} />
                        </button>
                      ))}
                    </div>
                    <div className="mt-4 text-xs font-extrabold uppercase text-slate-400">Practise together</div>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {acts.length ? acts.map((a) => (
                        <span key={a} className="chip bg-indigo-50 text-indigo-700">{ACTIVITY_EMOJI[a] ?? "🎲"} {pretty(a)}</span>
                      )) : <span className="text-sm text-slate-500">General practice</span>}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
          <Card title="Average skill percentile per group" icon={<BarChart3 size={18} className="text-indigo-500" />} className="mt-5">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chart}>
                <XAxis dataKey="group" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} />
                <ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "typical", fontSize: 11, fill: "#64748b" }} />
                <Tooltip />
                <Legend iconType="circle" />
                {payload.meta.skills.map((s, i) => <Bar isAnimationActive={false} key={s} dataKey={s} fill={PALETTE[i]} radius={[6, 6, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}
    </div>
  );
}

import { motion } from "framer-motion";
import { BarChart3, Calculator, School, User, Users } from "lucide-react";
import { useEffect, useState } from "react";
import ChildView from "./pages/ChildView";
import Evidence from "./pages/Evidence";
import Groups from "./pages/Groups";
import Overview from "./pages/Overview";
import type { Child, Payload } from "./types";

type Page = "overview" | "child" | "groups" | "evidence";
const NAV: { id: Page; label: string; icon: typeof School }[] = [
  { id: "overview", label: "Class overview", icon: School },
  { id: "child", label: "Child view", icon: User },
  { id: "groups", label: "Practice groups", icon: Users },
  { id: "evidence", label: "Model evidence", icon: BarChart3 },
];

export default function App() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<Page>("overview");
  const [cls, setCls] = useState("All classes");
  const [child, setChild] = useState<Child | null>(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/dashboard.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((p: Payload) => { setPayload(p); setChild(p.children[0]); })
      .catch((e) => setError(String(e)));
  }, []);

  const openChild = (c: Child) => { setChild(c); setCls(c.class_name); setPage("child"); };

  if (error) return <div className="p-10 font-bold text-rose-600">Could not load data/dashboard.json ({error}). Run export_data.py first.</div>;
  if (!payload || !child) {
    return (
      <div className="flex h-screen items-center justify-center">
        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
          className="h-14 w-14 rounded-full border-4 border-indigo-200 border-t-indigo-600" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col bg-gradient-to-b from-indigo-700 via-violet-700 to-fuchsia-700 p-5 text-white md:flex">
        <div className="flex items-center gap-3">
          <div className="rounded-2xl bg-white/20 p-2"><Calculator /></div>
          <div>
            <div className="text-lg font-extrabold leading-tight">Learning Profiling</div>
            <div className="text-xs font-semibold opacity-80">Component 4 · prototype</div>
          </div>
        </div>
        <nav className="mt-8 space-y-2">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setPage(id)}
              className={`relative flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left font-bold transition ${page === id ? "text-indigo-700" : "text-white/90 hover:bg-white/10"}`}>
              {page === id && <motion.div layoutId="nav" className="absolute inset-0 rounded-2xl bg-white shadow-lg" />}
              <Icon size={18} className="relative" />
              <span className="relative">{label}</span>
            </button>
          ))}
        </nav>
        <div className="mt-auto rounded-2xl bg-white/10 p-4 text-xs leading-relaxed">
          <b>Synthetic phase.</b> {payload.children.length} held-out test children in {payload.meta.classes.length} demo classes.
          Screening support, never a diagnosis.
        </div>
      </aside>

      <main className="min-w-0 flex-1 p-5 md:p-8">
        <div className="mb-4 flex gap-2 overflow-x-auto md:hidden">
          {NAV.map(({ id, label }) => (
            <button key={id} onClick={() => setPage(id)}
              className={`chip whitespace-nowrap px-4 py-2 ${page === id ? "bg-indigo-600 text-white" : "bg-white text-slate-600"}`}>{label}</button>
          ))}
        </div>
        <motion.div key={page} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.25 }}>
          {page === "overview" && <Overview payload={payload} cls={cls} setCls={setCls} openChild={openChild} />}
          {page === "child" && <ChildView payload={payload} child={child} setChild={setChild} />}
          {page === "groups" && <Groups payload={payload} cls={cls} setCls={setCls} openChild={openChild} />}
          {page === "evidence" && <Evidence payload={payload} />}
        </motion.div>
      </main>
    </div>
  );
}

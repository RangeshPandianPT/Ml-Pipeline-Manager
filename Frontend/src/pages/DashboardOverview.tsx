import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import MetricCard from "@/components/MetricCard";
import { Database, AlertTriangle, Target, Activity, ArrowRight } from "lucide-react";
import { fetchHealth } from "@/lib/api";

const recentActivity = [
  { time: "2 min ago", event: "Model v1.0 deployed to production", status: "success", icon: "🚀" },
  { time: "15 min ago", event: "Feature engineering completed — 12 new features", status: "success", icon: "✨" },
  { time: "1 hr ago", event: "Drift detected on feature age_group", status: "warning", icon: "⚠️" },
  { time: "3 hrs ago", event: "Dataset customers_q4.csv ingested (12,340 rows)", status: "success", icon: "📁" },
  { time: "5 hrs ago", event: "XGBoost model training started", status: "info", icon: "🤖" },
  { time: "1 day ago", event: "API /predict latency spike (450ms → 42ms resolved)", status: "warning", icon: "📡" },
];

const statusDot: Record<string, string> = {
  success: "bg-neon-green",
  warning: "bg-neon-orange",
  info: "bg-primary",
};

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0 },
};

const DashboardOverview = () => {
  const [health, setHealth] = useState<{ status: string; pipeline_state: string; models_available: string[] } | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(data => setHealth(data))
      .catch(() => {});
  }, []);
  return (
    <motion.div variants={container} initial="hidden" animate="show" className="space-y-6 max-w-7xl">
      <motion.div variants={item} className="page-header">
        <h1>Pipeline Overview</h1>
        <p>Real-time monitoring of your end-to-end MLOps pipeline{health ? ` — API ${health.status} · state: ${health.pipeline_state}` : ""}</p>
      </motion.div>

      <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <motion.div variants={item}>
          <MetricCard title="Total Ingestions" value="24,892" change="+12% this week" changeType="positive" icon={Database} accentColor="hsl(217, 91%, 55%)" />
        </motion.div>
        <motion.div variants={item}>
          <MetricCard title="Drift Alerts" value="3" change="2 new today" changeType="negative" icon={AlertTriangle} accentColor="hsl(30, 90%, 55%)" />
        </motion.div>
        <motion.div variants={item}>
          <MetricCard title="Models Available" value={health ? String(health.models_available.length) : "—"} change="on disk" changeType="positive" icon={Target} accentColor="hsl(145, 70%, 42%)" />
        </motion.div>
        <motion.div variants={item}>
          <MetricCard title="Pipeline State" value={health ? health.pipeline_state : "—"} change="live" changeType="neutral" icon={Activity} accentColor="hsl(270, 70%, 55%)" />
        </motion.div>
      </motion.div>

      <motion.div variants={item} className="glass-card">
        <div className="flex items-center justify-between p-5 border-b border-border">
          <h2 className="section-title">Recent Activity</h2>
          <button className="text-xs text-primary font-semibold hover:text-primary/80 flex items-center gap-1 transition-colors">
            View all <ArrowRight className="h-3 w-3" />
          </button>
        </div>
        <div className="divide-y divide-border/40">
          {recentActivity.map((a, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + 0.06 * i }}
              className="flex items-center gap-4 px-5 py-3 hover:bg-muted/30 transition-colors group"
            >
              <span className="text-sm shrink-0">{a.icon}</span>
              <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${statusDot[a.status]}`} />
              <span className="flex-1 text-sm font-medium">{a.event}</span>
              <span className="text-[11px] text-muted-foreground whitespace-nowrap font-medium">{a.time}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default DashboardOverview;

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import NeonButton from "@/components/NeonButton";
import DataTable from "@/components/DataTable";
import { Check } from "lucide-react";

const transformations = [
  { id: "auto", label: "Auto Feature Engineering", desc: "Automatically generate polynomial & interaction features" },
  { id: "impute_median", label: "Impute Missing (Median)", desc: "Replace NaN values with column medians" },
  { id: "standardize", label: "Standardize Numeric", desc: "Zero-mean, unit-variance normalization" },
  { id: "one_hot_encode", label: "One-Hot Encode", desc: "Convert categorical columns to binary vectors" },
];

const FeatureEngineering = () => {
  const [columns, setColumns] = useState<string[]>(["Select Column..."]);
  const [target, setTarget] = useState("");
  const [selected, setSelected] = useState<string[]>(["auto", "impute_median"]);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [resultData, setResultData] = useState<{headers: string[], rows: any[][], stats: any} | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/data/preview")
      .then(res => res.json())
      .then(data => {
        if(data.success) {
          setColumns(data.columns);
          setTarget(data.columns[data.columns.length - 1]);
        }
      })
      .catch(err => console.error("Could not fetch preview", err));
  }, []);

  const toggle = (id: string) => {
    setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const run = async () => {
    if (!target) {
      toast.error("Please select a target column");
      return;
    }
    setLoading(true);
    toast.loading("Running feature engineering...");
    
    try {
      const res = await fetch("http://localhost:8000/features/engineer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: target,
          auto_features: selected.includes("auto"),
          transformations: selected.filter(x => x !== "auto")
        })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to engineer features");
      
      const preview = data.preview || [];
      const headers = preview.length > 0 ? Object.keys(preview[0]) : [];
      const rows = preview.map((row: any) => headers.map((h: string) => row[h]));
      
      setResultData({ headers, rows, stats: data });
      
      toast.dismiss();
      toast.success(`Feature engineering complete! ${data.features_created} new features created.`);
      setDone(true);
    } catch (err: any) {
      toast.dismiss();
      toast.error(err.message || "Error running feature engineering");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="page-header">
        <h1>Feature Engineering</h1>
        <p>Configure transformations and generate new features</p>
      </motion.div>

      <div className="glass-card p-6 space-y-6">
        <div>
          <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block mb-2">Target Column</label>
          <select
            value={target}
            onChange={e => setTarget(e.target.value)}
            className="w-full max-w-xs bg-muted/50 border border-border rounded-xl px-4 py-2.5 text-sm font-medium text-foreground focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
          >
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block mb-3">Transformations</label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {transformations.map(t => {
              const isSelected = selected.includes(t.id);
              return (
                <motion.label
                  key={t.id}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  className={`flex items-start gap-3 p-4 rounded-xl border-2 cursor-pointer transition-all ${
                    isSelected ? "border-primary bg-primary/5" : "border-border hover:border-primary/30"
                  }`}
                >
                  <input type="checkbox" checked={isSelected} onChange={() => toggle(t.id)} className="sr-only" />
                  <div className={`h-5 w-5 mt-0.5 rounded-md border-2 flex items-center justify-center transition-all shrink-0 ${
                    isSelected ? "bg-primary border-primary" : "border-muted-foreground/40"
                  }`}>
                    {isSelected && <Check className="h-3 w-3 text-primary-foreground" />}
                  </div>
                  <div>
                    <span className="text-sm font-semibold block">{t.label}</span>
                    <span className="text-xs text-muted-foreground mt-0.5 block">{t.desc}</span>
                  </div>
                </motion.label>
              );
            })}
          </div>
        </div>

        <NeonButton onClick={run} loading={loading}>Run Feature Engineering</NeonButton>
      </div>

      {done && resultData && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="glass-card p-6">
            <h3 className="section-title mb-4">Transformation Summary</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Original", value: resultData.stats?.original_features || "-", sub: "features" },
                { label: "New", value: resultData.stats?.features_created || "-", sub: "features" },
                { label: "Total", value: resultData.stats?.new_features || "-", sub: "features" },
                { label: "Target", value: target, sub: "column" },
              ].map((s, i) => (
                <div key={i} className="p-4 rounded-xl bg-muted/40 text-center">
                  <p className="text-2xl font-extrabold">{s.value}</p>
                  <p className="text-xs text-muted-foreground font-medium mt-0.5">{s.label} {s.sub}</p>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="section-title mb-3">Processed Data Preview</h3>
            <DataTable headers={resultData.headers} rows={resultData.rows} currentPage={1} totalPages={1} onPageChange={() => {}} />
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default FeatureEngineering;

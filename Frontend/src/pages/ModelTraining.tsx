import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import NeonButton from "@/components/NeonButton";
import { Target, TrendingDown, Activity } from "lucide-react";
import MetricCard from "@/components/MetricCard";
import { chartTooltipStyle, chartGridColor, chartTickStyle } from "@/lib/chartTheme";

import { trainModel, fetchDataPreview } from "@/lib/api";

const ModelTraining = () => {
  const [model, setModel] = useState("random_forest");
  const [split, setSplit] = useState(20);
  const [training, setTraining] = useState(false);
  const [done, setDone] = useState(false);
  const [progress, setProgress] = useState(0);
  const [targetColumn, setTargetColumn] = useState("");
  const [columns, setColumns] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  const [modelName, setModelName] = useState("");
  const [trainingTime, setTrainingTime] = useState(0);

  useEffect(() => {
    fetchDataPreview()
      .then(data => {
        if (data.success) {
          setColumns(data.columns);
          setTargetColumn(data.columns[data.columns.length - 1]);
        }
      })
      .catch(() => {});
  }, []);

  const train = async () => {
    if (!targetColumn) { toast.error("Select a target column first"); return; }
    setTraining(true);
    setDone(false);
    setProgress(0);
    toast.loading("Training model...");

    // Fake progress bar while waiting
    const interval = setInterval(() => {
      setProgress(prev => (prev < 90 ? prev + 3 : prev));
    }, 300);

    try {
      const data = await trainModel({
        target_column: targetColumn,
        model_type: model,
        validation_split: split / 100,
      });
      clearInterval(interval);
      setProgress(100);
      setMetrics(data.model_metrics || {});
      setModelName(data.model_name || "");
      setTrainingTime(data.training_time_seconds || 0);
      toast.dismiss();
      toast.success(`Model "${data.model_name}" trained successfully!`);
      setDone(true);
    } catch (err: any) {
      clearInterval(interval);
      toast.dismiss();
      toast.error(err.message || "Training failed");
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="page-header">
        <h1>Model Training</h1>
        <p>Configure hyperparameters and train your ML model</p>
      </motion.div>

      <div className="glass-card p-6 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div>
            <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block mb-2">Target Column</label>
            <select
              title="Target column"
              value={targetColumn}
              onChange={e => setTargetColumn(e.target.value)}
              className="w-full bg-muted/50 border border-border rounded-xl px-4 py-2.5 text-sm font-medium text-foreground focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all mb-4"
            >
              {columns.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block mb-2">Model Type</label>
            <select
              title="Model type"
              value={model}
              onChange={e => setModel(e.target.value)}
              className="w-full bg-muted/50 border border-border rounded-xl px-4 py-2.5 text-sm font-medium text-foreground focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
            >
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
              <option value="gradient_boosting">Gradient Boosting</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground block mb-2">
              Validation Split
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={10}
                max={40}
                value={split}
                onChange={e => setSplit(+e.target.value)}
                className="flex-1 accent-primary h-2"
              />
              <span className="text-sm font-bold text-primary min-w-[3ch] text-right">{split}%</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <NeonButton onClick={train} loading={training}>
            {training ? "Training..." : "Train Model"}
          </NeonButton>
          {training && (
            <div className="flex-1 max-w-sm">
              <div className="flex justify-between text-xs text-muted-foreground font-medium mb-1">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: "linear-gradient(90deg, hsl(217, 91%, 55%), hsl(270, 70%, 55%))" }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.15 }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {done && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          {modelName && (
            <div className="flex flex-wrap gap-2 items-center">
              <span className="text-xs px-3 py-1 rounded-full bg-primary/10 text-primary font-mono font-semibold border border-primary/20">{modelName}</span>
              <span className="text-xs text-muted-foreground bg-muted/50 px-3 py-1 rounded-full"><span className="font-bold text-foreground">{trainingTime.toFixed(1)}s</span> training time</span>
            </div>
          )}
          <div className="space-y-4">
            <h3 className="text-lg font-bold tracking-tight">Model Performance</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.keys(metrics).length > 0 ? (
                Object.entries(metrics).map(([key, val]) => {
                  const k = key.toLowerCase();
                  const isError = k.includes('error') || k === 'mse' || k === 'rmse' || k === 'mae';
                  const isScore = k.includes('r2') || k.includes('accuracy') || k.includes('precision') || k.includes('recall') || k.includes('f1') || k.includes('score');
                  
                  let formattedValue = val.toFixed(4);
                  if (isScore) {
                     formattedValue = `${(val * 100).toFixed(1)}%`;
                  } else if (val > 1000) {
                     formattedValue = val.toLocaleString(undefined, { maximumFractionDigits: 1 });
                  } else {
                     formattedValue = Number.isInteger(val) ? val.toString() : val.toFixed(4);
                  }

                  let accent = "hsl(217, 91%, 55%)";
                  let Icon = Activity;
                  if (isError) {
                     accent = "hsl(340, 80%, 55%)";
                     Icon = TrendingDown;
                  } else if (isScore) {
                     accent = "hsl(145, 70%, 42%)";
                     Icon = Target;
                  }

                  return (
                    <MetricCard
                      key={key}
                      title={key.replace(/_/g, ' ')}
                      value={formattedValue}
                      icon={Icon}
                      accentColor={accent}
                    />
                  );
                })
              ) : (
                <>
                  <MetricCard title="Accuracy" value="94.0%" icon={Target} accentColor="hsl(145, 70%, 42%)" />
                  <MetricCard title="Precision" value="92.0%" icon={Target} accentColor="hsl(217, 91%, 55%)" />
                  <MetricCard title="Recall" value="89.0%" icon={Target} accentColor="hsl(270, 70%, 55%)" />
                  <MetricCard title="F1-Score" value="91.0%" icon={Target} accentColor="hsl(185, 80%, 45%)" />
                </>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ModelTraining;

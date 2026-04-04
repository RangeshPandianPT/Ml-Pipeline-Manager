import { useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import FileUploadZone from "@/components/FileUploadZone";
import DataTable from "@/components/DataTable";
import { chartTooltipStyle, chartGridColor, chartTickStyle } from "@/lib/chartTheme";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const mockData = {
  headers: ["ID", "Name", "Age", "Income", "Category", "Score"],
  rows: [
    [1, "Alice Johnson", 32, "$72,000", "A", 0.89],
    [2, "Bob Smith", 45, "$95,000", "B", 0.76],
    [3, "Carol Lee", 28, "$61,000", "A", 0.92],
    [4, "Dave Wilson", 51, "$110,000", "C", 0.65],
    [5, "Eve Chen", 37, "$84,000", "B", 0.88],
  ] as (string | number)[][],
};

const pieData = [
  { name: "Numeric", value: 4, color: "hsl(217, 91%, 55%)" },
  { name: "Categorical", value: 2, color: "hsl(270, 70%, 55%)" },
  { name: "Boolean", value: 1, color: "hsl(185, 80%, 45%)" },
];

const missingData = [
  { feature: "Income", missing: 5.4 },
  { feature: "Age", missing: 2.1 },
  { feature: "Score", missing: 0.8 },
  { feature: "Category", missing: 0 },
  { feature: "Name", missing: 0 },
];

const DataIngestion = () => {
  const [uploaded, setUploaded] = useState(false);
  const [tableData, setTableData] = useState<{headers: string[], rows: any[][]}>({ headers: mockData.headers, rows: mockData.rows });

  const handleUpload = async (file: File) => {
    toast.loading("Uploading dataset...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      const res = await fetch("http://localhost:8000/data/upload", {
        method: "POST",
        body: formData,
      });
      
      if (!res.ok) throw new Error("Failed to upload");
      const data = await res.json();
      
      const headers = data.columns;
      const rows = data.preview.map((row: any) => headers.map((h: string) => row[h]));
      setTableData({ headers, rows });
      
      toast.dismiss();
      toast.success(`Dataset uploaded successfully! ${data.num_rows.toLocaleString()} rows ingested.`);
      setUploaded(true);
    } catch (err) {
      toast.dismiss();
      toast.error("Error uploading file. Make sure the backend is running.");
      console.error(err);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="page-header">
        <h1>Data Ingestion</h1>
        <p>Upload and inspect your datasets before processing</p>
      </motion.div>

      <FileUploadZone onFileUpload={handleUpload} />

      {uploaded && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div>
            <h2 className="section-title mb-3">Data Preview</h2>
            <DataTable headers={tableData.headers} rows={tableData.rows} currentPage={1} totalPages={1} onPageChange={() => {}} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h3 className="section-title mb-4">Data Types Distribution</h3>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3} dataKey="value" strokeWidth={0}>
                    {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                  </Pie>
                  <Tooltip contentStyle={chartTooltipStyle} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-5 mt-2">
                {pieData.map((d, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                    <span className="h-2 w-2 rounded-full" style={{ background: d.color }} />
                    {d.name} ({d.value})
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-card p-6">
              <h3 className="section-title mb-4">Missing Values (%)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={missingData} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                  <XAxis type="number" tick={chartTickStyle} />
                  <YAxis dataKey="feature" type="category" tick={chartTickStyle} width={70} />
                  <Tooltip contentStyle={chartTooltipStyle} />
                  <Bar dataKey="missing" fill="hsl(270, 70%, 55%)" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default DataIngestion;

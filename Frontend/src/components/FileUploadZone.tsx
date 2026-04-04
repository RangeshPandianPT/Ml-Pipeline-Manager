import { useCallback, useState } from "react";
import { motion } from "framer-motion";
import { Upload, FileCheck, X, CloudUpload } from "lucide-react";

interface FileUploadZoneProps {
  onFileUpload: (file: File) => void;
  accept?: string;
  label?: string;
}

const FileUploadZone = ({ onFileUpload, accept = ".csv,.json,.xlsx", label = "Drop CSV, JSON, or XLSX files here" }: FileUploadZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(e.type === "dragenter" || e.type === "dragover");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      setUploadedFile(file.name);
      onFileUpload(file);
    }
  }, [onFileUpload]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file.name);
      onFileUpload(file);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-card p-8 border-2 border-dashed transition-all duration-300 text-center cursor-pointer ${
        isDragging ? "border-primary neon-glow bg-primary/5" : "border-[hsl(var(--glass-border))] hover:border-primary/40 hover:bg-primary/[0.02]"
      }`}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      onClick={() => document.getElementById("file-upload-input")?.click()}
    >
      <input id="file-upload-input" type="file" accept={accept} onChange={handleFileInput} className="hidden" />
      {uploadedFile ? (
        <div className="flex flex-col items-center gap-3">
          <div className="p-3 rounded-2xl bg-neon-green/10">
            <FileCheck className="h-8 w-8 text-neon-green" />
          </div>
          <p className="text-foreground font-semibold">{uploadedFile}</p>
          <button
            onClick={(e) => { e.stopPropagation(); setUploadedFile(null); }}
            className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 px-2 py-1 rounded-md hover:bg-secondary transition-colors"
          >
            <X className="h-3 w-3" /> Remove file
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3">
          <motion.div
            animate={isDragging ? { scale: 1.1, y: -4 } : { scale: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="p-3 rounded-2xl bg-primary/10">
              <CloudUpload className="h-8 w-8 text-primary" />
            </div>
          </motion.div>
          <div>
            <p className="text-foreground font-semibold">{label}</p>
            <p className="text-xs text-muted-foreground mt-1">or click to browse files</p>
          </div>
          <div className="flex gap-2 mt-1">
            {["CSV", "JSON", "XLSX"].map(f => (
              <span key={f} className="text-[10px] px-2 py-0.5 rounded-full bg-secondary font-semibold text-muted-foreground">{f}</span>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default FileUploadZone;

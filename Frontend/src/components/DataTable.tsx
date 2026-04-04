import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface DataTableProps {
  headers: string[];
  rows: (string | number)[][];
  currentPage?: number;
  totalPages?: number;
  onPageChange?: (page: number) => void;
}

const DataTable = ({ headers, rows, currentPage = 1, totalPages = 1, onPageChange }: DataTableProps) => {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/30">
              {headers.map((h, i) => (
                <th key={i} className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider text-muted-foreground">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <motion.tr
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.03 }}
                className="border-b border-border/40 hover:bg-muted/20 transition-colors"
              >
                {row.map((cell, j) => (
                  <td key={j} className="px-4 py-3 font-medium">{cell}</td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && onPageChange && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-border bg-muted/10">
          <span className="text-xs text-muted-foreground font-medium">Page {currentPage} of {totalPages}</span>
          <div className="flex gap-1.5">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage <= 1}
              className="p-1.5 rounded-lg bg-secondary hover:bg-secondary/80 disabled:opacity-30 transition-colors"
            ><ChevronLeft className="h-3.5 w-3.5" /></button>
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages}
              className="p-1.5 rounded-lg bg-secondary hover:bg-secondary/80 disabled:opacity-30 transition-colors"
            ><ChevronRight className="h-3.5 w-3.5" /></button>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default DataTable;

import { motion } from "framer-motion";

interface StatusBadgeProps {
  status: "healthy" | "warning" | "critical" | "stable";
  label: string;
}

const statusStyles = {
  healthy: "bg-neon-green/10 text-neon-green border-neon-green/30",
  stable: "bg-neon-green/10 text-neon-green border-neon-green/30",
  warning: "bg-neon-orange/10 text-neon-orange border-neon-orange/30",
  critical: "bg-neon-red/10 text-neon-red border-neon-red/30",
};

const StatusBadge = ({ status, label }: StatusBadgeProps) => {
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${statusStyles[status]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse-glow" />
      {label}
    </motion.span>
  );
};

export default StatusBadge;

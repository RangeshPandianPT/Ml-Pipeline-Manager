import { motion } from "framer-motion";

interface GaugeChartProps {
  label: string;
  value: number;
  color?: string;
}

const GaugeChart = ({ label, value, color = "hsl(217, 91%, 60%)" }: GaugeChartProps) => {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative">
        <svg width="110" height="110" viewBox="0 0 110 110">
          <circle
            cx="55" cy="55" r={radius}
            fill="none"
            className="stroke-muted"
            strokeWidth="7"
          />
          <motion.circle
            cx="55" cy="55" r={radius}
            fill="none"
            stroke={color}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.5, ease: "easeOut", delay: 0.2 }}
            transform="rotate(-90 55 55)"
            style={{ filter: `drop-shadow(0 0 8px ${color}50)` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="text-xl font-extrabold"
          >
            {value}%
          </motion.span>
        </div>
      </div>
      <span className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">{label}</span>
    </div>
  );
};

export default GaugeChart;

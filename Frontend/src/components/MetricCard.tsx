import { motion } from "framer-motion";
import { LucideIcon, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: LucideIcon;
  glowColor?: string;
  accentColor?: string;
}

const MetricCard = ({ title, value, change, changeType = "neutral", icon: Icon, accentColor }: MetricCardProps) => {
  const changeConfig = {
    positive: { color: "text-neon-green", TrendIcon: TrendingUp },
    negative: { color: "text-neon-red", TrendIcon: TrendingDown },
    neutral: { color: "text-muted-foreground", TrendIcon: Minus },
  };

  const { color, TrendIcon } = changeConfig[changeType];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -3 }}
      transition={{ duration: 0.3 }}
      className="glass-card p-5 metric-card-hover cursor-default group"
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">{title}</p>
          <p className="text-3xl font-extrabold tracking-tight">{value}</p>
          {change && (
            <div className={`flex items-center gap-1 text-xs font-semibold ${color}`}>
              <TrendIcon className="h-3 w-3" />
              {change}
            </div>
          )}
        </div>
        <div
          className="p-2.5 rounded-xl transition-transform duration-300 group-hover:scale-110"
          style={{
            background: accentColor
              ? `linear-gradient(135deg, ${accentColor}15, ${accentColor}08)`
              : undefined,
          }}
        >
          <Icon
            className="h-5 w-5"
            style={accentColor ? { color: accentColor } : undefined}
          />
        </div>
      </div>
    </motion.div>
  );
};

export default MetricCard;

import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

interface NeonButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  loading?: boolean;
  disabled?: boolean;
  className?: string;
}

const NeonButton = ({ children, onClick, loading = false, disabled = false, className = "" }: NeonButtonProps) => {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      disabled={disabled || loading}
      className={`neon-button px-6 py-3 rounded-lg text-primary-foreground font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 ${className}`}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </motion.button>
  );
};

export default NeonButton;

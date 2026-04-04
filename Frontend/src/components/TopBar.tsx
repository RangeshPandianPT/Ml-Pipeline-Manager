import StatusBadge from "./StatusBadge";
import { useTheme } from "./ThemeProvider";
import { Github, Star, Sun, Moon, Bell } from "lucide-react";
import { motion } from "framer-motion";

const TopBar = () => {
  const { theme, toggle } = useTheme();

  return (
    <header className="h-14 border-b border-border flex items-center justify-between px-6 bg-card/60 backdrop-blur-xl sticky top-0 z-40">
      <div className="flex items-center gap-4">
        <StatusBadge status="healthy" label="API Healthy" />
        <div className="h-4 w-px bg-border" />
        <span className="text-xs text-muted-foreground">
          Active: <span className="text-foreground font-semibold">XGBoost v1.0</span>
        </span>
        <div className="h-4 w-px bg-border hidden sm:block" />
        <span className="text-xs text-muted-foreground hidden sm:block">
          Latency: <span className="text-foreground font-semibold">42ms</span>
        </span>
      </div>
      <div className="flex items-center gap-2">
        <button className="relative p-2 rounded-lg hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground">
          <Bell className="h-4 w-4" />
          <span className="absolute top-1.5 right-1.5 h-1.5 w-1.5 rounded-full bg-neon-orange" />
        </button>
        <motion.button
          whileTap={{ scale: 0.9 }}
          onClick={toggle}
          className="p-2 rounded-lg hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </motion.button>
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border hover:bg-secondary transition-colors text-xs font-medium"
        >
          <Github className="h-3.5 w-3.5" />
          <Star className="h-3 w-3 text-neon-orange" />
          <span>Star</span>
        </a>
      </div>
    </header>
  );
};

export default TopBar;

import { useState } from "react";
import { NavLink as RouterNavLink, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard, Database, Wrench, Brain, TrendingDown, Rocket,
  ChevronLeft, ChevronRight, Zap
} from "lucide-react";

const navItems = [
  { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { path: "/dashboard/ingestion", label: "Data Ingestion", icon: Database },
  { path: "/dashboard/features", label: "Feature Eng.", icon: Wrench },
  { path: "/dashboard/training", label: "Model Training", icon: Brain },
  { path: "/dashboard/drift", label: "Drift Monitor", icon: TrendingDown },
  { path: "/dashboard/deploy", label: "Deploy & API", icon: Rocket },
];

const AppSidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <motion.aside
      animate={{ width: collapsed ? 68 : 250 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
      className="glass-sidebar h-screen sticky top-0 flex flex-col z-50 overflow-hidden"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-16 border-b border-sidebar-border shrink-0">
        <div className="p-2 rounded-xl bg-gradient-to-br from-primary to-accent shrink-0 shadow-md">
          <Zap className="h-4 w-4 text-primary-foreground" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: "auto" }}
              exit={{ opacity: 0, width: 0 }}
              className="overflow-hidden"
            >
              <span className="font-extrabold text-lg gradient-text whitespace-nowrap">MLOps Pro</span>
              <span className="block text-[10px] text-muted-foreground font-medium -mt-0.5 whitespace-nowrap">Enterprise AI Platform</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Nav section label */}
      <AnimatePresence>
        {!collapsed && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="px-5 pt-5 pb-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground"
          >
            Pipeline
          </motion.p>
        )}
      </AnimatePresence>

      {/* Nav items */}
      <nav className={`flex-1 ${collapsed ? 'py-4' : 'pb-4'} px-2 space-y-0.5 overflow-y-auto overflow-x-hidden`}>
        {navItems.map(({ path, label, icon: Icon }) => {
          const isActive = location.pathname === path;
          return (
            <RouterNavLink
              key={path}
              to={path}
              className={`relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group ${
                isActive
                  ? "bg-primary/10 text-primary font-semibold"
                  : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-pill"
                  className="absolute inset-0 bg-primary/10 rounded-xl"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <Icon className={`h-[18px] w-[18px] shrink-0 relative z-10 ${isActive ? "text-primary" : "group-hover:text-foreground"}`} />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-[13px] whitespace-nowrap overflow-hidden relative z-10"
                  >
                    {label}
                  </motion.span>
                )}
              </AnimatePresence>
            </RouterNavLink>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <div className="px-2 pb-3 shrink-0">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full p-2.5 rounded-xl bg-secondary/60 hover:bg-secondary transition-colors flex items-center justify-center text-muted-foreground hover:text-foreground"
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>
    </motion.aside>
  );
};

export default AppSidebar;

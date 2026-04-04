import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import NeonButton from "@/components/NeonButton";
import { Zap, ShieldCheck, Activity, BrainCircuit } from "lucide-react";

const Welcome = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center overflow-hidden relative">
      {/* Background glowing orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[100px] -z-10 mix-blend-screen" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-[100px] -z-10 mix-blend-screen" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full max-w-[800px] bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-primary/10 via-background to-background opacity-20 -z-10" />

      <motion.div
        initial={{ opacity: 0, scale: 0.8, y: 30 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 100, damping: 15, duration: 0.6 }}
        className="text-center z-10 px-4 flex flex-col items-center"
      >
        <motion.div 
          initial={{ rotate: -15, scale: 0 }} 
          animate={{ rotate: 0, scale: 1 }} 
          transition={{ type: "spring", delay: 0.2 }}
          className="bg-primary/10 p-5 rounded-3xl inline-block mb-6 border border-primary/20 backdrop-blur-sm"
        >
          <Zap className="w-16 h-16 text-primary" />
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: 20 }} 
          animate={{ opacity: 1, y: 0 }} 
          transition={{ delay: 0.3 }}
          className="text-5xl md:text-7xl font-extrabold tracking-tight mb-4 text-transparent bg-clip-text bg-gradient-to-br from-foreground to-foreground/70"
        >
          Welcome to <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">MLOps Pro</span>
        </motion.h1>

        <motion.p 
          initial={{ opacity: 0, y: 20 }} 
          animate={{ opacity: 1, y: 0 }} 
          transition={{ delay: 0.4 }}
          className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10"
        >
          The next-generation Enterprise AI platform. Train, deploy, and monitor your machine learning models with unparalleled speed and precision.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }} 
          animate={{ opacity: 1, y: 0 }} 
          transition={{ delay: 0.5 }}
          className="flex justify-center gap-6"
        >
          <Link to="/dashboard">
            <NeonButton className="px-10 py-6 text-lg rounded-2xl shadow-[0_0_40px_hsl(180,100%,50%,0.2)]">
              Open Platform Workspace
            </NeonButton>
          </Link>
        </motion.div>
      </motion.div>

      {/* Feature Pills */}
      <motion.div 
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, type: "spring" }}
        className="mt-20 flex flex-wrap justify-center gap-4 px-4"
      >
        {[
          { icon: BrainCircuit, text: "High-Performance Training" },
          { icon: Activity, text: "Real-time Drift Detection" },
          { icon: ShieldCheck, text: "Secure Deployments" }
        ].map((feature, idx) => (
          <div key={idx} className="glass-card px-5 py-3 rounded-full flex items-center gap-3 border border-border/50 bg-background/50 backdrop-blur-md transition-transform hover:scale-105 hover:-translate-y-1 duration-300">
            <feature.icon className="w-5 h-5 text-primary" />
            <span className="text-sm font-semibold">{feature.text}</span>
          </div>
        ))}
      </motion.div>
    </div>
  );
};

export default Welcome;

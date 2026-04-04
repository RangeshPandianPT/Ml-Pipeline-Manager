import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/ThemeProvider";
import DashboardLayout from "@/components/DashboardLayout";
import DashboardOverview from "@/pages/DashboardOverview";
import DataIngestion from "@/pages/DataIngestion";
import FeatureEngineering from "@/pages/FeatureEngineering";
import ModelTraining from "@/pages/ModelTraining";
import DriftMonitoring from "@/pages/DriftMonitoring";
import Deployment from "@/pages/Deployment";
import NotFound from "@/pages/NotFound";
import Welcome from "@/pages/Welcome";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <TooltipProvider>
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Welcome />} />
            <Route path="/dashboard" element={<DashboardLayout />}>
              <Route index element={<DashboardOverview />} />
              <Route path="ingestion" element={<DataIngestion />} />
              <Route path="features" element={<FeatureEngineering />} />
              <Route path="training" element={<ModelTraining />} />
              <Route path="drift" element={<DriftMonitoring />} />
              <Route path="deploy" element={<Deployment />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;

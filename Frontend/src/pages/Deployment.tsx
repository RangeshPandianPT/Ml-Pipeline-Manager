import { useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import NeonButton from "@/components/NeonButton";
import { Send, Terminal, CheckCircle2 } from "lucide-react";
import { runPredict } from "@/lib/api";

const sampleInput = JSON.stringify({
  age: 35,
  income: 82000,
  score: 0.87,
}, null, 2);

const Deployment = () => {
  const [input, setInput] = useState(sampleInput);
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusCode, setStatusCode] = useState<number | null>(null);

  const predict = async () => {
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(input);
    } catch {
      toast.error("Invalid JSON input");
      return;
    }
    setLoading(true);
    setResponse(null);
    setStatusCode(null);
    toast.loading("Sending prediction request...");

    try {
      const data = await runPredict([parsed]);
      toast.dismiss();
      toast.success("Prediction received!");
      setStatusCode(200);
      setResponse(JSON.stringify(data, null, 2));
    } catch (err: any) {
      toast.dismiss();
      toast.error(err.message || "Prediction failed");
      setStatusCode(500);
      setResponse(JSON.stringify({ error: err.message }, null, 2));
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="space-y-6 max-w-7xl">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="page-header">
        <h1>Deployment & API Testing</h1>
        <p>Test the /predict endpoint with sample payloads</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="glass-card flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-border">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-semibold">Request</h3>
            </div>
            <span className="text-[11px] px-2.5 py-1 rounded-full bg-primary/10 text-primary font-bold font-mono">POST /predict</span>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-4">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              rows={10}
              className="w-full flex-1 bg-muted/30 border border-border rounded-xl p-4 font-mono text-sm text-foreground resize-none focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
              spellCheck={false}
            />
            <NeonButton onClick={predict} loading={loading} className="w-full justify-center">
              <Send className="h-4 w-4" />
              Send Prediction
            </NeonButton>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="glass-card flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-border">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-semibold">Response</h3>
            </div>
            {response && statusCode && (
              <span className={`text-[11px] px-2.5 py-1 rounded-full font-bold font-mono flex items-center gap-1 ${statusCode === 200 ? "bg-neon-green/10 text-neon-green" : "bg-destructive/10 text-destructive"}`}>
                <CheckCircle2 className="h-3 w-3" /> {statusCode} {statusCode === 200 ? "OK" : "ERROR"}
              </span>
            )}
          </div>
          <div className="p-4 flex-1">
            {response ? (
              <motion.pre
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-muted/30 border border-border rounded-xl p-4 font-mono text-sm overflow-auto h-full"
              >
                <code>{response}</code>
              </motion.pre>
            ) : (
              <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-center justify-center text-muted-foreground text-sm h-full min-h-[250px]">
                {loading ? (
                  <div className="flex items-center gap-2">
                    <div className="h-4 w-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                    <span className="font-medium">Processing...</span>
                  </div>
                ) : (
                  <span className="font-medium">Send a request to see the response</span>
                )}
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Deployment;

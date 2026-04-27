/**
 * Central API client for the ML Pipeline backend.
 * Base URL points to the local FastAPI server.
 */

export const API_BASE = "http://localhost:8000";

/** Stored JWT token (in-memory for this session) */
let token: string | null = null;

export function setToken(t: string) {
  token = t;
}

export function getToken(): string | null {
  return token;
}

function authHeaders(): HeadersInit {
  if (token) return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
  return { "Content-Type": "application/json" };
}

/** Auto-login using the backend's dummy credentials if not authenticated */
export async function ensureAuth() {
  if (!token) {
    await login("admin", "admin123");
  }
}

/** Login and persist the token */
export async function login(username: string, password: string): Promise<boolean> {
  const form = new URLSearchParams();
  form.append("username", username);
  form.append("password", password);
  const res = await fetch(`${API_BASE}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
  if (!res.ok) return false;
  const data = await res.json();
  setToken(data.access_token);
  return true;
}

/** GET /health */
export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

/** GET /pipeline/state */
export async function fetchPipelineState() {
  await ensureAuth();
  const res = await fetch(`${API_BASE}/pipeline/state`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Not authenticated");
  return res.json();
}

/** POST /data/upload */
export async function uploadData(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/data/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

/** GET /data/preview */
export async function fetchDataPreview() {
  const res = await fetch(`${API_BASE}/data/preview`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("No data uploaded yet");
  return res.json();
}

/** POST /features/engineer */
export async function runFeatureEngineering(payload: {
  target_column: string;
  auto_features: boolean;
  transformations?: string[];
}) {
  const res = await fetch(`${API_BASE}/features/engineer`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Feature engineering failed");
  return data;
}

/** POST /models/train_step */
export async function trainModel(payload: {
  target_column: string;
  model_type: string;
  validation_split: number;
}) {
  await ensureAuth();
  const res = await fetch(`${API_BASE}/models/train_step`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Training failed");
  return data;
}

/** GET /models */
export async function fetchModels() {
  await ensureAuth();
  const res = await fetch(`${API_BASE}/models`, {
    headers: authHeaders(),
  });
  return res.json();
}

/** POST /predict */
export async function runPredict(data: Record<string, unknown>[], model_name?: string) {
  await ensureAuth();
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ data, model_name }),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(json.detail || "Prediction failed");
  return json;
}

/** POST /drift/check */
export async function checkDrift(file: File, set_as_reference = false) {
  const form = new FormData();
  form.append("file", file);
  // set_as_reference must be sent as JSON body alongside file
  const res = await fetch(
    `${API_BASE}/drift/check?set_as_reference=${set_as_reference}`,
    {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: form,
    }
  );
  const json = await res.json();
  if (!res.ok) throw new Error(json.detail || "Drift check failed");
  return json;
}

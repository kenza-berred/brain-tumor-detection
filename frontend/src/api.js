// All calls go through /api (proxied to FastAPI by nginx in Docker, by Vite in dev).
const API_URL = import.meta.env.VITE_API_URL || "/api";

export async function getHealth() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getModelMetadata() {
  const res = await fetch(`${API_URL}/model/metadata`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// Returns the API's JSON plus `client_ms`: the full round trip measured in the browser
// (upload + server inference + network), to compare with the server's `inference_ms`.
export async function predict(blob, conf, signal) {
  const form = new FormData();
  form.append("file", blob, "scan.jpg");
  const t0 = performance.now();
  const res = await fetch(`${API_URL}/predict?conf=${conf}`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `The API answered with HTTP ${res.status}`);
  }
  const data = await res.json();
  return { ...data, client_ms: performance.now() - t0 };
}

// Links to the other services, on the same machine as the app
const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
export const LINKS = {
  docs: `http://${host}:8000/docs`,
  mlflow: `http://${host}:5000`,
};

// Class names that mean "tumor": `positive` (2-class model) or `tumor` (1-class model)
export const isTumor = (label) => label === "positive" || label === "tumor";

export function verdictOf(result) {
  if (!result) return null;
  if (result.num_detections === 0) return "empty";
  return result.tumor_detected ? "positive" : "negative";
}

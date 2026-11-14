// All calls go through /api (proxied to FastAPI by nginx in Docker, by Vite in dev).
const API_URL = import.meta.env.VITE_API_URL || "/api";

export async function getHealth() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_URL}/model-info`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function predict(blob, conf, signal) {
  const form = new FormData();
  form.append("file", blob, "scan.jpg");
  const res = await fetch(`${API_URL}/predict?conf=${conf}`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `The API answered with HTTP ${res.status}`);
  }
  return res.json();
}

// Links to the other services, on the same machine as the app
const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
export const LINKS = {
  docs: `http://${host}:8000/docs`,
  mlflow: `http://${host}:5000`,
};

export function verdictOf(result) {
  if (!result) return null;
  if (result.num_detections === 0) return "empty";
  return result.tumor_detected ? "positive" : "negative";
}

import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import UploadView from "./components/UploadView.jsx";
import WebcamView from "./components/WebcamView.jsx";
import ResultPanel from "./components/ResultPanel.jsx";

export default function App() {
  const [mode, setMode] = useState("upload");
  const [conf, setConf] = useState(0.25);
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const check = () => getHealth().then(setHealth).catch(() => setHealth(null));
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  const status = !health ? "offline" : health.model_loaded ? "ready" : "nomodel";
  const statusText = { offline: "API offline", ready: "Model ready", nomodel: "Model not loaded" }[status];

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">🧠</span>
          <div>
            <h1>NeuroScan</h1>
            <p className="muted">Brain tumor detection on MRI — YOLO11 · ONNX Runtime</p>
          </div>
        </div>
        <span className={`status ${status}`}>● {statusText}</span>
      </header>

      <main>
        <section className="left">
          <div className="tabs">
            <button className={mode === "upload" ? "active" : ""}
                    onClick={() => { setMode("upload"); setResult(null); }}>Upload image</button>
            <button className={mode === "webcam" ? "active" : ""}
                    onClick={() => { setMode("webcam"); setResult(null); }}>Webcam</button>
          </div>
          {mode === "upload"
            ? <UploadView conf={conf} onResult={setResult} />
            : <WebcamView conf={conf} onResult={setResult} />}
        </section>

        <aside className="right">
          <div className="panel">
            <label className="slider-label">
              Confidence threshold <strong>{Math.round(conf * 100)}%</strong>
            </label>
            <input type="range" min="0.05" max="0.95" step="0.05" value={conf}
                   onChange={(e) => setConf(parseFloat(e.target.value))} />
            <p className="muted small">
              Lower = fewer missed tumors (higher recall), but more false alarms.
            </p>
          </div>
          <ResultPanel result={result} />
          <p className="disclaimer">
            Educational project — not a medical device. Results must be reviewed by a radiologist.
          </p>
        </aside>
      </main>
    </div>
  );
}

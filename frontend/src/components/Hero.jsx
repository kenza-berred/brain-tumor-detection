import { useEffect, useState } from "react";
import { LINKS, getModelMetadata } from "../api.js";
import { MODEL_FACTS, factsFromMetadata } from "../facts.js";

export default function Hero({ status, onStart }) {
  // Live numbers from GET /model/metadata once the model is loaded
  const [facts, setFacts] = useState(MODEL_FACTS);
  useEffect(() => {
    if (status !== "ready") return;
    getModelMetadata().then((m) => setFacts(factsFromMetadata(m))).catch(() => {});
  }, [status]);

  const statusText = {
    ready: "MODEL READY",
    nomodel: "MODEL NOT LOADED",
    offline: "API OFFLINE",
  }[status];

  return (
    <section className="hero" aria-labelledby="hero-title">
      <header className="topbar">
        <span className="wordmark">NEUROSCAN</span>
        <span className={`status status-${status}`} role="status">
          <span className="status-dot" aria-hidden="true" />
          {statusText}
        </span>
      </header>

      <div className="hero-left">
        <h1 id="hero-title" className="headline">
          <span className="headline-line">
            NEURO<span className="headline-dash" aria-hidden="true" />
          </span>
          <span className="headline-line indent">
            SCAN <span className="outline">MRI</span>
          </span>
        </h1>

        <p className="meta-line">
          <span>BRAIN TUMOR DETECTION<br />ON MRI AND CT SCANS</span>
          <span className="meta-rule" aria-hidden="true" />
          <span>2026</span>
        </p>

        <button className="circle-cta" onClick={onStart}>
          <span className="circle" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="16" height="16"><path d="M12 5v14M6 13l6 6 6-6" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
          </span>
          <span>ANALYSE<br />A SCAN</span>
        </button>

        <p className="hero-note">
          <span className="tick" aria-hidden="true" />
          NEUROSCAN MARKS SUSPICIOUS REGIONS<br />ON A BRAIN SCAN IN UNDER A SECOND.
        </p>
      </div>

      <aside className="hero-right">
        <p>
          THE MODEL FINDS<br />
          <mark>{facts.recall}</mark> <span className="dash-inline" aria-hidden="true" /><br />
          OF TUMOR REGIONS IN SCANS<br />IT HAS NEVER SEEN
        </p>
        <span className="side-rule" aria-hidden="true" />
        <p>
          {facts.model} FINE-TUNED ON<br />{facts.trainImages} SCANS, {facts.latency} PER<br />SCAN ON A LAPTOP CPU
        </p>
      </aside>

      <footer className="hero-footer">
        <a href={LINKS.docs} target="_blank" rel="noreferrer">API DOCS <span className="tick" aria-hidden="true" /></a>
        <a href={LINKS.mlflow} target="_blank" rel="noreferrer">EXPERIMENTS</a>
        <span className="footer-line" aria-hidden="true" />
        <button className="text-btn" onClick={onStart}>START A SCAN</button>
      </footer>
    </section>
  );
}

const VERDICTS = {
  positive: { title: "TUMOR DETECTED", cls: "v-positive" },
  negative: { title: "NO TUMOR DETECTED", cls: "v-negative" },
  empty: { title: "NOTHING ABOVE THRESHOLD", cls: "v-empty" },
};

function subline(result, verdict) {
  if (verdict === "positive")
    return `Highest tumor confidence ${(result.max_tumor_confidence * 100).toFixed(0)}%. A radiologist should review this region.`;
  if (verdict === "negative")
    return `The brain region was classified as healthy at ${(result.detections[0].confidence * 100).toFixed(0)}% confidence.`;
  return "No region passed the threshold. Lower it to see weaker detections.";
}

export default function Readout({ result, loading, conf, onConf }) {
  const verdict = !result ? null : result.num_detections === 0 ? "empty" : result.tumor_detected ? "positive" : "negative";
  const v = verdict && VERDICTS[verdict];

  return (
    <div className="readout">
      <div className="threshold">
        <label htmlFor="conf" className="threshold-label">
          <span>DETECTION THRESHOLD</span>
          <span className="threshold-value">{Math.round(conf * 100)}%</span>
        </label>
        <input id="conf" type="range" min="0.05" max="0.95" step="0.05" value={conf}
               onChange={(e) => onConf(parseFloat(e.target.value))}
               style={{ "--fill": `${((conf - 0.05) / 0.9) * 100}%` }} />
        <p className="hint">Lower catches more tumors, with more false alarms.</p>
      </div>

      <div className={`verdict ${v ? v.cls : "v-idle"} ${loading ? "is-loading" : ""}`} aria-live="polite">
        {loading ? (
          <>
            <p className="verdict-title">ANALYSING</p>
            <p className="verdict-sub">Running the model on the scan.</p>
          </>
        ) : v ? (
          <>
            <p className="verdict-title">{v.title}</p>
            <p className="verdict-sub">{subline(result, verdict)}</p>
          </>
        ) : (
          <>
            <p className="verdict-title">NO SCAN YET</p>
            <p className="verdict-sub">Pick a sample scan below or drop your own.</p>
          </>
        )}
      </div>

      <dl className="stats">
        <div><dt>REGIONS</dt><dd>{result ? result.num_detections : "–"}</dd></div>
        <div><dt>INFERENCE</dt><dd>{result ? `${Math.round(result.inference_ms)} MS` : "–"}</dd></div>
        <div><dt>IMAGE</dt><dd>{result ? `${result.image_width}×${result.image_height}` : "–"}</dd></div>
      </dl>

      {result && result.detections.length > 0 && (
        <ol className="detections">
          {result.detections.map((d, i) => (
            <li key={i} className={`det det-${d.label}`}>
              <span className="det-label">{d.label === "positive" ? "Tumor" : "Healthy region"}</span>
              <span className="det-bar" aria-hidden="true"><span style={{ width: `${d.confidence * 100}%` }} /></span>
              <span className="det-value">{(d.confidence * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ol>
      )}

      <p className="disclaimer">Student project, not a medical device. Every result must be checked by a radiologist.</p>
    </div>
  );
}

import { isTumor } from "../api.js";

// Results panel, styled with Tailwind utilities (tokens from styles.css -> @theme inline).
const VERDICTS = {
  positive: { title: "TUMOR DETECTED", border: "border-lesion", text: "text-lesion" },
  negative: { title: "NO TUMOR DETECTED", border: "border-clear", text: "text-clear" },
  empty: { title: "NOTHING ABOVE THRESHOLD", border: "border-steel", text: "text-ice" },
};
const IDLE = { border: "border-steel", text: "text-ice" };
const LOADING = { border: "border-signal", text: "text-signal" };

const LABEL = "font-cond text-[13px] font-normal leading-normal tracking-[0.09em]";

function subline(result, verdict) {
  if (verdict === "positive")
    return `Highest tumor confidence ${(result.max_tumor_confidence * 100).toFixed(0)}%. A radiologist should review this region.`;
  if (verdict === "negative")
    return `The brain region was classified as healthy at ${(result.detections[0].confidence * 100).toFixed(0)}% confidence.`;
  return "No region passed the threshold. Lower it to see weaker detections.";
}

const ms = (v) => (v != null ? `${Math.round(v)} MS` : "–");

function Stat({ label, value, title }) {
  return (
    <div title={title} className="py-3.5 [&+&]:border-l [&+&]:border-hair [&+&]:pl-4">
      <dt className={`${LABEL} text-steel`}>{label}</dt>
      <dd className="m-0 mt-0.5 font-cond text-[26px] font-light tracking-[0.02em]">{value}</dd>
    </div>
  );
}

function DetectionBar({ detection, index }) {
  const tumor = isTumor(detection.label);
  const pct = detection.confidence * 100;
  return (
    <li className="grid grid-cols-[100px_1fr_50px] items-center gap-3.5 py-2 text-[15px] sm:grid-cols-[120px_1fr_56px]">
      <span className={tumor ? "text-lesion" : "text-ice"}>{tumor ? "Tumor" : "Healthy region"}</span>
      <span className="h-0.5 bg-hair" aria-hidden="true">
        <span
          className={`block h-full origin-left animate-grow ${tumor ? "bg-lesion" : "bg-clear"}`}
          style={{ width: `${pct}%`, animationDelay: `${index * 90}ms` }}
        />
      </span>
      <span className="text-right font-cond text-[17px] tabular-nums">{pct.toFixed(1)}%</span>
    </li>
  );
}

export default function Readout({ result, loading, conf, onConf }) {
  const verdict = !result ? null : result.num_detections === 0 ? "empty" : result.tumor_detected ? "positive" : "negative";
  const v = loading ? LOADING : verdict ? VERDICTS[verdict] : IDLE;

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
        <p className="mt-2.5 mb-0 text-sm text-steel">Lower catches more tumors, with more false alarms.</p>
      </div>

      <div className={`mt-10 mb-8 min-h-[120px] border-l-2 pl-[18px] transition-colors ${v.border}`} aria-live="polite">
        <p className={`m-0 mb-2.5 font-cond text-[clamp(34px,3.4vw,52px)] leading-[0.95] font-normal tracking-[0.01em] ${v.text}`}>
          {loading ? "ANALYSING" : verdict ? VERDICTS[verdict].title : "NO SCAN YET"}
        </p>
        <p className="m-0 max-w-[40ch] text-[15px] text-mist">
          {loading ? "Running the model on the scan." : verdict ? subline(result, verdict) : "Pick a sample scan below or drop your own."}
        </p>
      </div>

      <dl className="m-0 grid grid-cols-2 border-y border-hair sm:grid-cols-4">
        <Stat label="REGIONS" value={result ? result.num_detections : "–"} />
        <Stat label="SERVER" title="Model time measured by the API" value={ms(result?.inference_ms)} />
        <Stat label="ROUND TRIP" title="Full round trip measured in the browser: upload, inference and network"
              value={ms(result?.client_ms)} />
        <Stat label="IMAGE" value={result ? `${result.image_width}×${result.image_height}` : "–"} />
      </dl>

      {result && result.detections.length > 0 && (
        // key on the result so the bars replay their fill animation for every new scan
        <ol key={`${result.inference_ms}-${result.num_detections}`} className="mt-5 mb-0 list-none p-0">
          {result.detections.map((d, i) => (
            <DetectionBar key={i} detection={d} index={i} />
          ))}
        </ol>
      )}

      <p className="mt-7 mb-0 text-[13px] text-steel">
        Student project, not a medical device. Every result must be checked by a radiologist.
      </p>
    </div>
  );
}

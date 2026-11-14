import { useEffect, useRef, useState } from "react";
import { predict, verdictOf } from "../api.js";
import { drawDetections } from "../draw.js";
import { SAMPLES } from "../facts.js";
import Readout from "./Readout.jsx";
import WebcamStage from "./WebcamStage.jsx";

let counter = 0;
const nextId = () => `scan-${++counter}`;

export default function Scanner({ onBrainState, sectionRef }) {
  const [mode, setMode] = useState("upload");
  const [conf, setConf] = useState(0.25);
  const [scan, setScan] = useState(null);       // { id, src, blob, name }
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [history, setHistory] = useState([]);
  const inputRef = useRef(null);
  const overlayRef = useRef(null);
  const lastScanId = useRef(null);

  // Tell the brain what is happening
  useEffect(() => {
    if (loading) onBrainState("scanning");
    else onBrainState(verdictOf(result) || "idle");
  }, [loading, result, onBrainState]);

  // Analyse the current scan; re-run when the threshold moves
  useEffect(() => {
    if (!scan || mode !== "upload") return;
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await predict(scan.blob, conf, ctrl.signal);
        setResult(res);
        if (lastScanId.current !== scan.id) {
          lastScanId.current = scan.id;
          setHistory((h) => [{ id: scan.id, src: scan.src, name: scan.name, verdict: verdictOf(res) }, ...h].slice(0, 8));
        } else {
          setHistory((h) => h.map((it) => (it.id === scan.id ? { ...it, verdict: verdictOf(res) } : it)));
        }
      } catch (e) {
        if (e.name !== "AbortError") setError(e.message);
      } finally {
        setLoading(false);
      }
    }, 200);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [scan, conf, mode]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas || !result || mode !== "upload") return;
    canvas.width = result.image_width;
    canvas.height = result.image_height;
    drawDetections(canvas, result.detections, result.image_width, result.image_height);
  }, [result, mode]);

  function openFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      setError("Choose an image file (JPG or PNG).");
      return;
    }
    setResult(null);
    setScan({ id: nextId(), src: URL.createObjectURL(file), blob: file, name: file.name });
  }

  async function openSample(sample) {
    const blob = await (await fetch(sample.src)).blob();
    setResult(null);
    setScan({ id: nextId(), src: sample.src, blob, name: sample.label });
  }

  function reopen(item) {
    fetch(item.src).then((r) => r.blob()).then((blob) => {
      setResult(null);
      setScan({ id: item.id, src: item.src, blob, name: item.name });
    });
  }

  function switchMode(next) {
    setMode(next);
    setResult(null);
    setError(null);
  }

  return (
    <section className="scanner" ref={sectionRef} aria-labelledby="scanner-title">
      <div className="scanner-head">
        <h2 id="scanner-title">SCANNER</h2>
        <div className="mode-switch" role="tablist" aria-label="Input">
          <button role="tab" aria-selected={mode === "upload"} onClick={() => switchMode("upload")}>IMAGE</button>
          <button role="tab" aria-selected={mode === "webcam"} onClick={() => switchMode("webcam")}>WEBCAM</button>
        </div>
      </div>

      <div className="scanner-grid">
        <div className="stage-col">
          {mode === "upload" ? (
            <div
              className={`stage ${dragging ? "is-dragging" : ""} ${loading ? "is-scanning" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); openFile(e.dataTransfer.files[0]); }}
            >
              <span className="corner tl" /><span className="corner tr" />
              <span className="corner bl" /><span className="corner br" />
              {scan ? (
                <div className="stage-media">
                  <img src={scan.src} alt={`Scan ${scan.name}`} />
                  <canvas ref={overlayRef} className="overlay" />
                  <div className="scanline" aria-hidden="true" />
                </div>
              ) : (
                <button className="stage-empty" onClick={() => inputRef.current.click()}>
                  <span className="stage-empty-title">DROP A BRAIN SCAN HERE</span>
                  <span className="stage-empty-sub">or click to choose a file, JPG or PNG up to 10 MB</span>
                </button>
              )}
            </div>
          ) : (
            <WebcamStage conf={conf} onResult={setResult} onScanning={setLoading} />
          )}

          {mode === "upload" && (
            <div className="stage-actions">
              <button className="line-btn" onClick={() => inputRef.current.click()}>
                {scan ? "OPEN ANOTHER SCAN" : "CHOOSE A FILE"}
              </button>
              {scan && <span className="file-name">{scan.name}</span>}
            </div>
          )}
          {error && <p className="error" role="alert">{error}</p>}
          <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => openFile(e.target.files[0])} />
        </div>

        <Readout result={result} loading={loading} conf={conf} onConf={setConf} />
      </div>

      {mode === "upload" && (
        <div className="strips">
          <div className="strip">
            <h3>SAMPLE SCANS</h3>
            <p className="strip-hint">Validation images the model did not train on.</p>
            <ul className="thumbs">
              {SAMPLES.map((s) => (
                <li key={s.src}>
                  <button className={`thumb ${scan?.src === s.src ? "is-active" : ""}`} onClick={() => openSample(s)}>
                    <img src={s.src} alt="" loading="lazy" />
                    <span>{s.label}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {history.length > 0 && (
            <div className="strip">
              <h3>THIS SESSION</h3>
              <ul className="thumbs">
                {history.map((h) => (
                  <li key={h.id}>
                    <button className={`thumb verdict-${h.verdict}`} onClick={() => reopen(h)}>
                      <img src={h.src} alt="" />
                      <span>{{ positive: "Tumor", negative: "No tumor", empty: "Nothing" }[h.verdict]}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

import { useEffect, useRef, useState } from "react";
import { predict } from "../api.js";
import { drawDetections } from "../draw.js";

export default function UploadView({ conf, onResult }) {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const inputRef = useRef(null);

  function pick(f) {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    onResult(null);
    setImageUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
  }

  // (Re)run the prediction when the image or the confidence threshold changes
  useEffect(() => {
    if (!file) return;
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await predict(file, conf, ctrl.signal);
        setResult(res);
        onResult(res);
      } catch (e) {
        if (e.name !== "AbortError") setError(e.message);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [file, conf]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !result) return;
    canvas.width = result.image_width;
    canvas.height = result.image_height;
    drawDetections(canvas, result.detections, result.image_width, result.image_height);
  }, [result]);

  return (
    <div>
      {!imageUrl && (
        <div
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]); }}
        >
          <div className="dropzone-icon">🧠</div>
          <p><strong>Drop an MRI scan here</strong> or click to browse</p>
          <p className="muted">JPG or PNG, up to 10 MB</p>
        </div>
      )}

      {imageUrl && (
        <>
          <div className="stage">
            <img ref={imgRef} src={imageUrl} alt="scan" />
            <canvas ref={canvasRef} className="overlay" />
            {loading && <div className="stage-loading">Analyzing…</div>}
          </div>
          <button className="btn secondary" onClick={() => inputRef.current.click()}>
            Choose another image
          </button>
        </>
      )}

      {error && <p className="error">⚠️ {error}</p>}
      <input ref={inputRef} type="file" accept="image/*" hidden
             onChange={(e) => pick(e.target.files[0])} />
    </div>
  );
}

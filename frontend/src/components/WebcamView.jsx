import { useEffect, useRef, useState } from "react";
import { predict } from "../api.js";
import { drawDetections } from "../draw.js";

// Streams webcam frames to /predict one after another (next frame sent when the
// previous answer arrives), so the API is never flooded.
export default function WebcamView({ conf, onResult }) {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const confRef = useRef(conf);
  const runningRef = useRef(false);
  const [running, setRunning] = useState(false);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => { confRef.current = conf; }, [conf]);
  useEffect(() => () => stop(), []); // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      runningRef.current = true;
      setRunning(true);
      loop();
    } catch (e) {
      setError(`Cannot access webcam: ${e.message}`);
    }
  }

  function stop() {
    runningRef.current = false;
    setRunning(false);
    const video = videoRef.current;
    if (video?.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    const overlay = overlayRef.current;
    overlay?.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  }

  async function loop() {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const grab = document.createElement("canvas");

    while (runningRef.current) {
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w) { await new Promise((r) => setTimeout(r, 100)); continue; }
      grab.width = w;
      grab.height = h;
      grab.getContext("2d").drawImage(video, 0, 0, w, h);
      const blob = await new Promise((r) => grab.toBlob(r, "image/jpeg", 0.8));

      const t0 = performance.now();
      try {
        const res = await predict(blob, confRef.current);
        if (!runningRef.current) break;
        overlay.width = w;
        overlay.height = h;
        drawDetections(overlay, res.detections, res.image_width, res.image_height);
        onResult(res);
        setFps(1000 / (performance.now() - t0));
      } catch (e) {
        setError(e.message);
        await new Promise((r) => setTimeout(r, 1000));
      }
    }
  }

  return (
    <div>
      <div className="stage webcam">
        <video ref={videoRef} muted playsInline />
        <canvas ref={overlayRef} className="overlay" />
        {!running && (
          <div className="stage-placeholder">
            <p>📷 Point the camera at a printed or on-screen MRI scan</p>
          </div>
        )}
        {running && <div className="fps">{fps.toFixed(1)} FPS</div>}
      </div>
      {!running
        ? <button className="btn" onClick={start}>Start webcam</button>
        : <button className="btn danger" onClick={stop}>Stop</button>}
      {error && <p className="error">⚠️ {error}</p>}
    </div>
  );
}

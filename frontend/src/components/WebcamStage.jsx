import { useEffect, useRef, useState } from "react";
import { predict } from "../api.js";
import { drawDetections } from "../draw.js";

// Streams webcam frames to /predict one after another: the next frame is sent
// only when the previous answer has arrived, so the API is never flooded.
export default function WebcamStage({ conf, onResult, onScanning }) {
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
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      runningRef.current = true;
      setRunning(true);
      loop();
    } catch (e) {
      setError(`The webcam could not start: ${e.message}. Allow camera access in the browser and try again.`);
    }
  }

  function stop() {
    runningRef.current = false;
    setRunning(false);
    onScanning(false);
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
      const w = video.videoWidth, h = video.videoHeight;
      if (!w) { await new Promise((r) => setTimeout(r, 100)); continue; }
      grab.width = w; grab.height = h;
      grab.getContext("2d").drawImage(video, 0, 0, w, h);
      const blob = await new Promise((r) => grab.toBlob(r, "image/jpeg", 0.8));
      const t0 = performance.now();
      try {
        const res = await predict(blob, confRef.current);
        if (!runningRef.current) break;
        overlay.width = w; overlay.height = h;
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
    <>
      <div className={`stage webcam ${running ? "is-live" : ""}`}>
        <span className="corner tl" /><span className="corner tr" />
        <span className="corner bl" /><span className="corner br" />
        <div className="stage-media">
          <video ref={videoRef} muted playsInline />
          <canvas ref={overlayRef} className="overlay" />
        </div>
        {!running && (
          <button className="stage-empty" onClick={start}>
            <span className="stage-empty-title">START THE WEBCAM</span>
            <span className="stage-empty-sub">Hold a printed scan or a phone showing one in front of the camera</span>
          </button>
        )}
        {running && <span className="live-badge"><span className="live-dot" />LIVE {fps.toFixed(1)} FPS</span>}
      </div>
      <div className="stage-actions">
        {running
          ? <button className="line-btn" onClick={stop}>STOP THE WEBCAM</button>
          : <button className="line-btn" onClick={start}>START THE WEBCAM</button>}
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth } from "./api.js";
import BrainCanvas from "./components/BrainCanvas.jsx";
import Hero from "./components/Hero.jsx";
import Scanner from "./components/Scanner.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [brainState, setBrainState] = useState("idle");
  const [focus, setFocus] = useState(0);
  const scannerRef = useRef(null);

  useEffect(() => {
    const check = () => getHealth().then(setHealth).catch(() => setHealth(null));
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  // 0 on the landing screen, 1 once the scanner fills the view:
  // the brain slides behind the scanner and dims.
  useEffect(() => {
    const onScroll = () => setFocus(Math.min(1, window.scrollY / (window.innerHeight * 0.8)));
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const onBrainState = useCallback((s) => setBrainState(s), []);
  const status = !health ? "offline" : health.model_loaded ? "ready" : "nomodel";
  const start = () => scannerRef.current?.scrollIntoView({ behavior: "smooth" });

  return (
    <>
      <BrainCanvas state={brainState} focus={focus} />
      <main className="page">
        <Hero status={status} onStart={start} />
        <Scanner sectionRef={scannerRef} onBrainState={onBrainState} />
      </main>
    </>
  );
}

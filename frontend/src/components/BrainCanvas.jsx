import { useEffect, useRef } from "react";

/*
 * A procedural 3D "neural" brain drawn on a 2D canvas.
 * Points are sampled inside a brain-shaped volume (cerebrum, cerebellum, stem),
 * linked to their nearest neighbours, slowly rotated and projected in perspective.
 * Signals travel along the links. The brain reacts to the analysis state:
 *   idle      calm blue
 *   scanning  faster rotation, more signals
 *   positive  a red lesion pulses inside the brain
 *   negative  cool, steady glow
 *   empty     dimmed
 */

const BLUE = [46, 123, 255];
const ICE = [190, 215, 255];
const RED = [255, 69, 96];
const MINT = [92, 225, 230];

// Side profile, front of the head = +x. Each part is sampled with its own
// point budget so small structures (cerebellum, stem) stay visible.
const PARTS = [
  { // cerebrum: dome, flat underneath, fissure between hemispheres, lateral sulcus
    share: 0.62, box: [-1, 1, -0.2, 0.82, -0.8, 0.8],
    test: (x, y, z) => {
      const az = Math.abs(z);
      if ((x / 1.0) ** 2 + ((y - 0.14) / 0.64) ** 2 + (z / 0.78) ** 2 >= 1) return false;
      if (az < 0.035 && y > 0.12) return false;
      return true;
    },
  },
  { // temporal lobes, below the lateral sulcus
    share: 0.16, box: [-0.5, 0.75, -0.5, 0.05, -0.75, 0.75],
    test: (x, y, z) => ((x - 0.12) / 0.6) ** 2 + ((y + 0.22) / 0.25) ** 2 + ((Math.abs(z) - 0.42) / 0.32) ** 2 < 1,
  },
  { // cerebellum, tucked under the back, finely folded
    share: 0.13, box: [-0.98, -0.22, -0.64, -0.2, -0.5, 0.5],
    test: (x, y, z) => ((x + 0.6) / 0.36) ** 2 + ((y + 0.42) / 0.2) ** 2 + (z / 0.48) ** 2 < 1,
  },
  { // brain stem
    share: 0.09, box: [-0.33, -0.07, -1.0, -0.15, -0.14, 0.14],
    test: (x, y, z) => (x + 0.2) ** 2 + (z / 1.2) ** 2 < 0.11 ** 2,
  },
];

// The lateral sulcus: a gap on the side of the brain separating the temporal lobe.
function inSulcus(x, y, z) {
  if (x < -0.35 || x > 0.62 || Math.abs(z) < 0.3) return false;
  const line = -0.1 + (0.62 - x) * 0.26;
  return Math.abs(y - line) < 0.035;
}

function isSurface(part, x, y, z, rand) {
  // keep most points near the outside of each part: the surface carries the silhouette
  const k = 1.12;
  const cx = (part.box[0] + part.box[1]) / 2, cy = (part.box[2] + part.box[3]) / 2, cz = 0;
  const deep = part.test(cx + (x - cx) * k, cy + (y - cy) * k, cz + (z - cz) * k);
  return !deep || rand() < 0.2;
}

function buildBrain(count, rand) {
  const pts = [];
  for (const part of PARTS) {
    const target = Math.round(count * part.share);
    const [x0, x1, y0, y1, z0, z1] = part.box;
    let n = 0, guard = 0;
    while (n < target && guard++ < target * 400) {
      const x = x0 + rand() * (x1 - x0);
      const y = y0 + rand() * (y1 - y0);
      const z = z0 + rand() * (z1 - z0);
      if (!part.test(x, y, z) || inSulcus(x, y, z) || !isSurface(part, x, y, z, rand)) continue;
      pts.push({ x, y, z, phase: rand() * Math.PI * 2, size: 0.6 + rand() * 1.1 });
      n++;
    }
  }
  // links to the 3 nearest neighbours
  const edges = [];
  for (let i = 0; i < pts.length; i++) {
    const near = [];
    for (let j = 0; j < pts.length; j++) {
      if (i === j) continue;
      const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y, dz = pts[i].z - pts[j].z;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < 0.025) near.push([d, j]);
    }
    near.sort((a, b) => a[0] - b[0]);
    for (const [, j] of near.slice(0, 3)) {
      if (j > i) {
        edges.push({
          a: i, b: j,
          // control point offset -> slightly curved, fibre-like links
          cx: (rand() - 0.5) * 0.05, cy: (rand() - 0.5) * 0.05, cz: (rand() - 0.5) * 0.05,
        });
      }
    }
  }
  return { pts, edges };
}

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

export default function BrainCanvas({ state = "idle", focus = 0 }) {
  const canvasRef = useRef(null);
  const stateRef = useRef(state);
  const focusRef = useRef(focus);
  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { focusRef.current = focus; }, [focus]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const rand = mulberry32(7);
    const small = window.innerWidth < 700;
    const { pts, edges } = buildBrain(small ? 1000 : 1900, rand);
    const stars = Array.from({ length: small ? 60 : 140 }, () => ({
      x: rand(), y: rand(), r: rand() * 1.2 + 0.2, a: rand() * 0.5 + 0.1, s: rand() * 0.00004 + 0.00001,
    }));
    // lesion: a cluster in the upper-front cerebrum, only lit when a tumor is found
    const lesion = { x: 0.35, y: 0.32, z: 0.28 };
    pts.forEach((p) => {
      const d = Math.hypot(p.x - lesion.x, p.y - lesion.y, p.z - lesion.z);
      p.lesion = Math.max(0, 1 - d / 0.38);
    });
    const pulses = Array.from({ length: 70 }, () => ({ e: Math.floor(rand() * edges.length), t: rand() }));

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, dpr = 1, raf = 0, phase = 0, last = performance.now();
    // smoothed animation parameters
    const anim = { speed: 0.12, lesion: 0, glow: 1, focus: focusRef.current, color: BLUE };

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth; h = canvas.clientHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const proj = new Array(pts.length);

    function frame(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const s = stateRef.current;
      const target = {
        speed: s === "scanning" ? 0.9 : s === "empty" ? 0.06 : 0.12,
        lesion: s === "positive" ? 1 : 0,
        glow: s === "empty" ? 0.45 : s === "scanning" ? 1.35 : 1,
        color: s === "negative" ? MINT : s === "scanning" ? ICE : BLUE,
      };
      const k = 1 - Math.pow(0.02, dt);
      anim.speed += (target.speed - anim.speed) * k;
      anim.lesion += (target.lesion - anim.lesion) * k;
      anim.glow += (target.glow - anim.glow) * k;
      anim.focus += (focusRef.current - anim.focus) * k;
      anim.color = mix(anim.color, target.color, k);
      // gentle yaw around the side profile so the silhouette stays readable
      if (!reduced) phase += anim.speed * dt * 2.2;
      const angle = 0.55 * Math.sin(phase) - 0.15;

      ctx.clearRect(0, 0, w, h);

      // star dust
      for (const st of stars) {
        if (!reduced) st.y = (st.y - st.s * 300 * dt + 1) % 1;
        ctx.fillStyle = `rgba(150,180,255,${st.a * 0.6})`;
        ctx.fillRect(st.x * w, st.y * h, st.r, st.r);
      }

      // brain placement: right of the headline in the hero, centred behind the scanner
      const f = anim.focus;
      const narrow = w < 860;
      const cx = w * (narrow ? 0.55 : 0.6 - 0.1 * f);
      const cy = h * (narrow ? 0.4 : 0.5 + 0.03 * f);
      const scale = Math.min(w * (narrow ? 0.4 : 0.27), h * 0.42) * (1 - 0.1 * f);
      const ca = Math.cos(angle), sa = Math.sin(angle);
      const tilt = 0.18 + Math.sin(now * 0.0003) * 0.05;
      const ct = Math.cos(tilt), stt = Math.sin(tilt);

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        let x = p.x * ca - p.z * sa;
        let z = p.x * sa + p.z * ca;
        const y = p.y * ct - z * stt;
        z = p.y * stt + z * ct;
        const persp = 3 / (3 - z);
        proj[i] = { x: cx + x * scale * persp, y: cy - y * scale * persp, z, persp };
      }

      ctx.globalCompositeOperation = "lighter";
      const [r, g, b] = anim.color;
      const fade = (1 - 0.5 * f) * (narrow ? 0.65 : 1);

      // soft glow behind the brain
      const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, scale * 1.3);
      halo.addColorStop(0, `rgba(${anim.color[0]},${anim.color[1]},${anim.color[2]},${0.13 * anim.glow * fade})`);
      halo.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = halo;
      ctx.fillRect(cx - scale * 1.3, cy - scale * 1.3, scale * 2.6, scale * 2.6);

      // links, in two depth layers
      for (const front of [false, true]) {
        ctx.beginPath();
        for (const e of edges) {
          const A = proj[e.a], B = proj[e.b];
          if ((A.z + B.z > 0) !== front) continue;
          ctx.moveTo(A.x, A.y);
          ctx.quadraticCurveTo(
            (A.x + B.x) / 2 + e.cx * scale, (A.y + B.y) / 2 + e.cy * scale, B.x, B.y);
        }
        if (front) {
          ctx.strokeStyle = `rgba(${r},${g},${b},${0.07 * anim.glow * fade})`;
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        ctx.strokeStyle = `rgba(${r},${g},${b},${(front ? 0.42 : 0.14) * anim.glow * fade})`;
        ctx.lineWidth = front ? 0.9 : 0.6;
        ctx.stroke();
      }

      // nodes
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i], q = proj[i];
        const tw = reduced ? 0.7 : 0.55 + 0.45 * Math.sin(now * 0.002 + p.phase);
        const depth = (q.z + 1) / 2;
        const lesionMix = p.lesion * anim.lesion;
        const [nr, ng, nb] = lesionMix > 0.01 ? mix(anim.color, RED, Math.min(1, lesionMix * 1.6)) : anim.color;
        const pulse = lesionMix > 0 ? 1 + 0.8 * lesionMix * (0.5 + 0.5 * Math.sin(now * 0.006)) : 1;
        ctx.fillStyle = `rgba(${nr},${ng},${nb},${(0.25 + 0.6 * depth) * tw * anim.glow * fade})`;
        const size = p.size * q.persp * pulse * (lesionMix > 0.3 ? 1.8 : 1);
        ctx.fillRect(q.x - size / 2, q.y - size / 2, size, size);
      }

      // lesion halo
      if (anim.lesion > 0.02) {
        let lx = lesion.x * ca - lesion.z * sa;
        let lz = lesion.x * sa + lesion.z * ca;
        const ly = lesion.y * ct - lz * stt;
        lz = lesion.y * stt + lz * ct;
        const lp = 3 / (3 - lz);
        const hx = cx + lx * scale * lp, hy = cy - ly * scale * lp;
        const rad = scale * 0.32 * (1 + 0.08 * Math.sin(now * 0.006));
        const grad = ctx.createRadialGradient(hx, hy, 0, hx, hy, rad);
        grad.addColorStop(0, `rgba(255,69,96,${0.35 * anim.lesion})`);
        grad.addColorStop(1, "rgba(255,69,96,0)");
        ctx.fillStyle = grad;
        ctx.fillRect(hx - rad, hy - rad, rad * 2, rad * 2);
      }

      // signals travelling along the links
      if (!reduced) {
        const active = s === "scanning" ? pulses.length : 26;
        for (let i = 0; i < active; i++) {
          const pu = pulses[i];
          pu.t += dt * (s === "scanning" ? 2.6 : 0.9);
          if (pu.t >= 1) { pu.t = 0; pu.e = Math.floor(Math.random() * edges.length); }
          const e = edges[pu.e], A = proj[e.a], B = proj[e.b];
          const t = pu.t, mt = 1 - t;
          const qx = (A.x + B.x) / 2 + e.cx * scale, qy = (A.y + B.y) / 2 + e.cy * scale;
          const px = mt * mt * A.x + 2 * mt * t * qx + t * t * B.x;
          const py = mt * mt * A.y + 2 * mt * t * qy + t * t * B.y;
          ctx.fillStyle = `rgba(220,235,255,${0.9 * fade})`;
          ctx.fillRect(px - 1.2, py - 1.2, 2.4, 2.4);
        }
      }

      ctx.globalCompositeOperation = "source-over";
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={canvasRef} className="brain-canvas" aria-hidden="true" />;
}

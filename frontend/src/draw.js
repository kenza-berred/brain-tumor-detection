// Draws the predicted boxes on a canvas laid over the image or video.
// Viewfinder-style corners instead of full rectangles, to match the interface.
const COLORS = { positive: "#ff4560", negative: "#5ce1e6" };

export function drawDetections(canvas, detections, srcWidth, srcHeight) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (!detections) return;

  const sx = width / srcWidth;
  const sy = height / srcHeight;
  const unit = Math.max(1.5, width / 300);
  const font = Math.max(12, Math.round(width / 38));

  for (const d of detections) {
    const color = COLORS[d.label] || "#2e7bff";
    const x = d.box.x1 * sx;
    const y = d.box.y1 * sy;
    const w = (d.box.x2 - d.box.x1) * sx;
    const h = (d.box.y2 - d.box.y1) * sy;
    const c = Math.min(w, h) * 0.28;

    // faint full frame + strong corners
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = unit * 0.6;
    ctx.strokeRect(x, y, w, h);
    ctx.globalAlpha = 1;
    ctx.lineWidth = unit * 1.6;
    ctx.beginPath();
    for (const [px, py, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
      ctx.moveTo(px + dx * c, py);
      ctx.lineTo(px, py);
      ctx.lineTo(px, py + dy * c);
    }
    ctx.stroke();

    // soft fill for tumors
    if (d.label === "positive") {
      ctx.fillStyle = "rgba(255,69,96,0.10)";
      ctx.fillRect(x, y, w, h);
    }

    const text = `${d.label === "positive" ? "TUMOR" : "NO TUMOR"} ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = `500 ${font}px "Barlow Condensed", sans-serif`;
    const tw = ctx.measureText(text).width + font * 0.8;
    const ty = y > font * 1.6 ? y - font * 1.5 : y + h + font * 0.3;
    ctx.fillStyle = color;
    ctx.fillRect(x, ty, tw, font * 1.3);
    ctx.fillStyle = "#000";
    ctx.fillText(text, x + font * 0.4, ty + font * 0.98);
  }
}

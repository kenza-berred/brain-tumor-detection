// Draws the predicted bounding boxes on a canvas laid over the image/video.
const COLORS = { positive: "#ef4444", negative: "#22c55e" };

export function drawDetections(canvas, detections, srcWidth, srcHeight) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (!detections) return;

  const sx = width / srcWidth;
  const sy = height / srcHeight;
  const font = Math.max(12, Math.round(width / 45));

  for (const d of detections) {
    const color = COLORS[d.label] || "#3b82f6";
    const x = d.box.x1 * sx;
    const y = d.box.y1 * sy;
    const w = (d.box.x2 - d.box.x1) * sx;
    const h = (d.box.y2 - d.box.y1) * sy;

    ctx.lineWidth = Math.max(2, width / 250);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const text = `${d.label === "positive" ? "Tumor" : "No tumor"} ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = `600 ${font}px system-ui, sans-serif`;
    const tw = ctx.measureText(text).width + 10;
    const ty = y > font + 6 ? y - font - 6 : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, ty, tw, font + 6);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + 5, ty + font);
  }
}

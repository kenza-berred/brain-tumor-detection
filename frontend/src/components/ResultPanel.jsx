export default function ResultPanel({ result }) {
  if (!result) {
    return (
      <div className="panel empty">
        <p className="muted">Results will appear here.</p>
      </div>
    );
  }

  const positive = result.tumor_detected;
  return (
    <div className="panel">
      <div className={`verdict ${positive ? "positive" : "negative"}`}>
        <span className="verdict-icon">{positive ? "⚠️" : "✅"}</span>
        <div>
          <div className="verdict-title">{positive ? "Tumor detected" : "No tumor detected"}</div>
          {positive && (
            <div className="verdict-sub">
              Max confidence: {(result.max_tumor_confidence * 100).toFixed(1)}%
            </div>
          )}
        </div>
      </div>

      <div className="stats">
        <div><span className="stat-value">{result.num_detections}</span><span className="stat-label">detections</span></div>
        <div><span className="stat-value">{result.inference_ms.toFixed(0)} ms</span><span className="stat-label">inference</span></div>
        <div><span className="stat-value">{result.image_width}×{result.image_height}</span><span className="stat-label">image</span></div>
      </div>

      {result.detections.length > 0 && (
        <table className="det-table">
          <thead><tr><th>#</th><th>Class</th><th>Confidence</th></tr></thead>
          <tbody>
            {result.detections.map((d, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td><span className={`tag ${d.label}`}>{d.label}</span></td>
                <td>
                  <div className="bar"><div className={`bar-fill ${d.label}`} style={{ width: `${d.confidence * 100}%` }} /></div>
                  {(d.confidence * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

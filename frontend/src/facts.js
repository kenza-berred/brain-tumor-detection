// Numbers shown on the landing screen. Update them from reports/metrics.json
// and reports/export.json if you retrain the model.
export const MODEL_FACTS = {
  model: "YOLO11N",
  recall: "84%",          // reports/export.json -> onnx_recall
  mAP50: "0.49",          // reports/export.json -> onnx_mAP50
  latency: "43 MS",       // reports/export.json -> onnx_cpu_ms
  trainImages: 893,
};

export const SAMPLES = [
  { src: "/samples/tumor-1.jpg", label: "Scan A" },
  { src: "/samples/clear-1.jpg", label: "Scan B" },
  { src: "/samples/tumor-2.jpg", label: "Scan C" },
  { src: "/samples/clear-2.jpg", label: "Scan D" },
  { src: "/samples/tumor-3.jpg", label: "Scan E" },
  { src: "/samples/clear-3.jpg", label: "Scan F" },
  { src: "/samples/tumor-4.jpg", label: "Scan G" },
  { src: "/samples/clear-4.jpg", label: "Scan H" },
];

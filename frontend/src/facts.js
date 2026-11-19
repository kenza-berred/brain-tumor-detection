// Numbers shown on the landing screen. They are read live from GET /model/metadata
// (models/metadata.json, written by `dvc repro`); these values are only the fallback
// shown while the API is offline.
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

// Turn the API's model card into the landing-screen numbers (fallback for any missing field)
export function factsFromMetadata(meta) {
  if (!meta) return MODEL_FACTS;
  const recall = meta.validation_metrics?.recall;
  const ms = meta.benchmark?.fp32_cpu_ms ?? meta.benchmark?.onnx_cpu_ms;
  const train = /(\d+)\s*train/.exec(meta.dataset || "");
  return {
    model: (meta.architecture || MODEL_FACTS.model).toUpperCase(),
    recall: recall != null ? `${Math.round(recall * 100)}%` : MODEL_FACTS.recall,
    mAP50: meta.validation_metrics?.mAP50?.toFixed(2) ?? MODEL_FACTS.mAP50,
    latency: ms != null ? `${Math.round(ms)} MS` : MODEL_FACTS.latency,
    trainImages: train ? Number(train[1]) : MODEL_FACTS.trainImages,
  };
}

"""Baseline vs advanced comparison table, built from the MLflow runs.

Run with:  python -m src.utils.compare_runs
Writes:    reports/comparison.md   (Markdown table, pasted in the README / report)
           reports/comparison.csv
           reports/comparison.png  (mAP50 / mAP50-95 per run)

Only FINISHED runs are kept; the newest run of each name wins.
"""

from __future__ import annotations

import csv
import math

from src.utils.common import ROOT, load_params, setup_mlflow

COLUMNS = ["run", "model", "epochs", "augment", "precision", "recall", "mAP50", "mAP50_95", "minutes"]


def collect() -> list[dict]:
    params = load_params()
    mlflow = setup_mlflow(params["mlflow"]["experiment"])
    runs = mlflow.search_runs(
        experiment_names=[params["mlflow"]["experiment"]],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    rows, seen = [], set()
    for _, r in runs.iterrows():
        name = r["tags.mlflow.runName"]
        score = r.get("metrics.final_mAP50_95")
        if name in seen or score is None or math.isnan(score):
            continue
        seen.add(name)
        minutes = (r["end_time"] - r["start_time"]).total_seconds() / 60
        rows.append(
            {
                "run": name,
                "model": str(r["params.model"]).replace(".pt", ""),
                "epochs": r["params.epochs"],
                "augment": r["params.augment"],
                "precision": round(r["metrics.final_precision"], 3),
                "recall": round(r["metrics.final_recall"], 3),
                "mAP50": round(r["metrics.final_mAP50"], 3),
                "mAP50_95": round(r["metrics.final_mAP50_95"], 3),
                "minutes": round(minutes, 1),
            }
        )
    return sorted(rows, key=lambda x: x["mAP50_95"])


def write(rows: list[dict]) -> None:
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(str(r[c]) for c in COLUMNS) + " |" for r in rows]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(rows) + 1.2))
    y = range(len(rows))
    for offset, key, color in ((0.2, "mAP50", "#9db4d8"), (-0.2, "mAP50_95", "#2e7bff")):
        ax.barh([i + offset for i in y], [r[key] for r in rows], height=0.38, color=color, label=key)
    for i, r in enumerate(rows):
        ax.text(r["mAP50_95"] + 0.005, i - 0.2, f"{r['mAP50_95']:.3f}", va="center", fontsize=8)
    ax.set_yticks(list(y), [r["run"] for r in rows])
    ax.set_xlabel("Validation score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    print("\n".join(lines))


if __name__ == "__main__":
    write(collect())

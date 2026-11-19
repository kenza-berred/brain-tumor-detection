"""Shared helpers for the training pipeline."""

import os
from pathlib import Path

import yaml

# src/utils/common.py -> project root
ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment: str):
    """Point MLflow at the tracking server (docker compose service `mlflow`).

    Override with the MLFLOW_TRACKING_URI environment variable, e.g.
    `sqlite:///mlflow.db` when training without the server (Colab).
    """
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if uri.startswith("http"):
        import urllib.request

        try:
            urllib.request.urlopen(f"{uri}/health", timeout=5)
        except OSError as e:
            raise SystemExit(
                f"MLflow server not reachable at {uri}.\n"
                "Start it first:  docker compose up -d mlflow\n"
                "or train without it:  MLFLOW_TRACKING_URI=sqlite:///mlflow.db"
            ) from e
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow

# 🧠 NeuroScan — Détection de tumeurs cérébrales (YOLO11 + MLOps)

Projet de fin de module **Industrialisation Vision & MLOps** — Axe 3 : détection d'objets en temps réel.

Un modèle **YOLO11** pré-entraîné sur COCO est *fine-tuné* (transfer learning) sur des IRM/CT
cérébrales pour localiser les tumeurs. Le modèle est exporté en **ONNX**, servi par une API
**FastAPI** asynchrone et utilisé depuis une application **React** (upload d'image + webcam).

> ⚠️ Projet pédagogique — ce n'est pas un dispositif médical.

## Architecture

```
            ┌──────────── Pipeline d'entraînement (dvc repro) ────────────┐
            │  prepare_data.py ──► train.py ──────────► export.py          │
            │  (dataset, DVC)     (YOLO11, MLflow)      (ONNX, benchmark,  │
            │                                            Model Registry)   │
            └───────────────────────────────┬──────────────────────────────┘
                                            │ models/best.onnx
   docker compose up                        ▼
   ┌───────────────┐   /api/*   ┌──────────────────────┐     ┌──────────────┐
   │ React + nginx │ ─────────► │ FastAPI + ONNX Runtime│     │ MLflow server│
   │  :3000        │            │  :8000  (/docs)       │     │  :5000       │
   └───────────────┘            └──────────────────────┘     └──────────────┘
```

| Exigence | Où |
|---|---|
| FastAPI async + Pydantic + Swagger | `backend/app/main.py`, `schemas.py` → http://localhost:8000/docs |
| React : upload, webcam, boxes | `frontend/src/` |
| DVC (versioning données + pipeline) | `dvc.yaml`, `params.yaml` |
| MLflow (tracking + Model Registry) | `src/train.py`, `src/export.py` |
| ONNX (optimisation / déploiement) | `src/export.py`, `backend/app/inference.py` |
| Docker Compose | `docker-compose.yml` |
| GitHub Actions CI/CD | `.github/workflows/ci.yml` |

## Dataset

[Ultralytics Brain Tumor](https://docs.ultralytics.com/datasets/detect/brain-tumor/) :
1 116 images IRM/CT (893 train / 223 val), 2 classes : `negative`, `positive`. Téléchargé
automatiquement par `src/prepare_data.py`, puis versionné par DVC.

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt

git init
dvc init
```

> **GPU NVIDIA** : installe d'abord PyTorch avec CUDA depuis https://pytorch.org
> (sinon l'entraînement tourne sur CPU, beaucoup plus lent).

## 1. Entraîner (pipeline DVC + MLflow)

```bash
docker compose up -d mlflow      # serveur MLflow sur http://localhost:5000
dvc repro                        # prepare -> train -> export
```

Ce que produit `dvc repro` :
- `data/brain-tumor/` — dataset (versionné DVC)
- `models/best.pt`, `models/best.onnx`, `models/labels.json`
- `reports/metrics.json` (mAP, precision, recall) et `reports/export.json` (ONNX vs PyTorch)
- un run MLflow avec courbes, matrice de confusion, poids, et le modèle ONNX enregistré
  dans le **Model Registry** sous `brain-tumor-detector`

### Comparer des expériences

Modifie `params.yaml`, relance, compare dans MLflow :

```bash
dvc exp run -S train.model=yolo11s.pt         # modèle plus gros
dvc exp run -S train.augment=false            # sans augmentation
dvc exp show                                  # tableau comparatif côté DVC
dvc metrics diff                              # différences de métriques
```

### Sans serveur MLflow (ex. Colab)

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db dvc repro          # Linux / Mac / Colab
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"; dvc repro  # Windows PowerShell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 2. Lancer l'application

```bash
docker compose up --build
```

- Application : http://localhost:3000
- API Swagger : http://localhost:8000/docs
- MLflow : http://localhost:5000

L'API lit `models/best.onnx` monté en volume : il faut avoir lancé `dvc repro` avant.

### Mode développement (sans Docker)

```bash
cd backend && pip install -r requirements-dev.txt
MODEL_PATH=../models/best.onnx LABELS_PATH=../models/labels.json uvicorn app.main:app --reload
# PowerShell : $env:MODEL_PATH="../models/best.onnx"; $env:LABELS_PATH="../models/labels.json"; uvicorn app.main:app --reload
cd frontend && npm install && npm run dev      # http://localhost:5173
```

## 3. Tests & CI

```bash
cd backend && ruff check . && pytest -v
```

La CI GitHub Actions lance à chaque push : lint + tests backend, build frontend,
validation du pipeline DVC (`dvc dag`), build des images Docker et smoke test de l'API.

## Choix techniques (à savoir expliquer)

- **Transfer learning** : poids COCO → seulement ~900 images suffisent.
- **YOLO11n** : petit, rapide, adapté au temps réel et au CPU.
- **ONNX Runtime** : l'image API n'embarque ni PyTorch ni Ultralytics (image plus légère),
  inférence plus rapide sur CPU (voir `reports/export.json`). Pré/post-traitement
  (letterbox, décodage, NMS) réécrits en NumPy dans `backend/app/inference.py`.
- **NMS class-agnostic** : une région est soit `positive` soit `negative`, jamais les deux.
- **Recall > precision** : rater une tumeur (faux négatif) est plus grave qu'une fausse alerte
  → le slider de seuil dans l'UI permet de privilégier le recall.
- **Limite connue** : le split train/val n'est pas fait par patient → risque de fuite de données
  (scores possiblement optimistes).

# VietToxic Detector

Research + demo system for Vietnamese toxic-content detection from URLs.

## What is active right now

This repo currently runs a **web UI → FastAPI → comment-only crawl → local inference** flow.

Active runtime path:

1. User submits URLs in the React UI.
2. Backend crawls **comment sections only** (via `comment_crawl.py` through `backend/crawl_adapter.py`).
3. Backend runs local model inference (`infer_crawled_local.py`).
4. UI shows page-level and segment-level toxicity results.

The old article/video crawl lane (`setup_and_crawl.py`) is still in the repo, but it is **not** the active crawl adapter path.

---

## Architecture

- **Frontend**: Vite + React 18 + TypeScript (`comprehensive_ui/`)
- **Backend**: FastAPI (`backend/app.py`)
- **Inference**: local PhoBERT / TF-IDF-LR (`infer_crawled_local.py`)
- **Crawler in active path**: `comment_crawl.py`
- **Feedback / metadata storage**: SQLite (`data/processed/feedback/feedback.db`)

---

## Current frontend wiring (actual)

### App shell and navigation

Main app: `comprehensive_ui/src/app/App.tsx`

- Primary pages in top nav: `home`, `results`, `dataset`, `model`, `contact`, `admin_mlflow`
- `dataset_synthetic` route exists in app shell but is **not shown in top navigation**
- `protocol` route is reachable from Dataset page CTA (legacy dataset mode)
- Theme and language are persisted in localStorage

### Analysis defaults sent by UI

`App.tsx` currently sends:

- `batch_size: 8`
- `max_length: 256`
- `page_threshold: 0.25`
- `seg_threshold: 0.4`
- `enable_video: false`
- `selenium_fallback_mode: "auto"`

### UI pages and endpoint usage

- **Home/App flow**: `/api/models`, `/api/analyze`, `/api/analyze_compare`
- **Results page**: display + local history/model switching (no direct feedback/threshold/ask-ai calls in current component)
- **Dataset page**: `/api/dataset/preview`, `/api/dataset/export`, `/api/feedback/segment/delete`
- **Model page**: `/api/experiments/registry`, `/api/preprocessing/steps`, `/api/eval/policy` + training tracker store endpoints
- **Admin MLflow page**: uses `/api/mlflow/*` endpoints + `/api/models/import-zip`
- **Protocol page**: `/api/protocols/summary`
- **Synthetic generation page**: `/api/dataset/synthetic/preview|generate|review|delete|export`

---

## Active crawl schema

`comment_crawl.py` currently writes `segments.jsonl` rows like:

```json
{
  "text": "...",
  "segment_index": 0,
  "url_hash": "<md5_of_url>",
  "html_tag_effective": "comment",
  "segment_hash": "<sha256(normalized_text + '|' + html_tag_effective)>"
}
```

- `COMMENT_CRAWL_SCHEMA_VERSION = "comment_only_v3"`
- `text` is kept for compatibility

---

## Important behavior notes (code-verified)

1. `AnalyzeRequest` schema is now:
   - `urls`
   - `options`
   - (no `pending_job_id`, no `fallback_decisions` fields)

2. In current active backend path, `analyze()` calls `crawl_urls(urls, out_dir=...)` without forwarding fallback/video controls.

3. `comment_crawl.crawl_urls()` accepts legacy params (`enable_video`, `enable_asr`, `allow_selenium_fallback`, `fallback_decisions`) but currently discards them in comment-only flow.

So old docs describing interactive ask-mode Selenium decisions or active video crawl in `/api/analyze` are outdated for current runtime.

---

## API surface (grouped)

Source of truth: `backend/app.py`.

### Core inference/model

- `GET /`
- `GET /health`
- `POST /api/analyze`
- `POST /api/analyze_compare`
- `POST /api/analyze/rerun`
- `GET /api/models`
- `POST /api/models/import-zip`

### Feedback + dataset

- `POST /api/feedback`
- `POST /api/feedback/segment`
- `POST /api/feedback/segment/delete`
- `GET /api/dataset/preview`
- `POST /api/dataset/export`

### Synthetic dataset

- `POST /api/dataset/synthetic/generate`
- `GET /api/dataset/synthetic/preview`
- `POST /api/dataset/synthetic/review`
- `POST /api/dataset/synthetic/delete`
- `GET /api/dataset/synthetic/stats`
- `POST /api/dataset/synthetic/export`

### MLflow/admin

- `POST /api/mlflow/ingest`
- `GET /api/mlflow/overview`
- `GET /api/mlflow/batches`
- `GET /api/mlflow/crawl-history`
- `POST /api/mlflow/clear-batch`
- `POST /api/mlflow/clear-all`
- `GET /api/mlflow/review-history`
- `GET /api/mlflow/candidates`
- `POST /api/mlflow/candidates/review`
- `GET /api/mlflow/threshold-status`
- `POST /api/mlflow/manual/export-bundle`
- `GET /api/mlflow/manual/export-bundle/download`
- `POST /api/mlflow/manual/import-artifact`
- `GET /api/mlflow/do/preflight`
- `POST /api/mlflow/do/trigger`
- `GET /api/mlflow/do/status`
- `GET /api/mlflow/compare/latest`
- `POST /api/mlflow/promote`

### Training tracker

- `GET /api/training-tracker`
- `POST /api/training-tracker/phases`
- `PATCH /api/training-tracker/phases/{phase_id}`
- `DELETE /api/training-tracker/phases/{phase_id}`
- `POST /api/training-tracker/phases/reorder`
- `POST /api/training-tracker/groups`
- `PATCH /api/training-tracker/groups/{group_id}`
- `DELETE /api/training-tracker/groups/{group_id}`
- `POST /api/training-tracker/groups/reorder`
- `POST /api/training-tracker/tasks`
- `PATCH /api/training-tracker/tasks/{task_id}`
- `DELETE /api/training-tracker/tasks/{task_id}`
- `POST /api/training-tracker/tasks/reorder`
- `POST /api/training-tracker/tasks/{task_id}/check`
- `POST /api/training-tracker/results`
- `DELETE /api/training-tracker/results/{result_id}`

### Experiment/policy/protocol utilities

- `POST /api/ask-ai`
- `GET /api/gemini/models`
- `GET /api/preprocessing/steps`
- `GET /api/experiments/registry`
- `GET /api/eval/policy`
- `GET /api/eval/errors`
- `GET /api/eval/hard-cases`
- `GET /api/protocols/summary`

---

## Model resolution and default model

Model root resolution order:

1. `VIETTOXIC_MODEL_OPTIONS_DIR`
2. fallback: `models/options`

Default model selection in backend (`get_default_model_id`):

1. Prefer `phobert/finetune_phobert_focalgamma_2` if present and non-deprecated.
2. Else prefer `phobert/v2` if present.
3. Else first non-deprecated PhoBERT model.
4. Else fallback to first available model.

Deprecated model names (contains `deprecated`) are skipped when possible.

---

## Dataset version aliasing used by backend dataset endpoints

`dataset_version` alias map in backend:

- `v1` / `victsd_v1` → canonical `victsd_v1` directory: `data/victsd`
- `latest` / `victsd_gold` → canonical `victsd_gold` directory: `data/processed/victsd_gold`

---

## Local run

### Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

### Frontend

```bash
cd comprehensive_ui
npm install
npm run dev
```

Default local URLs:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

---

## Kaggle Mirror Notebook Workflow

De maintain de hon, repo da co mirror source cho notebook Kaggle:

- Mirror file: `kaggle/notebooks/mlflow_retrain/viettoxic_mlflow_retrain.py`
- Publish script: `scripts/publish_kaggle_kernel.ps1`

Workflow:

1. Sua mirror file trong repo.
2. Publish len Kaggle kernel:

```powershell
.\scripts\publish_kaggle_kernel.ps1 -Owner <kaggle_username> -Slug <kernel_slug>
```

Tuy chon:

```powershell
.\scripts\publish_kaggle_kernel.ps1 -Owner <kaggle_username> -Slug <kernel_slug> -Title "VietToxic Retrain" -Accelerator NvidiaTeslaT4 -Private
```

Luu y:

- `KAGGLE_NOTEBOOK_URL` trong backend map toi kernel URL: `https://www.kaggle.com/code/<owner>/<slug>`.
- Cach nay giup ban cap nhat notebook tu code trong repo, khong phai sua truc tiep tren UI Kaggle.

---

## Docker (CPU first)

Key files in current repo:

- `backend/Dockerfile`
- `comprehensive_ui/Dockerfile`
- `docker-compose.yml`
- `requirements-ml.txt`
- `requirements-base.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/build.yml`

Quick run:

```bash
docker compose up --build
docker compose up
docker compose down
docker compose down -v
```

---

## Backend-only or not wired in current UI

These endpoints exist but are not currently called from the main UI flow/components:

- `/api/feedback`
- `/api/feedback/segment`
- `/api/analyze/rerun`
- `/api/ask-ai`
- `/api/gemini/models`
- `/api/dataset/synthetic/stats`
- `/api/eval/errors`
- `/api/eval/hard-cases`
- `/api/mlflow/manual/export-bundle/download`

(They can still be used directly via API tooling/scripts.)

---

## Rule for future updates

If documentation conflicts with code, trust the running code (`backend/app.py`, `backend/crawl_adapter.py`, `comment_crawl.py`, `infer_crawled_local.py`, and frontend `src/app/*`).

# CLAUDE.md

## MCP & Context Optimization
- **Priority Tooling**: Use the `serena` MCP server first for code search/symbol lookup.
- **Token Efficiency**: Prefer targeted reads (`Grep`, symbol lookup, partial `Read`) over full-file reads.
- **Workflow**:
  1. Locate relevant symbols/files first.
  2. Read only necessary sections.
  3. Edit only source files (not generated artifacts).

## Purpose of this file

This file is the code-aligned guide for working in this repo.
If docs and code conflict, trust running code.

Snapshot verified on **2026-04-10**.

---

## 1) What this project is

Thesis/capstone project for **Vietnamese toxic content detection** with two lanes:

1. **Research/training lane**
   - dataset export/preprocess
   - baseline + PhoBERT/LoRA training
   - experiment tracking and plots

2. **Demo application lane**
   - URL input in React UI
   - crawl comment sections
   - local inference
   - page/segment toxicity results via FastAPI

Core label semantics remain binary:
- `0 = clean`
- `1 = toxic`

---

## 2) Active source-of-truth files

### Highest-priority runtime files

- `backend/app.py`
  - API schema, endpoint behavior, model resolution, dataset endpoints
- `backend/crawl_adapter.py`
  - active crawl binding layer
- `comment_crawl.py`
  - active crawler implementation in runtime flow
- `infer_crawled_local.py`
  - local inference pipeline
- `domain_classifier.py`
  - domain/category threshold logic
- `comprehensive_ui/src/app/App.tsx`
  - frontend app shell + API calls for analyze flow
- `comprehensive_ui/src/app/components/Navigation.tsx`
  - top-level navigable routes
- `comprehensive_ui/src/app/components/DatasetPage.tsx`
  - preview/export wiring + segment-feedback deletion
- `comprehensive_ui/src/app/components/ModelPage.tsx`
  - registry/preprocess/policy display + training tracker UI wiring
- `comprehensive_ui/src/app/components/MLFlowPage.tsx`
  - admin MLflow workflows
- `comprehensive_ui/src/hooks/useMlflowStore.ts`
  - MLflow/admin endpoint wiring
- `comprehensive_ui/src/hooks/useTrainingStore.tsx`
  - training-tracker endpoint wiring

### Important but not in active crawl runtime path

- `setup_and_crawl.py`
  - kept for older article/video crawling workflows; not used by current `backend/crawl_adapter.py`

### Do not edit as source code

- `comprehensive_ui/dist/` (build output)
- generated data/artifact folders under `data/processed/`, `data/raw/crawled_urls/`, zip exports

---

## 3) Crawl lane reality (active vs deprecated)

### Active runtime lane

- `backend/crawl_adapter.py` imports:
  - `from comment_crawl import crawl_urls as crawl_comment_urls`
- `crawl_adapter.crawl_urls(...)` forwards to `comment_crawl.crawl_urls(...)`
- Backend `/api/analyze` and `/api/analyze_compare` use this adapter.

### Current comment crawler schema

`comment_crawl.py`:
- `COMMENT_CRAWL_SCHEMA_VERSION = "comment_only_v3"`
- segment rows include:
  - `text`
  - `segment_index`
  - `url_hash`
  - `html_tag_effective` (default `"comment"`)
  - `segment_hash` (`sha256(normalized_text + '|' + html_tag_effective)`)

### Deprecated lane note

`setup_and_crawl.py` still exists, but the active backend adapter is comment-only.
Do not describe article/video crawling as current default runtime.

---

## 4) FastAPI API contract (current)

Main file: `backend/app.py`

### Analyze schema and defaults

`AnalyzeOptions` currently includes:
- `batch_size=8`
- `max_length=256`
- `page_threshold=0.25`
- `seg_threshold=0.4`
- `model_name: Optional[str]`
- `model_path: Optional[str]`
- `enable_video=False`
- `selenium_fallback_mode: Literal["auto", "ask"] = "auto"`

`AnalyzeRequest` currently includes only:
- `urls`
- `options`

`pending_job_id` and `fallback_decisions` are **not** fields on current request model.

### Important runtime behavior detail

In current `/api/analyze` implementation, backend calls:
- `crawl_urls(urls, out_dir=str(DATA_DIR))`

So fallback/video controls are not forwarded in active analyze flow.

In `comment_crawl.crawl_urls(...)`, legacy parameters exist but are currently discarded (`del ...`).

### Endpoint groups (current code)

Admin auth:
- `POST /api/admin/login`
- `GET /api/admin/session`
- MLflow endpoints and `POST /api/models/import-zip` require `Authorization: Bearer <admin_token>`.
- Credentials/session config come from `VIETTOXIC_ADMIN_USERNAME`, `VIETTOXIC_ADMIN_PASSWORD`, `VIETTOXIC_ADMIN_SESSION_SECRET`, and optional `VIETTOXIC_ADMIN_SESSION_TTL_SECONDS`.

#### Core infer/model
- `GET /`
- `GET /health`
- `POST /api/analyze`
- `POST /api/analyze_compare`
- `POST /api/analyze/rerun`
- `GET /api/models`
- `POST /api/models/import-zip`

#### Feedback + dataset
- `POST /api/feedback`
- `POST /api/feedback/segment`
- `POST /api/feedback/segment/delete`
- `GET /api/dataset/preview`
- `POST /api/dataset/export`

#### Synthetic dataset
- `POST /api/dataset/synthetic/generate`
- `GET /api/dataset/synthetic/preview`
- `POST /api/dataset/synthetic/review`
- `POST /api/dataset/synthetic/delete`
- `GET /api/dataset/synthetic/stats`
- `POST /api/dataset/synthetic/export`

#### MLflow/admin
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

#### Training tracker
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

#### Experiment/policy helpers
- `POST /api/ask-ai`
- `GET /api/gemini/models`
- `GET /api/preprocessing/steps`
- `GET /api/experiments/registry`
- `GET /api/eval/policy`
- `GET /api/eval/errors`
- `GET /api/eval/hard-cases`

---

## 5) Frontend wiring reality

### App shell (`comprehensive_ui/src/app/App.tsx`)

- Analyze flow calls:
  - `/api/analyze`
  - `/api/analyze_compare`
  - `/api/models`
- Current payload hardcodes:
  - `enable_video: false`
  - `selenium_fallback_mode: "auto"`
- Stores:
  - theme: `viettoxic:theme`
  - language: `viettoxic:language`
  - model selection: `viettoxic:models`, `viettoxic:model`
  - scan history: `viettoxic:scan-history`

### Navigation (`Navigation.tsx`)

Top nav includes:
- `home`, `results`, `dataset`, `model`, `contact`, `admin_mlflow`

Not in top nav:
- `dataset_synthetic` (route exists but not listed)

### Results page

Current `ResultsPage.tsx` is display-focused in active UI path:
- shows thresholds and per-segment/page outcomes
- model-switch for compare results is handled from App state
- no current direct API calls from this component

### Dataset page

Uses:
- `GET /api/dataset/preview`
- `POST /api/dataset/export`
- `POST /api/feedback/segment/delete`

Dataset page uses the fixed `latest` dataset version (`victsd_gold`). The legacy protocol page and v1 switch were removed.

### Model page

Not purely static. It fetches live backend data:
- `/api/experiments/registry`
- `/api/preprocessing/steps`
- `/api/eval/policy`

Also mounts Training tracker UI backed by `/api/training-tracker*` endpoints.

### Admin MLflow page

`MLFlowPage` + `useMlflowStore` actively call many `/api/mlflow/*` endpoints and `/api/models/import-zip`.

### Synthetic page

`SyntheticGenerationPage` actively uses:
- `/api/dataset/synthetic/preview`
- `/api/dataset/synthetic/generate`
- `/api/dataset/synthetic/review`
- `/api/dataset/synthetic/delete`
- `/api/dataset/synthetic/export`

---

## 6) Backend-only / not wired in current UI flow

Exists in backend but not currently called by main UI components:

- `/api/feedback`
- `/api/feedback/segment`
- `/api/analyze/rerun`
- `/api/ask-ai`
- `/api/gemini/models`
- `/api/dataset/synthetic/stats`
- `/api/eval/errors`
- `/api/eval/hard-cases`
- `/api/mlflow/manual/export-bundle/download`

Use label `backend-only` or `not wired in current UI` in docs.

---

## 7) Model resolution and defaults (current code)

Model root resolution order:
1. `VIETTOXIC_MODEL_OPTIONS_DIR`
2. fallback: `models/options`

`get_default_model_id()` in backend:
1. prefer `phobert/finetune_phobert_focalgamma_2` (non-deprecated)
2. else `phobert/v2` if present
3. else first non-deprecated PhoBERT
4. else first non-deprecated model across all types
5. else first available model

Names containing `deprecated` are skipped when possible.

Required artifacts:
- `phobert`: `config.json` + (`model.safetensors` or `pytorch_model.bin`)
- `tfidf_lr`: `vectorizer.pkl` + `model_lr.pkl`

---

## 8) Dataset version aliasing and UI wiring

Backend alias map (`DATASET_VERSION_ALIASES`):
- `latest` / `victsd_gold` -> canonical `victsd_gold`

Backend directory map (`DATASET_VERSION_DIRS`):
- `victsd_gold` -> `data/processed/victsd_gold`

Legacy `v1` / `victsd_v1` requests return `400`. `data/victsd` and `data/processed/victsd_v1` are not active repo data; keep `data/raw/**`.

Frontend `DatasetPage` sends `dataset_version=latest`.

---

## 9) Training/data pipeline references

Main scripts still used for thesis workflow:
- `scripts/01_export_raw.py`
- `scripts/02_preprocess.py`
- `scripts/02b_prepare_gold_dataset.py`
- `scripts/03_eda.py`
- `scripts/04_baseline_tfidf_lr.py`
- `scripts/05_train_phobert.py`
- `scripts/06_train_phobert_lora.py`

---

## 10) Docker / CI-CD (current files)

Present files:
- `backend/Dockerfile`
- `comprehensive_ui/Dockerfile`
- `docker-compose.yml`
- `requirements-ml.txt`
- `requirements-base.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/build.yml`

---

## 11) Practical gotchas

1. Do not document ask-mode two-step fallback as active UI behavior; current App sends `selenium_fallback_mode="auto"`.
2. Do not document active video crawl from UI analyze flow; current App sends `enable_video=false` and backend analyze path does not forward it to crawl call.
3. `setup_and_crawl.py` exists but is not current backend crawl adapter path.
4. `ModelPage` is partially dynamic now; avoid calling it purely static/mock.
5. `dataset_synthetic` exists but is not in top navigation.
6. If docs disagree with code, prefer:
   - `backend/app.py`
   - `backend/crawl_adapter.py`
   - `comment_crawl.py`
   - `infer_crawled_local.py`
   - `comprehensive_ui/src/app/App.tsx`

---

## 12) Decision rule

For any implementation/doc conflict:

**running code > historical notes > README text > UI wording**

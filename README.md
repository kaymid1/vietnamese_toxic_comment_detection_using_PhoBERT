# VietComment Analyzer

Research + demo system for Vietnamese toxic-content detection from URLs.

## What is active right now

This repo currently runs a **web UI â†’ FastAPI â†’ comment-only crawl â†’ local inference** flow.

Active runtime path:

1. User submits URLs in the React UI.
2. Backend crawls **comment sections only** (via `comment_crawl.py` through `backend/crawl_adapter.py`).
3. Backend runs local model inference (`infer_crawled_local.py`).
4. Backend optionally stores new inferred comment data in the MLflow review DB for admin review/retrain bundles.
5. UI shows page-level and segment-level toxicity results.

The old article/video crawl lane (`setup_and_crawl.py`) is still in the repo, but it is **not** the active crawl adapter path.

---

## Architecture

- **Frontend**: Vite + React 18 + TypeScript (`comprehensive_ui/`)
- **Backend**: FastAPI (`backend/app.py`)
- **Inference**: local PhoBERT / TF-IDF-LR (`infer_crawled_local.py`)
- **Crawler in active path**: `comment_crawl.py`
- **Feedback / metadata storage**: SQLite (`data/processed/feedback/feedback.db`)
- **MLflow review candidates**: same SQLite DB, table `mlflow_comment_item`

---

## Current frontend wiring (actual)

### App shell and navigation

Main app: `comprehensive_ui/src/app/App.tsx`

- Primary pages in top nav: `home`, `results`, `dataset`, `model`, `contact`, `admin_mlflow`
- `dataset_synthetic` is admin-only: its Dataset panel and navigation item appear only after a verified admin login, and every `/api/dataset/synthetic/*` endpoint requires the admin bearer token.
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

4. `/api/analyze` now defaults to `collect_for_mlflow: true`.
   - New inferred segments are gated with `mlflow_gate_accept_threshold` / `mlflow_gate_discard_threshold`.
   - A user scan creates an `mlf_auto_*` MLflow batch only when at least one new row is inserted.
   - If the URL was already collected in `mlflow_comment_item`, the whole URL is skipped for MLflow collection.
   - If the segment already exists by `context_segment_hash`/`segment_hash` + `html_tag` (or computed `dedupe_key`), it is skipped.
   - Public analysis results are still returned normally even when MLflow collection skips all rows.

### MLflow review and training semantics

The following definitions describe the active implementation in `backend/app.py`, not just the UI labels:

- **Runtime DB**: MLflow batches and review rows are persisted in `data/processed/feedback/feedback.db`, table `mlflow_comment_item`.
- **Automatic gating** uses the model's raw `toxic_prob`. The defaults are `mlflow_gate_discard_threshold=0.20` and `mlflow_gate_accept_threshold=0.80`:
  - `score <= 0.20`: stored as `gate_bucket=accepted`, `pseudo_label=0`, `verification_status=auto_accepted`, `training_review_status=auto`, and selected for training.
  - `score >= 0.80`: stored as `gate_bucket=accepted`, `pseudo_label=1`, `verification_status=auto_accepted`, `training_review_status=auto`, and selected for training.
  - `0.20 < score < 0.80`: stored as `gate_bucket=candidate`, `verification_status=unverified`, `training_review_status=pending`, and not selected for training. Its provisional `pseudo_label` is split at `0.50`.
  - Batch thresholds may be recorded in the batch `options_json`, but `GET /api/mlflow/training-preview` currently does not return them. The UI therefore documents and displays severity using the verified defaults `0.20/0.80`; it does not infer per-batch values.
- **Manual Verify (DB persisted pool)** only lists `candidate` rows whose verification status is `unverified`; rows that passed the gate into Training Preview do not appear here again. The UI checkbox is temporary row selection; `Toxic`, `Clean`, and `Remove` call the review API and update DB state. Multiple selected candidates can use the same batched Gemini suggestion/apply flow as Training Preview. Applying a suggestion immediately moves the row into Training Preview with `training_review_status=manual_gemini` and `label_source=gemini_assist`, displayed as both **Review: Thủ công + Gemini** and **Gemini assisted**. `Remove` moves an unlocked row to the discarded/manual-rejected state rather than deleting it physically.
- **Training Preview** only lists rows in the `accepted` bucket that remain selected for training. Admins can directly correct each row to Toxic/Clean; this persists the corrected `pseudo_label`, `manual_accepted` verification, `manual_approved` review status, and `manual_override` label source. Removing a row from training keeps its DB record and lineage but hides it from Training Preview immediately. A distribution dialog summarizes Toxic/Clean and constructive/non-constructive/masked counts over the complete selected preview set. A row is eligible for the accepted export set only when it is accepted, has `selected_for_training=1` (or the legacy nullable equivalent accepted by the export query), and has `pseudo_label` 0 or 1. Candidate rows remain available through Manual Verify and the bundle's candidate file without being eligible for the accepted training set. Admin-confirmed Synthetic rows enter this same accepted pool with `source_type=synthetic`, `label_source=synthetic_review`, and their original `source_row_id`; repeated confirmation does not duplicate them.
- **Training Preview list UX** requests up to 300 rows per page, renders every returned row, and provides a draggable/keyboard-accessible resize separator below the list. Each row is explicitly labeled **Nguồn: Thu thập từ website** or **Nguồn: Tạo sinh bằng Gemini** from its persisted `source_type` provenance. Source origin and review status are deliberately shown as separate badges.
- **`selected_for_training`** is a persistent DB selection flag, not export or training lineage. Value `1` means “selected for training consideration”; it does not prove that the row was included by balancing, exported, submitted to Kaggle, or used by a successful training run. The row checkboxes in Training Preview are a separate in-memory UI selection used for current-page actions such as Gemini review.
- **`manual_approved`** is the stored `training_review_status` written when the Training Preview selection endpoint sets `selected_for_training=1`. Candidate Toxic/Clean review can also produce the same status. It therefore does not always prove that a human verified the final toxicity label, and it does not mean exported or trained.
- **Gemini-assisted review** accepts a configurable maximum per admin operation (`GEMINI_REVIEW_MAX_ITEMS`, default 9, hard cap 25) in Manual Verify, Training Preview, and Synthetic review. The backend processes at most 3 comments per Gemini request, spaces live Gemini calls by `GEMINI_MIN_REQUEST_INTERVAL_SECONDS` (default 13 seconds), retries transient minute-window/503 failures a bounded number of times, moves directly to a configured fallback for daily-quota exhaustion, and retries omitted/invalid multi-comment output one row at a time. Every suggestion includes the actual provider/model that returned its chunk. Suggestions remain pending until the admin applies one row or uses **Apply Gemini** for all visible suggestions; Apply persists `review_provider` and `review_model_name` on the training/synthetic row. Applying also preserves the review origin in `training_review_status`: automatic Training Preview rows become `auto_gemini` (shown as **Tự động + Gemini**), while previously manual rows and candidate rows applied from Manual Verify become `manual_gemini` (shown as **Thủ công + Gemini**).
- **Constructiveness** is stored independently as `constructiveness_label` 0/1. `NULL` means the label is absent/masked; active code can produce it when no constructiveness output is available, when the helper masks a `0.30 < score < 0.70` result without an explicit label, or when an admin clears the label. Because an explicit 0/1 model label takes priority, `NULL` does not always mean low confidence. The repository does not currently define a richer business taxonomy beyond the binary constructiveness task.
- **Balanced export** is performed in the backend by `balance_training_rows`. With the `balanced_50_50` strategy, the accepted Toxic/Clean sets are reduced to the smaller class when both classes exist, so an eligible selected row can still be omitted. Export does not persist a per-row “used for training” marker.
- **Training plan** (`GET /api/mlflow/training-plan`) uses the same selection, balancing, normalization, and deduplication helpers as bundle creation. It reports gold train size, eligible MLflow rows, rows after balancing, duplicates/empty rows removed, MLflow rows added, final train size, and a per-row `will_finetune` reason shown in Training Preview.
- **Dataset Bundle readiness** is derived from `trainingPlan.summary.mlflow_added`, not the historical count of every row in the accepted bucket. The compact bundle card therefore mirrors the next export snapshot: eligible MLflow rows, rows after balancing, duplicates removed, rows actually added, and final train size. The configured target is considered ready only when the actual MLflow rows added to that snapshot reach it.
- **Lock** is a limited guard: it prevents Manual Verify `Remove` and prevents deselecting a locked training row. It is not a universal write/export lock.
- **Kaggle bundle triggering** now creates a fresh `clean_victsd_gold` bundle snapshot before contacting `KAGGLE_WEBHOOK_URL`. The run stores its bundle path, SHA-256 checksum, token-protected download URL, balance policy, merge counts, and exact included `mlflow_comment_item` IDs in `mlflow_do_run`; the webhook receives that exact URL and checksum. The generated Kaggle notebook requires this run bundle and verifies SHA-256 before extraction. LR Smoke preserves every included reviewed row, whether it came from website collection (`MLFlowAccepted`) or Gemini generation (`SyntheticReviewed`), and downsamples only gold rows. Configure `KAGGLE_BUNDLE_PUBLIC_BASE_URL` when the backend's request origin is not publicly reachable from Kaggle.
- **MLflow Automation** is disabled by default. Admin System Settings provides one global switch and an independent mode per family: `disabled`, `train_only`, or `full_auto`. A new accepted training row may start an automation cycle only after the configured minimum of newly eligible rows and cooldown are satisfied; one active run is reserved per family. Every automation decision is recorded in `mlflow_automation_state` and `mlflow_automation_event`. `train_only` leaves a completed candidate awaiting an admin promotion. `full_auto` promotes only automation-created runs that pass the existing checksum, serving-contract, same-semantic-test-set, and metric gates. Automation uses `MLFLOW_AUTOMATION_DRY_RUN=true` by default; a real automatic run additionally requires `KAGGLE_BUNDLE_PUBLIC_BASE_URL`. The in-process watcher polls Kaggle status while the backend stays alive; `GET /api/mlflow/kaggle/status` also resumes terminal handling after a restart.
- **Gemini Evaluate** is available only after a Kaggle run has completed with a real artifact. It receives structured evidence for the candidate, the current production model, the latest earlier run of the same model kind, test comparability, and production-gate results. Its Vietnamese assessment and the actual model that returned it (including fallback) are cached in `mlflow_gemini_evaluation` and shown beside the Kaggle metrics. The assessment is advisory: it never promotes a model or bypasses a gate. Admins can inspect and edit the active Review/Evaluate instructions in System Settings → **AI Instructions**; the fixed JSON schema and safety constraints remain enforced by backend code.

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

- All Synthetic endpoints require admin authentication. The generation page is an unreviewed-only queue: `/api/dataset/synthetic/preview` rejects `reviewed=true`, and the UI has no DB Reviewed mode, accepted/rejected history filter, reviewed export button, or standalone transfer action. Save review finalizes every row on the current page (checked = accepted, unchecked = rejected); after Save review or Apply Gemini, processed rows leave the page immediately. Accepted rows can enter Training Preview only through the confirmation dialog opened by that review action. Rejected and already-transferred rows are excluded, and the transfer endpoint remains idempotent.
- Newly generated rows expose their current Toxic/Clean and constructiveness labels before persistence. Admins may select rows for Gemini Review, inspect each pending suggestion/confidence/reason and the actual Gemini model, and apply individually or in bulk. A normal Save review persists `review_method=manual`; applying Gemini persists `review_method=gemini_assisted`, `review_provider`, and `review_model_name`. On transfer these become `manual_approved`/`synthetic_review` and `manual_gemini`/`gemini_assist`, respectively, while Gemini provenance follows the row into Training Preview.
- `POST /api/dataset/synthetic/generate`
- `GET /api/dataset/synthetic/preview`
- `POST /api/dataset/synthetic/gemini-review`
- `POST /api/dataset/synthetic/review`
- `GET /api/dataset/synthetic/training-preview-summary`
- `POST /api/dataset/synthetic/transfer-to-training-preview`
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
- `GET /api/mlflow/training-preview`
- `POST /api/mlflow/training-preview/gemini-review`
- `POST /api/mlflow/training-preview/review`
- `POST /api/mlflow/manual/export-bundle`
- `GET /api/mlflow/manual/export-bundle/download`
- `POST /api/mlflow/manual/import-artifact`
- `GET /api/mlflow/kaggle/preflight`
- `POST /api/mlflow/kaggle/trigger`
- `GET /api/mlflow/kaggle/status`
- `GET /api/mlflow/kaggle/artifact/download`
- `POST /api/mlflow/kaggle/evaluate`
- `GET /api/mlflow/automation/status`
- `POST /api/mlflow/automation/cycle`
- `GET /api/mlflow/compare/latest`
- `POST /api/mlflow/promote`
- `POST /api/mlflow/rollback`

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

### Experiment/policy utilities

- `POST /api/ask-ai`
- `GET /api/gemini/models`
- `GET /api/preprocessing/steps`
- `GET /api/experiments/registry`
- `GET /api/eval/policy`
- `GET /api/eval/errors`
- `GET /api/eval/hard-cases`

---

## Runtime models, selection, and default

Model root resolution order:

1. `VIETTOXIC_MODEL_OPTIONS_DIR`
2. fallback: `models/options`

The user-selectable runtime models are:

- **TF-IDF baseline** — `tfidf_lr/baseline_tfidf`, using TF-IDF features with Logistic Regression.
- **PhoBERT v1 transformer baseline** — `phobert/baseline`, fully fine-tuned from `vinai/phobert-base`.
- **PhoBERT v2 fine-tuned model** — `phobert/phobert_v2_finetuned`, displayed as **PhoBERT v2 Fine-tuned**. Its unchanged physical folder is `phobert_lora_4.7` for backward compatibility. The folder's `run_config.json` and `training_manifest.json` both record `vinai/phobert-base-v2` as the base model. The checkpoint contains the full model weights; it is not a LoRA or PEFT adapter.

The runtime maintains independent production slots for `tfidf_lr` and `phobert`.
Kaggle candidates are compared only with the production model from the same family.
Every bundle records `included_mlflow_ids_sha256` and a content-based
`feedback_snapshot_sha256`, allowing LR and PhoBERT runs to prove that the same new
reviewed rows were injected even when their effective gold-data sampling differs.
Promotion requires a completed real run, verified artifact SHA-256, a compatible
serving artifact, the same semantic test-set fingerprint, and passing metric gates.
The artifact is installed under an immutable family/version directory before the
production pointer is updated. Rollback swaps the family pointer back to the prior
version without deleting either artifact. `GET /api/models` exposes both pointers in
`production_slots`; `/api/analyze` includes `serving_evidence` for the model actually
used. The fast Logistic Regression Kaggle profile is recorded as `retrain`, not
`finetune`.

When the UI first loads with no saved user choice, it uses the default returned by
`GET /api/models`. An explicit user selection is sent as `options.model_name` and
always takes precedence over backend fallback behavior. Runtime responses expose the
selected internal identifier in `model_name`, and the backend also logs it.

Default model selection in the backend and local inference CLI (`get_default_model_id`):

1. The compatible model recorded in the `phobert` production slot.
2. Verified `phobert/phobert_v2_finetuned` fine-tuned from `vinai/phobert-base-v2`, when its backing folder is present and compatible.
3. An available, compatible, non-deprecated PhoBERT checkpoint other than the legacy v1 baseline.
4. The compatible legacy `phobert/baseline` v1 checkpoint.
5. The first compatible available model.

Missing or incomplete model folders are skipped during fallback selection. The
selected PhoBERT checkpoint's local tokenizer is loaded with the model; if local
tokenizer files are unavailable, inference resolves the recorded base model from
`training_manifest.json` or `run_config.json`. TF-IDF inference does not load a
transformer tokenizer.

---

## Dataset version aliasing used by backend dataset endpoints

`dataset_version` alias map in backend:

- `latest` / `victsd_gold` â†’ canonical `victsd_gold` directory: `data/processed/victsd_gold`

Legacy `v1` / `victsd_v1` requests are rejected with `400`. The old protocol dataset directory `data/victsd` was removed. Raw source data under `data/raw/**` is intentionally kept.

---

## ViCTSD preprocessing (gold dataset)

Current dataset build scripts:

- `scripts/02_preprocess.py` (main preprocess + leakage report)
- `scripts/02b_prepare_gold_dataset.py` (alternate builder with summary report)
- `scripts/02c_validate_victsd_gold.py` (post-build validation)

Default raw/input split source (must keep source split files unchanged):

- `data/raw/victsd/train.jsonl`
- `data/raw/victsd/validation.jsonl`
- `data/raw/victsd/test.jsonl`

Default processed output:

- `data/processed/victsd_gold/train.jsonl`
- `data/processed/victsd_gold/validation.jsonl`
- `data/processed/victsd_gold/test.jsonl`

Processed row schema (current):

```json
{
  "text": "...",
  "toxicity": 0,
  "label": 0,
  "constructiveness": 1,
  "meta": {
    "source": "victsd",
    "split": "train",
    "topic": "...",
    "title": "...",
    "original_comment": "..."
  }
}
```

Notes:

- `toxicity` remains the active binary label key for existing training/inference scripts.
- `label` is mirrored from `toxicity` for compatibility with audit/tools that expect `label`.
- `constructiveness` is preserved from raw ViCTSD for constructiveness analysis extension.
- Preprocess keeps Vietnamese text features: trim, NFC normalization, whitespace normalization, no forced lowercase, punctuation preserved.
- Current preprocess defaults to `--cross-split-dedup strong` (priority order: `train` -> `validation` -> `test`) to remove cross-split overlap.
- PhoBERT full fine-tuning macro-F1 scripts train multi-label binary tasks with two logits: `toxicity` and `constructiveness`. Toxicity remains the primary detection/deployment task and constructiveness is logged as an auxiliary task metric. Some script filenames retain `lora` as a legacy internal name, but the active implementation performs full fine-tuning and contains no LoRA/PEFT path.

Run preprocessing:

```bash
python scripts/02_preprocess.py --input-dir data/raw/victsd --output-dir data/processed/victsd_gold --cross-split-dedup strong
```

Validate processed files:

```bash
python scripts/02c_validate_victsd_gold.py --data-dir data/processed/victsd_gold --sample-n 3
```

---

## Local run

### Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

### Admin access for MLflow

MLflow admin UI and APIs require an admin session. Configure these in `.env.local` or `backend/.env.local` before using the MLflow page:

```env
VIETTOXIC_ADMIN_USERNAME=admin
VIETTOXIC_ADMIN_PASSWORD=change-me
VIETTOXIC_ADMIN_SESSION_SECRET=replace-with-a-long-random-secret
VIETTOXIC_ADMIN_SESSION_TTL_SECONDS=28800
```

Protected routes:

- `/api/mlflow/*`
- `POST /api/models/import-zip`

### Frontend

```bash
cd comprehensive_ui
npm install
npm run dev
```

Default local URLs:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

### One-command startup (backend + frontend + webhook receiver + ngrok)

Windows:

```powershell
cd D:\Code\Thesis\Thesis
.\start.ps1
```

macOS/Linux:

```bash
cd /path/to/Thesis/Thesis
chmod +x ./start.sh
./start.sh
```

Default ports and tunnel:

- Backend API: `http://127.0.0.1:8000`
- Frontend UI: `http://127.0.0.1:5173`
- Webhook receiver: `http://127.0.0.1:9001`
- Webhook public URL: `https://living-rare-ram.ngrok-free.app`

Optional:

- Start extra frontend tunnel: `START_FRONTEND_NGROK=1 ./start.sh` or `.\start.ps1 -StartFrontendNgrok`
- Override webhook domain: set `WEBHOOK_NGROK_DOMAIN` in `start.sh` env or use `-WebhookNgrokDomain` in `start.ps1`
- Use template config for `ngrok start --all`: `scripts/ngrok.example.yml`

### Mock Kaggle webhook receiver (for end-to-end local test)

Added local service file:

- `backend/kaggle_webhook_receiver.py`

Exposed endpoints:

- `POST /kaggle/trigger`
- `GET /kaggle/status?job_id=...`

This receiver is compatible with backend calls from:

- `KAGGLE_WEBHOOK_URL`
- `KAGGLE_STATUS_WEBHOOK_URL`

Example in `backend/.env.local`:

```env
KAGGLE_WEBHOOK_URL=https://living-rare-ram.ngrok-free.app/kaggle/trigger
KAGGLE_STATUS_WEBHOOK_URL=https://living-rare-ram.ngrok-free.app/kaggle/status
```

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
.\scripts\publish_kaggle_kernel.ps1 -Owner <kaggle_username> -Slug <kernel_slug> -Title "VietComment Analyzer Retrain" -Accelerator NvidiaTeslaT4 -Private
```

Luu y:

- `KAGGLE_NOTEBOOK_URL` trong backend map toi kernel URL: `https://www.kaggle.com/code/<owner>/<slug>`.
- Cach nay giup ban cap nhat notebook tu code trong repo, khong phai sua truc tiep tren UI Kaggle.

### LR Smoke demo profile

`LR Smoke` la profile kiem thu nhanh pipeline bang TF-IDF + Logistic Regression, khong phai PhoBERT fine-tuning.

- Moi lan trigger tao mot bundle snapshot moi va Kaggle xac minh SHA-256 truoc khi train.
- Toan bo row `MLFlowAccepted` trong `build_report.json` duoc giu lai; chi phan gold cu bi downsample de dat gioi han smoke. Run se fail neu MLflow ID trong bundle khong khop tap train thuc te.
- Artifact ZIP chua `metrics.json` va `training_evidence.json`: kich thuoc dataset, MLflow IDs da dung, hash cua danh sach ID, seed, duration, precision/recall va confusion matrix.
- MLflow page hien thi provenance, bieu do validation/test, confusion matrix, bundle checksum va artifact checksum de phan biet real run voi placeholder.
- Profile nay giup giam thoi gian train model, nhung khong dam bao tong thoi gian 1-2 phut vi Kaggle queue/cold start nam ngoai ung dung.

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

(They can still be used directly via API tooling/scripts.)

---

## Rule for future updates

If documentation conflicts with code, trust the running code (`backend/app.py`, `backend/crawl_adapter.py`, `comment_crawl.py`, `infer_crawled_local.py`, and frontend `src/app/*`).


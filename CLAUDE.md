# CLAUDE.md

## MCP & Context Optimization
- **Priority Tooling**: Always prioritize using the `serena` MCP server for context gathering, project indexing, and code search.
- **Token Efficiency**: Before reading multiple files manually with `ls` or `cat`, use `serena`'s search/indexing tools to identify and fetch only the relevant code snippets.
- **Workflow**:
  1. Use `serena` to get a high-level overview of the project structure.
  2. Use `serena` to locate specific logic or variable definitions instead of broad file reads.
  3. Only request full file content if summaries are insufficient.

## Purpose of this file

This file is the "base knowledge" guide so Claude can work in the right direction inside this repository.
It prioritizes **what the code currently does** over general descriptions in the README.

This snapshot was compiled from the repo on **2026-03-22**.
If the code changes later, always re-check the source before making edits.

---

## 1. What this project actually is

This repository is for a thesis / capstone project about **Vietnamese toxic content detection**.

It has 2 main parts:

1. **Research / experimentation**
   - export the ViCTSD dataset
   - preprocess data
   - run EDA
   - train a TF-IDF + Logistic Regression baseline
   - fine-tune PhoBERT
   - generate plots / logs for the thesis

2. **Demo application**
   - crawl content from URLs
   - split text into segments
   - run inference with a local PhoBERT checkpoint
   - apply domain-aware thresholds
   - return results through FastAPI
   - show results in a Vite/React frontend

The core task is still:

- input: Vietnamese text
- output: binary `toxicity`
- `0 = clean`
- `1 = toxic`

The repo does not only classify dataset comments. It also includes a URL crawling + page/segment inference pipeline.

---

## 2. Source of truth and active files

When editing code, prioritize these files:

- `backend/app.py`
  - FastAPI backend used by the UI
- `infer_crawled_local.py`
  - main local inference script for crawled data
- `comment_crawl.py` **(latest)**
  - comment-only crawl pipeline (news site comment sections + Facebook comments)
  - outputs segments.jsonl in the same schema as setup_and_crawl.py
  - replaces setup_and_crawl.py for comment-focused use cases
- `setup_and_crawl.py` **(deprecated for comment use case)**
  - hybrid crawler for full article body, video transcript handling, ASR
  - use `comment_crawl.py` instead for comment section crawling
- `domain_classifier.py`
  - domain classification and threshold selection
- `scripts/01_export_raw.py`
  - export raw ViCTSD
- `scripts/02_preprocess.py`
  - preprocess dataset into `data/processed/victsd_v1`
- `scripts/02b_prepare_gold_dataset.py`
  - build leakage-aware gold base from raw ViCTSD into `data/processed/victsd_gold`
- `scripts/02a_build_protocol_datasets.py`
  - build protocol A/B/C datasets from ViCTSD + ViHSD raw (includes offensive-only ViHSD preprocessing + merge)
- `scripts/03_eda.py`
  - EDA
- `scripts/04_baseline_tfidf_lr.py`
  - baseline model
- `scripts/05_train_phobert.py`
  - PhoBERT fine-tuning
- `comprehensive_ui/src/app/App.tsx`
  - frontend root app
- `comprehensive_ui/src/app/components/HomePage.tsx`
  - URL input, model selection, compare-mode entry point
- `comprehensive_ui/src/app/components/ResultsPage.tsx`
  - inference result display, feedback, threshold tuning UI
- `comprehensive_ui/src/app/components/DatasetPage.tsx`
  - dataset preview / export and feedback-data inspection

These files / folders are **not the main source**:

- `comprehensive_ui/dist/`
  - build artifact, do not edit directly
- `comprehensive_ui/src/app/components/ModelPage.tsx`
  - mostly static / mock content
- `comprehensive_ui/src/app/components/ContactPage.tsx`
  - placeholder / demo content

---

## 3. Technical stack

### Backend / ML

- Python
- FastAPI + Uvicorn
- Hugging Face `transformers`
- `torch`
- `datasets`
- `scikit-learn`
- `mlflow`
- `sqlite3` for feedback / threshold storage
- `trafilatura`
- `selenium` + `undetected-chromedriver` (Chrome fallback) / `selenium.webdriver.Edge` (Edge, via selenium-manager)
- `vncorenlp`
- `yt-dlp`
- `youtube-transcript-api`
- `faster-whisper` + `ffmpeg` for native video ASR
- Gemini / Google Generative Language REST API via `urllib`

### Frontend

- Vite
- React 18
- TypeScript
- Tailwind CSS v4
- many Radix / bundled UI components

---

## 4. Important directory layout

### Data

- `data/raw/victsd/`
  - raw ViCTSD export from Hugging Face
- `data/raw/vihsd/`
  - raw UIT-ViHSD export (`free_text`, `label_id`, `label`) + `metadata.json` label map
- `data/processed/victsd_v1/`
  - preprocessed ViCTSD dataset used as base input
- `data/victsd/`
  - protocol artifacts and reports for A/B/C comparisons
  - includes:
    - `protocol_a/victsd_v1_protocol_a_{train,validation,test}_augmented.jsonl`
    - `protocol_b/victsd_v1_protocol_b_{train,validation,test}_augmented.jsonl`
    - `protocol_c/victsd_v1_protocol_c_{train,validation,test}_augmented.jsonl`
    - `victsd_v1_protocol_build_report.json`
- `data/raw/crawled_urls/<url_hash>/`
  - crawl artifact for each URL
- `data/raw/mock_crawled_urls/`
  - mock crawl inputs for inference sanity checks
- `data/processed/job_<uuid>/`
  - output for each `/api/analyze` API request
- `data/processed/feedback/feedback.db`
  - SQLite store for page feedback, segment feedback, and threshold overrides
- `data/processed/combined_dataset.jsonl`
  - generated export from `/api/dataset/export` when requested

### Models / results

- `models/options/phobert/`
  - local PhoBERT checkpoints used by the backend / inference script
- `models/options/tfidf_lr/baseline/`
  - local TF-IDF + LR serving artifacts for compare mode / alternate inference
- `results/baseline/`
  - baseline metrics + serialized LR/vectorizer when the training script is run
- `results/phobert/`
  - expected output from the training script, but there is no committed `results/` folder in the current repo snapshot
- `experiments/phobert_optimization_log.md`
  - important tuning log
- `experiments/crawling_log.md`
  - crawl log

### App

- `backend/`
  - API
- `comprehensive_ui/`
  - frontend
- `scripts/`
  - training / EDA / run scripts

---

## 5. Dataset and data schema

The main dataset is **ViCTSD**.

### Raw export

`scripts/01_export_raw.py` downloads dataset `tarudesu/ViCTSD` and writes:

- `data/raw/victsd/train.jsonl`
- `data/raw/victsd/validation.jsonl`
- `data/raw/victsd/test.jsonl`

The raw schema in this repo includes fields such as:

- `Comment`
- `Constructiveness`
- `Toxicity`
- `Title`
- `Topic`

### Processed dataset

`scripts/02_preprocess.py` reads the raw files and writes:

- `data/processed/victsd_v1/train.jsonl`
- `data/processed/victsd_v1/validation.jsonl`
- `data/processed/victsd_v1/test.jsonl`

For gold-base retraining input (separate from protocol artifacts),
`scripts/02b_prepare_gold_dataset.py` reads the same raw files and writes:

- `data/processed/victsd_gold/train.jsonl`
- `data/processed/victsd_gold/validation.jsonl`
- `data/processed/victsd_gold/test.jsonl`
- `data/processed/victsd_gold/build_report.json`

Schema:

```json
{
  "text": "...",
  "toxicity": 0,
  "meta": {
    "source": "ViCTSD",
    "original_length": 42,
    "processed_length": 40
  }
}
```

### Preprocessing rules currently implemented

- trim leading / trailing whitespace
- normalize Unicode to NFC
- normalize whitespace
- preserve case
- do not aggressively remove punctuation
- do not remove emojis
- do not remove toxic keywords

For `scripts/02b_prepare_gold_dataset.py`, additional constraints are applied:

- map `text <- Comment`, `toxicity <- Toxicity` (cast to int, expect binary 0/1)
- drop rows with empty text after cleaning
- deduplicate exact normalized text within each split (keep first occurrence)
- print split-level summary and overlap counts (`train↔validation`, `train↔test`, `validation↔test`)
- persist all summary metrics to `data/processed/victsd_gold/build_report.json`

Why:

- PhoBERT is being used in a way that keeps text close to the original form
- sarcasm, punctuation, emojis, and teencode can all be important toxicity signals

---

## 6. Research / training pipeline

The current protocol-aware working order is:

1. `scripts/01_export_raw.py` (ViCTSD raw)
2. `scripts/02_preprocess.py` (ViCTSD preprocessing)
3. `scripts/02a_build_protocol_datasets.py` (build A/B/C from ViCTSD + ViHSD raw)
4. `scripts/03_eda.py` (optional per-protocol EDA)
5. `scripts/04_baseline_tfidf_lr.py` (protocol-specific run)
6. `scripts/05_train_phobert.py` or `scripts/06_train_phobert_lora.py` (protocol-specific run)
7. `generate_thesis_plots.py`

### Protocol semantics used in thesis workflow

- **Protocol A**: ViCTSD-only (legacy anchor baseline)
- **Protocol B**: ViCTSD + ViHSD OFFENSIVE in train only; validation/test remain ViCTSD
- **Protocol C**: merged benchmark with global exact dedup + stratified split

This supports a final decision phase where A/B/C are compared by gating + weighted metrics.

### EDA

`scripts/03_eda.py` generates:

- `reports/eda/eda_summary.md`
- `reports/eda/label_distribution.png`
- `reports/eda/length_distribution.png`
- `reports/eda/top_tokens_clean.png`
- `reports/eda/top_tokens_toxic.png`
- `reports/eda/top_bigrams_clean.png`
- `reports/eda/top_bigrams_toxic.png`

### Baseline

`scripts/04_baseline_tfidf_lr.py` now supports protocol-specific dataset loading via env:

- `DATA_DIR`
- `DATASET_PREFIX`
- optional `OUTPUT_BASE`, `RESULTS_BASE`, `SEED`

Dataset file pattern:

- `${DATA_DIR}/${DATASET_PREFIX}_train_augmented.jsonl`
- `${DATA_DIR}/${DATASET_PREFIX}_validation_augmented.jsonl`
- `${DATA_DIR}/${DATASET_PREFIX}_test_augmented.jsonl`

This avoids filename collisions when training multiple protocols on Google Drive / Colab.

`04_baseline_tfidf_lr.py` uses:

- `TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"(?u)\\b\\w+\\b")`
- `LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1)`

Artifacts:

- `results/baseline/metrics.json`
- `results/baseline/vectorizer.pkl`
- `results/baseline/model_lr.pkl`

Important note:

- the current repo snapshot does **not** contain a committed `results/` directory
- if you need reference numbers without rerunning training, use `experiments/phobert_optimization_log.md`
- baseline reference in that log:
  - test macro F1: `0.7043`
  - test F1_toxic: `0.4844`

### PhoBERT training scripts

`scripts/05_train_phobert.py` and `scripts/06_train_phobert_lora.py` support protocol-specific dataset loading via env:

- `DATA_DIR`
- `DATASET_PREFIX`
- optional `OUTPUT_BASE`, `RESULTS_BASE`, `SEED`

Dataset file pattern:

- `${DATA_DIR}/${DATASET_PREFIX}_train_augmented.jsonl`
- `${DATA_DIR}/${DATASET_PREFIX}_validation_augmented.jsonl`
- `${DATA_DIR}/${DATASET_PREFIX}_test_augmented.jsonl`

Both scripts keep Colab compatibility while avoiding cross-protocol file collisions.

`scripts/05_train_phobert.py`:

- base model: `vinai/phobert-base`
- `MAX_LENGTH = 128`
- `BATCH_SIZE = 16`
- `EPOCHS = 4`
- `LR = 2e-5`
- custom `WeightedTrainer` with weighted cross-entropy
- best model selected by `macro_f1`
- MLflow logging is enabled

Important note:

- this script writes to `models/phobert/` and `results/phobert/`
- but the backend / inference path currently reads checkpoints from `models/options/phobert/` or `VIETTOXIC_MODEL_OPTIONS_DIR`
- this is an **important mismatch** in the current repo

### Existing optimization log

`experiments/phobert_optimization_log.md` is more important than the static UI text.

Current information there:

- baseline LR test macro F1: `0.7043`
- PhoBERT exp-01 test macro F1: `0.7171`
- PhoBERT exp-01 test F1_toxic: `0.4928`
- best validation F1_toxic in the log:
  - `v1`: `0.5514`
  - `v2`: `0.5504`

If there is a conflict between static UI text and this log, trust the log + source code.

Note: model options now live under `models/options/` (with type/name IDs like `phobert/v1`, `tfidf_lr/baseline`).

---

## 7. Model directory and model selection

The backend and inference script resolve checkpoints in this order:

1. `VIETTOXIC_MODEL_OPTIONS_DIR` if the env var exists
2. otherwise local repo fallback: `models/options`

`get_default_model()` logic:

- if model `v2` exists, prefer `v2`
- otherwise choose the first entry from `sorted()`

Repo snapshot on 2026-03-22:

- `models/options/phobert/phobert`
- `models/options/phobert/phobert_lora_latest`
- `models/options/phobert/v1`
- `models/options/tfidf_lr/baseline`

There is **no `v2` in the repo snapshot**.
Because the code falls back to the first `sorted()` entry, the current local default in-repo is effectively `phobert/phobert`.
However, the user machine may still expose additional model directories through `VIETTOXIC_MODEL_OPTIONS_DIR`.

A valid checkpoint directory must contain at least:

- `config.json`
- `model.safetensors` or `pytorch_model.bin`

And often also:

- `vocab.txt`
- `bpe.codes`
- `tokenizer_config.json`
- `added_tokens.json`
- `threshold.json` in some model folders
- `temperature_scaling.json` in calibrated model folders such as `phobert_lora_latest`

Important caveat:

- `backend/app.py` currently gets serving thresholds from `CATEGORY_THRESHOLDS` + feedback-driven overrides in `feedback.db`
- it does **not** automatically read per-model `threshold.json` for domain thresholds

---

## 8. Crawl pipelines

### Comment-only crawl pipeline (latest)

Main file: `comment_crawl.py`

#### Goal

Take a URL, extract **only the comment section** (not article body), and write artifacts in the same schema as `setup_and_crawl.py`.

#### Supported sources

| Source type | Strategy | Status |
|---|---|---|
| Vietnamese news sites | Selenium + domain-specific CSS selectors | Working (VNExpress tested) |
| Facebook posts | mbasic.facebook.com + optional cookie injection | Implemented, untested |
| X / Twitter | Not supported | — |

#### Supported news domains (dedicated CSS selectors)

`vnexpress.net`, `tuoitre.vn`, `thanhnien.vn`, `dantri.com.vn`, `vietnamnet.vn`.
Unknown news domains fall back to heuristic selectors.

#### Browser detection (cross-platform)

Preference order: Edge → Chrome.

- **Windows**: searches known install paths + `shutil.which()`
- **macOS**: checks `/Applications/` paths + `shutil.which()`
- **Linux / Ubuntu VM**: `shutil.which()` for `microsoft-edge-stable`, `google-chrome-stable`, `chromium-browser`, etc.

Edge uses `selenium.webdriver.Edge` (selenium-manager auto-downloads msedgedriver).
Chrome uses `undetected_chromedriver` for anti-bot patching.

#### Output

- `data/raw/crawled_urls/<url_hash>/segments.jsonl`
- `data/raw/crawled_urls/<url_hash>/extracted.txt`
- `data/raw/crawled_urls/<url_hash>/meta.json`

`html_tag_effective` defaults to `"comment"` (vs `"body"` in the article pipeline).

#### CLI

```bash
python comment_crawl.py "https://vnexpress.net/some-article-123.html"
python comment_crawl.py "https://facebook.com/..." --fb-cookies cookies.json --no-headless
python comment_crawl.py "https://..." --max-load-more 20
```

---

### Article + video crawl pipeline (deprecated for comment use case)

Main file: `setup_and_crawl.py`

#### Goal

Take a web URL, crawl text, split it into segments, optionally fetch video / transcript data, and write artifacts to disk.

#### Text crawl logic

`crawl_and_save()` does the following:

1. use `trafilatura.fetch_url()`
2. run `trafilatura.extract(..., include_comments=True)`
3. if text is missing / short (`< 800` chars):
   - with `allow_selenium_fallback=True`: continue to Selenium
   - with `allow_selenium_fallback=False`: return `needs_fallback_confirmation` metadata (no Selenium yet)
4. Selenium scrolls the page and extracts again (only when allowed)
5. if text is still `< 200` chars, mark as failure
6. on success, segment with VnCoreNLP; if VnCoreNLP fails, fall back to regex sentence splitting
7. save:
   - `extracted.txt`
   - `segments.jsonl`
   - `meta.json`

#### Video pipeline

Current crawler defaults:

- `enable_video=True`
- `enable_asr=True`

If HTML is available:

- detect YouTube IDs from iframe / anchor / regex
- fetch transcript through `youtube_transcript_api` if possible
- fall back to `yt-dlp` for metadata / captions
- for native video, ASR may run through `faster-whisper`

Possible video artifacts:

- `video_data.jsonl`
- `videos/` folder if `keep_artifacts=True`

#### Runtime prerequisites for video / ASR

For the video/ASR pipeline to work well, the environment may need:

- `yt-dlp`
- `ffmpeg`
- `faster-whisper`
- working system / network dependencies

The backend currently exposes only `enable_video`; it does **not expose `enable_asr`**.
So when the backend calls `crawl_urls(..., enable_video=True)`, `enable_asr` still remains `True` by default.

---

## 9. Actual inference pipeline

Main file: `infer_crawled_local.py`

### Expected input

`data_dir` should contain multiple subfolders, one per URL:

- `<url_hash>/meta.json`
- `<url_hash>/segments.jsonl`

`segments.jsonl` schema:

```json
{
  "text": "...",
  "segment_index": 0,
  "url_hash": "<md5_of_url>",
  "segment_hash": "<sha256(normalized_text + '|' + html_tag_effective)>"
}
```

Notes:

- `text` remains the compatibility field consumed by older readers.
- At crawl/segmentation time, `html_tag_effective` defaults to `"body"` for `segment_hash` generation (same hash formula as `backend/app.py` + `infer_crawled_local.py`).

### Device selection

`pick_device()` prefers:

1. CUDA
2. MPS
3. CPU

### Tokenizer / model

- default tokenizer: `vinai/phobert-base`
- local checkpoint is loaded with `local_files_only=True`

### Segment-level prediction

For each segment:

- tokenize
- run softmax
- use `prob[:, 1]` as `P(toxic)`

### Page-level aggregation

For each page:

- `toxic_label = 1 if toxic_prob > effective_threshold else 0`
- `toxic_ratio = toxic_segments / total_segments`
- `page_toxic = 1 if toxic_ratio > page_threshold else 0`
- `avg_toxic_prob = mean(segment toxic prob)`

Top 5 highest-probability segments are stored for debugging.

### Output files

`infer_crawled()` writes:

- `crawled_predictions.jsonl`
- `page_level_results.json`
- `page_level_results.csv`

Important page-level fields:

- `url_hash`
- `url`
- `domain_category`
- `seg_threshold_used`
- `method`
- `status`
- `total_segments`
- `toxic_segments`
- `toxic_ratio`
- `page_toxic`
- `avg_toxic_prob`
- `top5_toxic_segments`

Important segment-level fields:

- `url_hash`
- `url`
- `domain_category`
- `seg_threshold_used`
- `text`
- `toxic_prob`
- `toxic_label`

There is also `debug_force_prob` for sanity checking threshold behavior without loading a real model.

---

## 10. Domain-aware thresholding

`domain_classifier.py` is the source of truth for this logic.

Categories:

- `news`
- `social`
- `forum`
- `unknown`

Default thresholds in code:

- `news = 0.72`
- `social = 0.50`
- `forum = 0.60`
- `unknown = 0.62`

There are 3 heuristic layers:

1. exact domain whitelist
2. strip subdomain and try parent domain
3. regex heuristics on domain name + path patterns

Formal news-like content uses a higher threshold to reduce false positives.

Committed sanity-check assets:

- `data/raw/mock_crawled_urls/`
- `data/processed/mock_debug_060/`
- `data/processed/mock_debug_099/`

The `debug_force_prob=0.60` case shows:

- social threshold `0.50` -> toxic
- news threshold `0.72` -> non-toxic

This is an important proof point for the domain-aware threshold design.

---

## 11. FastAPI backend

Main file: `backend/app.py`

### Endpoints

- `GET /`
  - simple health text
- `GET /health`
  - `{ "status": "ok" }`
- `GET /api/models`
  - list available model directories
- `POST /api/analyze`
  - crawl + infer + return results to the UI
- `POST /api/analyze_compare`
  - run the same crawl against 2+ selected models and return per-model result bundles
- `POST /api/feedback`
  - store page-level human labels
- `POST /api/feedback/segment`
  - store segment-level human labels
- `POST /api/feedback/segment/delete`
  - delete selected segment-feedback rows by id
- `POST /api/thresholds/preview`
  - compute suggested per-domain thresholds from page feedback
- `POST /api/thresholds/apply`
  - persist EMA-smoothed threshold overrides
- `POST /api/thresholds/current`
  - read current effective thresholds + saved overrides for a model
- `GET /api/dataset/preview`
  - paginate combined dataset + collected feedback rows
- `POST /api/dataset/export`
  - write filtered rows to `data/processed/combined_dataset.jsonl`
- `POST /api/ask-ai`
  - send result context to Gemini and return a short explanation
- `GET /api/gemini/models`
  - list available Gemini models for the configured API key

### Current CORS

Allows:

- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `https://*.ngrok-free.app`

### `/api/analyze` request schema

```json
{
  "urls": ["https://example.com"],
  "options": {
    "batch_size": 8,
    "max_length": 256,
    "page_threshold": 0.25,
    "seg_threshold": 0.4,
    "model_name": "phobert/v1",
    "model_path": null,
    "enable_video": false,
    "selenium_fallback_mode": "auto"
  },
  "pending_job_id": null,
  "fallback_decisions": []
}
```

`selenium_fallback_mode` supports:

- `auto`: keep old behavior (auto Selenium when trafilatura text is short)
- `ask`: detect URLs that may need Selenium, return them for user decision, then resume with `pending_job_id` + `fallback_decisions`

### `/api/analyze` flow

1. validate request
2. create `job_id`
3. create output dir `data/processed/job_<job_id>`
4. resolve model from `model_name` or `model_path`
5. crawl step:
   - `selenium_fallback_mode="auto"`: call `crawl_urls()` with normal Selenium fallback
   - `selenium_fallback_mode="ask"`: run detect-only crawl first; if needed, return `flow_state="awaiting_user_choice"` + `pending_fallback_urls`
6. resume step (only for ask mode): client sends `pending_job_id` + `fallback_decisions`, backend applies per-URL decisions (`use_selenium` or `skip`)
7. if `enable_video=True` and `video_data.jsonl` exists, merge text segments + transcripts into `merged_crawl/`
8. resolve `thresholds_by_domain` from `CATEGORY_THRESHOLDS` + saved overrides in `feedback.db`
9. call `infer_crawled()`
10. read page-level + segment-level artifacts
11. map everything back by URL for the response

The backend also deletes old `data/processed/job_*` folders using `JOB_RETENTION_HOURS` (default 24h).

### Response schema overview

```json
{
  "job_id": "uuidhex",
  "flow_state": "completed",
  "model_name": "phobert/v1",
  "thresholds": {
    "seg_threshold": 0.4,
    "page_threshold": 0.25
  },
  "thresholds_by_domain": {
    "news": 0.72,
    "social": 0.5,
    "forum": 0.6,
    "unknown": 0.62
  },
  "results": [
    {
      "url": "https://example.com",
      "url_hash": "<hash>",
      "status": "ok",
      "error": null,
      "warnings": [],
      "crawl_output_dir": "data/raw/crawled_urls/<hash>",
      "segments_path": "data/raw/crawled_urls/<hash>/segments.jsonl",
      "videos": [],
      "domain_category": "news",
      "seg_threshold_used": 0.72,
      "page_toxic": 0,
      "toxicity": {
        "overall": 0.31,
        "by_segment": [
          {
            "segment_id": "<hash>:0",
            "score": 0.82,
            "text_preview": "...",
            "text": "...",
            "domain_category": "news",
            "seg_threshold_used": 0.72
          }
        ]
      }
    }
  ]
}
```

Important notes:

- `overall` prefers `avg_toxic_prob`, and only falls back to `toxic_ratio`
- the backend returns `videos` loaded from `video_data.jsonl`
- `job_id` is an important debugging handle
- in ask mode, backend can return an intermediate payload:
  - `flow_state: "awaiting_user_choice"`
  - `pending_fallback_urls: [{url, url_hash, reason, trafilatura_text_len}]`
- page / segment feedback is stored in `data/processed/feedback/feedback.db`

---

## 12. Current frontend

The frontend lives in `comprehensive_ui/`.

### Actual frontend wiring

The real backend-connected frontend logic is in:

- `src/app/App.tsx`
- `src/app/components/HomePage.tsx`
- `src/app/components/ResultsPage.tsx`
- `src/app/components/DatasetPage.tsx`
- `src/app/components/Navigation.tsx`
  - top bar, including dark/light toggle UI

### App shell (`App.tsx`)

- handles two-step Selenium decision flow in ask mode with in-app popup (no browser `confirm`)
- stores scan history in `localStorage` (`viettoxic:scan-history`)
- stores theme preference in `localStorage` (`viettoxic:theme`)
- applies dark mode via `document.documentElement.classList.toggle("dark", ...)`

### API base

- read from `VITE_API_BASE_URL`
- if unset, use relative `/api/...`
- Vite dev server proxies `/api` and `/health` to `http://127.0.0.1:8000`

### HomePage

- fetches `GET /api/models` on mount
- accepts one or more URLs
- allows single-model selection
- supports compare mode and sends `POST /api/analyze_compare` when 2+ models are selected
- stores the last selected single model in `localStorage` under `viettoxic:model`
- sends `POST /api/analyze` in normal mode
- currently hardcodes:
  - `batch_size: 8`
  - `max_length: 256`
  - `page_threshold: 0.25`
  - `seg_threshold: 0.4`
  - `enable_video: true`
  - `selenium_fallback_mode: "ask"`
- loading UX now shows animated percentage progress (UI-driven ramp) instead of a static spinner while waiting for API response

### ResultsPage

- displays `overall` score as a percent
- counts toxic segments using `result.seg_threshold_used` when present
- shows the top 3 highest-score segments
- uses `page_toxic` if present; otherwise falls back to `overall >= page_threshold`
- can submit page feedback, segment feedback, preview/apply current thresholds, and call `/api/ask-ai`
- when compare results are present, `App.tsx` lets the user switch the displayed model payload without rerunning the crawl
- explicitly renders skipped URLs with neutral amber status message (`status: "skipped"`) when user chooses not to chuyển qua Selenium

### DatasetPage

- uses `GET /api/dataset/preview` to inspect `data/victsd/*.jsonl` plus collected feedback rows
- uses `POST /api/dataset/export` to write `data/processed/combined_dataset.jsonl`
- can delete selected segment-feedback rows through `POST /api/feedback/segment/delete`

### Static / mock pages

`ModelPage.tsx` and `ContactPage.tsx` are currently mostly illustrative:

- metrics are hardcoded / approximate
- charts use mock data
- contact links are placeholders

Do not treat them as the source of truth for research results.

---

## 13. Quick runbook

### Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

Or:

```bash
./scripts/run_backend.sh
```

Notes:

- `scripts/run_backend.sh` does **not** automatically activate `venv` / `.venv`
- `scripts/run_server_ngrok.sh` is the one that does activate a virtualenv before running

### Frontend

```bash
cd comprehensive_ui
npm install
npm run dev
```

Or:

```bash
VITE_API_BASE_URL=https://living-rare-ram.ngrok-free.app ./scripts/run_ui.sh
```

### Full local demo

1. run backend
2. run frontend
3. open the UI on port `5173`

### Ngrok

Relevant scripts:

- `scripts/run_server_ngrok.sh`
- `scripts/run_ngrok_all.sh`

The README contains the public demo instructions through ngrok.

---

## 14. Plot generation for the thesis

`generate_thesis_plots.py` is a lightweight script to generate figures without retraining.

It:

- reads dataset JSONL files
- locates `metrics.json` if available
- writes:
  - `plots/label_distribution.png`
  - `plots/comment_length_hist.png`
  - `plots/confusion_matrix.png`

Important note:

- the script may default to `data/victsd`
- while the training scripts use `data/processed/victsd_v1`
- always verify the intended data source before changing the plot script

---

## 15. Important gotchas

0. Protocol outputs must be treated as versioned artifacts.
   - Do not reuse generic `train_augmented.jsonl` naming across A/B/C in shared directories.
   - Use `DATASET_PREFIX` consistently in Colab runs to avoid accidental overwrites.

1. The README, static UI text, and actual code are not fully synchronized.
   - If they conflict, trust the source code + real artifacts.

2. `scripts/05_train_phobert.py` writes to `models/phobert`, but serving reads from `models/options/phobert`.
   - If training a new model for the API, resolve this path mismatch explicitly.

3. In this repo snapshot there is no `v2` under `models/options/phobert`.
   - The in-repo default is currently `phobert/phobert`, not `phobert/v1`.

4. `enable_video=True` from the frontend can trigger transcript + ASR paths as well.
   - If the API feels slow, think about crawler/video work, not only model inference.

5. `comprehensive_ui/dist/` is build output.
   - Edit `src/`, not `dist/`.

6. Threshold artifacts inside model folders (`threshold.json`, `temperature_scaling.json`) are not the same thing as the backend's domain-threshold overrides.
   - Serving currently uses `CATEGORY_THRESHOLDS` + SQLite overrides from feedback.

7. There is no formal automated test suite in the repo.
   - Verification usually means running scripts, calling the API, and testing the UI manually.

8. Protocol build output currently confirms:
   - B-train toxic ratio ~0.3016 (inside target 0.30–0.40)
   - B has zero train-vs-val/test overlap
   - C has zero overlap across train/validation/test
   - A preserves ViCTSD behavior and may retain original cross-split duplicates from source data

9. The repo contains many generated artifacts and the worktree may be dirty.
   - Distinguish source files from outputs / data before editing.

---

## 16. Guidance for Claude when coding in this repo

When asked to make changes, Claude should:

1. Identify which lane the task belongs to:
   - research / training
   - crawling / inference / backend
   - frontend demo

2. Read the source-of-truth files before editing.

3. For inference / API tasks:
   - preserve response-schema backward compatibility if possible
   - do not accidentally break artifact paths used by the backend / UI
   - remember domain-aware thresholds

4. For frontend tasks:
   - edit `comprehensive_ui/src`
   - validate against the real API contract, not the mock text in `ModelPage`

5. For training tasks:
   - confirm where the output model must be stored for serving
   - align model paths if needed

6. For crawling tasks:
   - preserve compatibility for `data/raw/crawled_urls/<hash>/segments.jsonl` (`text` must remain available)
   - current segment rows include `text`, `segment_index`, `url_hash`, `segment_hash` (with crawl-time default html tag `body`)
   - ensure `meta.json` includes `url_hash` for cross-reference
   - be careful with system dependencies like Java, Chrome, yt-dlp, and ffmpeg

7. If information conflicts:
   - source code > experiment logs > README > UI text

---

## 17. Decision workflow for thesis protocol selection

After running A/B/C, use this order:

1. **Gating rules** (elimination):
   - leakage constraints
   - reproducibility evidence (script + report)
   - deploy feasibility
2. **Weighted scoring** (100 points):
   - F1_toxic (30), Macro-F1 (20), ECE (10), Brier (5), Robustness (15), Stability (10), Practicality (5), Scientific clarity (5)
3. **Tie-break**:
   - `F1_toxic > Macro-F1 > ECE > Robustness`

Interpretation:

- A is anchor baseline
- B is primary deploy candidate on ViCTSD-comparable evaluation
- C is new benchmark contribution and should be reported alongside A/B

## 18. Docker & CI/CD setup (added 2026-04-06)

### File layout

```
CrawlingData/
├── .dockerignore
├── requirements-ml.txt          # torch ecosystem (heavy layer, cached separately)
├── requirements-base.txt        # fastapi, crawling libs, mlflow, etc.
├── docker-compose.yml
├── backend/
│   └── Dockerfile               # CPU-only torch; GPU notes in TODO comments
└── comprehensive_ui/
    ├── Dockerfile                # node:20-slim build → nginx:1.27-alpine serve
    └── nginx.conf                # SPA fallback + /api → backend:8000 proxy
.github/workflows/
├── ci.yml                        # lint + build check on every PR / push
└── build.yml                     # multi-platform image push on merge to main
```

### Requirements split rationale

`requirements.txt` (the original monolith) is **not used by Docker**.
Docker uses two files for layer-cache efficiency:

- `requirements-ml.txt` — torch ecosystem, changes rarely → installed as its own layer
- `requirements-base.txt` — fastapi, crawling, mlflow, etc.

`torch==2.9.1` is installed separately before both files via:
```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
```
This forces CPU-only wheel and prevents the 2.5 GB CUDA wheel from being pulled.

### Docker Compose quick reference

```bash
docker compose up --build     # first time (slow: torch ~800 MB download)
docker compose up             # subsequent (fast: layers cached)
docker compose down           # stop
docker compose down -v        # stop + wipe sqlite_data volume
```

Prerequisites before first run:
- PhoBERT weights at `./models/options/<model-name>/` (mounted read-only to `/app/models`)
- VnCoreNLP JARs already in `./vncorenlp/` (mounted read-only)
- SQLite feedback DB is auto-created inside `sqlite_data` named volume

### CI workflow (ci.yml)

Trigger: `push` or `pull_request` to `main` / `develop`

Jobs:
- `backend-checks`: Python 3.12, installs torch CPU + `requirements-ml.txt` + `requirements-base.txt`, runs `ruff` lint and `pytest` (pytest is `continue-on-error` until a `tests/` dir exists)
- `frontend-checks`: Node 20, `npm install`, `npm run build` (verifies TypeScript/bundler errors)

Caching: `actions/setup-python` with `cache: 'pip'` — after first run, pip deps are cached.

### Build workflow (build.yml)

Trigger: `push` to `main` only

Builds and pushes multi-platform images (`linux/amd64` + `linux/arm64`) to GHCR.

**Before first push**: replace `<your-github-username>` in the `env:` block of `build.yml`.

Images:
- `ghcr.io/<user>/viettoxic-backend:latest` and `:<sha>`
- `ghcr.io/<user>/viettoxic-frontend:latest` and `:<sha>`

Layer cache: `cache-from/to: type=gha` — first build ~15–20 min, subsequent ~2–3 min.

### GPU upgrade path

Every file that needs changing for GPU is marked with `TODO(gpu-upgrade):` comments:
- `backend/Dockerfile` — swap base image + torch index URL
- `docker-compose.yml` — add `deploy.resources.reservations.devices` (nvidia)
- `.github/workflows/ci.yml` — no change needed (CI stays CPU-only)

---

## 19. Short summary

Remember that this repo is a combination of:

- a research pipeline for Vietnamese toxic text classification
- and a demo app that crawls URLs and runs local PhoBERT inference

The most important current source-of-truth files are:

- `backend/app.py`
- `infer_crawled_local.py`
- `setup_and_crawl.py`
- `domain_classifier.py`
- `scripts/02_preprocess.py`
- `scripts/04_baseline_tfidf_lr.py`
- `scripts/05_train_phobert.py`
- `experiments/phobert_optimization_log.md`
- `comprehensive_ui/src/app/App.tsx`

If you must choose between "what the docs say" and "what the code actually does", follow **the running code**.
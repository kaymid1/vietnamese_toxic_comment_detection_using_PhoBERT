# CLAUDE.md

## Purpose of this file

This file is the "base knowledge" guide so Claude can work in the right direction inside this repository.
It prioritizes **what the code currently does** over general descriptions in the README.

This snapshot was compiled from the repo on **2026-03-16**.
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
- output: binary `label`
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
- `setup_and_crawl.py`
  - hybrid crawler, video transcript handling, ASR
- `domain_classifier.py`
  - domain classification and threshold selection
- `scripts/01_export_raw.py`
  - export raw ViCTSD
- `scripts/02_preprocess.py`
  - preprocess dataset into `data/processed/victsd_v1`
- `scripts/03_eda.py`
  - EDA
- `scripts/04_baseline_tfidf_lr.py`
  - baseline model
- `scripts/05_train_phobert.py`
  - PhoBERT fine-tuning
- `comprehensive_ui/src/app/App.tsx`
  - frontend root app
- `comprehensive_ui/src/app/components/HomePage.tsx`
  - URL input and model selection
- `comprehensive_ui/src/app/components/ResultsPage.tsx`
  - inference result display

These files / folders are **not the main source**:

- `backup_infer_crawled_local.py`
  - backup copy, not the main inference file
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
- `trafilatura`
- `selenium` + `undetected-chromedriver`
- `vncorenlp`
- `yt-dlp`
- `youtube-transcript-api`
- `faster-whisper` + `ffmpeg` for native video ASR

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
- `data/processed/victsd_v1/`
  - preprocessed dataset used by the baseline + PhoBERT scripts
- `data/victsd/`
  - another simplified dataset copy with schema `{"text","label"}`
- `data/raw/crawled_urls/<url_hash>/`
  - crawl artifact for each URL
- `data/raw/mock_crawled_urls/`
  - mock crawl inputs for inference sanity checks
- `data/processed/job_<uuid>/`
  - output for each `/api/analyze` API request

### Models / results

- `models_2/phobert/`
  - local checkpoints used by the backend / inference script
- `results/baseline/`
  - baseline metrics + serialized LR/vectorizer
- `results/phobert/`
  - expected output from the training script, but there are no committed metrics here in the current repo snapshot
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

Schema:

```json
{
  "text": "...",
  "label": 0,
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

Why:

- PhoBERT is being used in a way that keeps text close to the original form
- sarcasm, punctuation, emojis, and teencode can all be important toxicity signals

---

## 6. Research / training pipeline

The reasonable working order is:

1. `scripts/01_export_raw.py`
2. `scripts/02_preprocess.py`
3. `scripts/03_eda.py`
4. `scripts/04_baseline_tfidf_lr.py`
5. `scripts/05_train_phobert.py`
6. `generate_thesis_plots.py`

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

`scripts/04_baseline_tfidf_lr.py` uses:

- `TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=r"(?u)\\b\\w+\\b")`
- `LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1)`

Artifacts:

- `results/baseline/metrics.json`
- `results/baseline/vectorizer.pkl`
- `results/baseline/model_lr.pkl`

Metrics currently committed in the repo:

- validation macro F1: `0.6927`
- validation F1_toxic: `0.4618`
- test macro F1: `0.7043`
- test F1_toxic: `0.4844`

### PhoBERT training script

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
- but the backend / inference path currently reads checkpoints from `models_2/phobert/` or `VIETTOXIC_MODEL_BASE_DIR`
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

---

## 7. Model directory and model selection

The backend and inference script resolve checkpoints in this order:

1. `VIETTOXIC_MODEL_BASE_DIR` if the env var exists
2. otherwise local repo fallback: `models_2/phobert`

`get_default_model()` logic:

- if model `v2` exists, prefer `v2`
- otherwise choose the first entry from `sorted()`

Repo snapshot on 2026-03-16:

- `models_2/phobert/backup`
- `models_2/phobert/backup_no2`
- `models_2/phobert/new`
- `models_2/phobert/v1`

There is **no `v2` in the repo snapshot**, so the local default may not match the README.
However, the user machine may still have `/models_2/phobert/v2` outside the repo via env var.

A valid checkpoint directory must contain at least:

- `config.json`
- `model.safetensors` or `pytorch_model.bin`

And often also:

- `vocab.txt`
- `bpe.codes`
- `tokenizer_config.json`
- `added_tokens.json`
- `threshold.json` in some model folders

---

## 8. Actual crawl pipeline

Main file: `setup_and_crawl.py`

### Goal

Take a web URL, crawl text, split it into segments, optionally fetch video / transcript data, and write artifacts to disk.

### Text crawl logic

`crawl_and_save()` does the following:

1. use `trafilatura.fetch_url()`
2. run `trafilatura.extract(..., include_comments=True)`
3. if text is missing / short (`< 800` chars), fall back to Selenium
4. Selenium scrolls the page and extracts again
5. if text is still `< 200` chars, mark as failure
6. on success, segment with VnCoreNLP; if VnCoreNLP fails, fall back to regex sentence splitting
7. save:
   - `extracted.txt`
   - `segments.jsonl`
   - `meta.json`

### Video pipeline

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

### Runtime prerequisites for video / ASR

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
{"text": "..."}
```

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
    "model_name": "v1",
    "model_path": null,
    "enable_video": false
  }
}
```

### `/api/analyze` flow

1. validate request
2. create `job_id`
3. create output dir `data/processed/job_<job_id>`
4. resolve model from `model_name` or `model_path`
5. call `crawl_urls()`
6. if `enable_video=True` and `video_data.jsonl` exists, merge text segments + transcripts into `merged_crawl/`
7. call `infer_crawled()`
8. read page-level + segment-level artifacts
9. map everything back by URL for the response

### Response schema overview

```json
{
  "job_id": "uuidhex",
  "model_name": "v1",
  "thresholds": {
    "seg_threshold": 0.4,
    "page_threshold": 0.25
  },
  "results": [
    {
      "url": "https://example.com",
      "status": "ok",
      "error": null,
      "crawl_output_dir": "data/raw/crawled_urls/<hash>",
      "segments_path": "data/raw/crawled_urls/<hash>/segments.jsonl",
      "videos": [],
      "toxicity": {
        "overall": 0.31,
        "by_segment": [
          {
            "segment_id": "<hash>:0",
            "score": 0.82,
            "text_preview": "...",
            "text": "..."
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

---

## 12. Current frontend

The frontend lives in `comprehensive_ui/`.

### Actual frontend wiring

The real backend-connected frontend logic is in:

- `src/app/App.tsx`
- `src/app/components/HomePage.tsx`
- `src/app/components/ResultsPage.tsx`

### API base

- read from `VITE_API_BASE_URL`
- if unset, use relative `/api/...`
- Vite dev server proxies `/api` and `/health` to `http://127.0.0.1:8000`

### HomePage

- fetches `GET /api/models` on mount
- accepts one or more URLs
- allows model selection
- sends `POST /api/analyze`
- currently hardcodes:
  - `batch_size: 8`
  - `max_length: 256`
  - `page_threshold: 0.25`
  - `seg_threshold: 0.4`
  - `enable_video: true`

### ResultsPage

- displays `overall` score as a percent
- counts toxic segments using `thresholds.seg_threshold`
- shows the top 3 highest-score segments
- the UI "toxic/safe" color state is currently inferred from `overall > 50%`, not from `page_threshold`

### Important frontend caveat

`ResultsPage` uses the global `thresholds.seg_threshold` from the API response,
but the actual inference logic may use per-domain `seg_threshold_used`.
If semantic correctness matters, the UI should receive and use the effective threshold per URL or per segment.

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

1. The README, static UI text, and actual code are not fully synchronized.
   - If they conflict, trust the source code + real artifacts.

2. `scripts/05_train_phobert.py` writes to `models/phobert`, but serving reads from `models_2/phobert`.
   - If training a new model for the API, resolve this path mismatch explicitly.

3. In this repo snapshot there is no `v2` under `models_2/phobert`.
   - The actual default model may differ from the README.

4. `enable_video=True` from the frontend can trigger transcript + ASR paths as well.
   - If the API feels slow, think about crawler/video work, not only model inference.

5. `comprehensive_ui/dist/` is build output.
   - Edit `src/`, not `dist/`.

6. `backup_infer_crawled_local.py` is only a backup.
   - Do not accidentally patch that file when the task is about main inference.

7. There is no formal automated test suite in the repo.
   - Verification usually means running scripts, calling the API, and testing the UI manually.

8. The repo contains many generated artifacts and the worktree may be dirty.
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
   - preserve `data/raw/crawled_urls/<hash>/meta.json` + `segments.jsonl` format
   - be careful with system dependencies like Java, Chrome, yt-dlp, and ffmpeg

7. If information conflicts:
   - source code > experiment logs > README > UI text

---

## 17. Short summary

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

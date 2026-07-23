# Crawler + Inference Audit (for self-learning / pseudo-label readiness)

## 1. Output Format Inventory

### A. Crawl-time artifacts (per URL)

| Artifact | Path pattern | Producer | Main schema/contents | Current consumers |
|---|---|---|---|---|
| Raw HTML snapshot | `data/raw/crawled_urls/<url_hash>/raw.html` | `setup_and_crawl.py:crawl_and_save()` | Raw HTML string of fetched page (if available) | `infer_crawled_local.py` via `html_dir` (reads `<url_hash>/raw.html`) for domain threshold metadata extraction |
| Extracted text | `data/raw/crawled_urls/<url_hash>/extracted.txt` | `setup_and_crawl.py:crawl_and_save()` | Full extracted page text | Currently no strict downstream parser in backend; debug/manual use |
| Segment file (core) | `data/raw/crawled_urls/<url_hash>/segments.jsonl` | `setup_and_crawl.py:crawl_and_save()` using `build_crawl_segment_record()` | JSONL rows: `{text, segment_index, url_hash, segment_hash}` | `infer_crawled_local.py` (loads `text` only), `backend/app.py` (`build_merged_segments`) |
| Crawl metadata | `data/raw/crawled_urls/<url_hash>/meta.json` | `setup_and_crawl.py:crawl_and_save()` | Success case: `{url,url_hash,timestamp,num_segments,duration_sec,method,text_length,status,warnings?}`; failure case includes `{status:"failed",...}` | `infer_crawled_local.py` reads `url`, `method`, `status`; backend merge flow copies this file |
| Video/transcript data (optional) | `data/raw/crawled_urls/<url_hash>/video_data.jsonl` | `setup_and_crawl.py:crawl_and_save()` (video pipeline) | JSONL video records with metadata + `transcript` array (`[{text,start,duration}, ...]`) | `backend/app.py:load_video_results()` and `build_merged_segments()` (append transcript text into merged segments) |
| Optional video/ASR artifacts | `data/raw/crawled_urls/<url_hash>/videos/**` | `setup_and_crawl.py` when `keep_artifacts=True` | yt-dlp info, captions, ASR temp outputs | Not required by API inference path |

### B. Job-level artifacts (per analyze job)

| Artifact | Path pattern | Producer | Main schema/contents | Current consumers |
|---|---|---|---|---|
| Job metadata | `data/processed/job_<job_id>/job_meta.json` | `backend/app.py:save_job_meta()` | `{job_id,created_at,urls,url_hashes,model_ids,enable_video,merged_used}` | Resume/rerun endpoints |
| Pending crawl cache (ask-mode) | `data/processed/job_<job_id>/crawl_pending_results.json` | `backend/app.py:save_pending_crawl_results()` | List of crawl result dicts, including `needs_fallback_confirmation` entries | `/api/analyze` and `/api/analyze_compare` resume flow |
| Merged crawl folder (optional) | `data/processed/job_<job_id>/merged_crawl/<url_hash>/meta.json` | `backend/app.py:build_merged_segments()` (copied from raw crawl folder) | Same as crawl `meta.json` | Inference when `enable_video` and merged available |
| Merged segments (optional) | `data/processed/job_<job_id>/merged_crawl/<url_hash>/segments.jsonl` | `backend/app.py:build_merged_segments()` | JSONL rows: only `{"text": ...}` (no hash/index/url metadata) | `infer_crawled_local.py` (text-only loader) |
| Segment predictions | `data/processed/job_<job_id>/crawled_predictions.jsonl` | `infer_crawled_local.py:infer_crawled()` | Per-segment rows: includes `url_hash,url,text,toxic_prob,toxic_prob_adjusted,toxic_label,segment_hash,context_segment_hash,html_tags,og_types,seg_threshold_used,...` | `backend/app.py:load_segment_results()` -> response mapping |
| Page predictions (JSON) | `data/processed/job_<job_id>/page_level_results.json` | `infer_crawled_local.py` | Per-page rows: `url_hash,url,seg_threshold_used,effective_threshold,domain/formality fields,total_segments,toxic_segments,toxic_ratio,page_toxic,avg_toxic_prob,...` | `backend/app.py:load_page_results_map()` |
| Page predictions (CSV) | `data/processed/job_<job_id>/page_level_results.csv` | `infer_crawled_local.py` | CSV projection of page-level results | Backend fallback reader if JSON unavailable |

### C. Compare-mode per-model outputs

| Artifact | Path pattern | Producer | Contents |
|---|---|---|---|
| Model-specific infer outputs | `data/processed/job_<job_id>/models/<model_id_with_dash>/{crawled_predictions.jsonl,page_level_results.json,page_level_results.csv}` | `backend/app.py:/api/analyze_compare` + `infer_crawled_local.py` | Same schema as single-model infer artifacts |

---

## 2. segment_hash Consistency

### 2.1 Công thức ở `infer_crawled_local.py` (exact code)

```python
def normalize_segment_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()

def build_segment_hash(text: str, html_tag: str) -> str:
    base = f"{normalize_segment_text(text)}|{(html_tag or '').strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()
```

Ngoài ra có context hash:

```python
def build_context_segment_hash(prev_text, text, next_text, html_tag):
    base = "|".join([
        normalize_segment_text(prev_text),
        normalize_segment_text(text),
        normalize_segment_text(next_text),
        (html_tag or "").strip().lower(),
    ])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()
```

### 2.2 Công thức ở `backend/app.py` khi lookup/store feedback

`backend/app.py` có local function trùng logic:

```python
def normalize_segment_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()

def build_segment_hash(text: str, html_tag: str) -> str:
    base = f"{normalize_segment_text(text)}|{(html_tag or '').strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()
```

Khi ghi feedback segment (`/api/feedback/segment`), backend tạo:

```python
"segment_hash": build_segment_hash(item.text, item.html_tag_override or item.html_tag),
"context_segment_hash": item.context_segment_hash,
```

Khi dedupe/lookup learned feedback, backend ưu tiên:

- `context_segment_hash` nếu có,
- fallback `segment_hash`,
- kèm tag key `LOWER(COALESCE(html_tag_override, html_tag, ''))`.

### 2.3 Hai công thức có identical không?

- **Có** cho `segment_hash`: normalize + `|html_tag_lower` + `sha256(utf-8)` giống nhau.
- **Nhưng semantic input khác nhau theo stage**:
  - Crawl-time hash dùng `html_tag_effective="body"` (hardcoded default trong crawler).
  - Inference-time hash dùng `html_tag` lấy từ `threshold_info["html_tags"][0]` (ví dụ `unknown`, `article`, schema/og tag).
  - Vì vậy hash value có thể khác dù text giống nhau.

### 2.4 Nếu `segments.jsonl` chưa có `segment_hash`, hash được tính lần đầu khi nào?

Trong code hiện tại, crawler **đã ghi sẵn** `segment_hash` vào `segments.jsonl`.

Nếu giả sử thiếu hash (legacy/manual data), hash sẽ xuất hiện lần đầu ở:

1. **Inference time**: `infer_crawled_local.py` luôn tự tính `segment_hash` và `context_segment_hash` cho output `crawled_predictions.jsonl`.
2. **Feedback submit time**: backend `/api/feedback/segment` tự tính `segment_hash` từ `(text, html_tag_override/html_tag)`.

Không có cơ chế backfill lại hash vào chính `data/raw/.../segments.jsonl`.

---

## 3. Metadata Availability tại Crawl Time

Đánh giá tại thời điểm ghi `data/raw/crawled_urls/<url_hash>/segments.jsonl`:

- `url_hash`: **Có**
  - Có trong từng segment row và trong `meta.json`.
- `segment_index`: **Có**
  - Có trong từng row (`segment_index`).
- `html_tag_effective`: **Không persist explicit**
  - Có tham số đầu vào khi build record (default `"body"`) nhưng **không ghi thành field riêng**; chỉ ảnh hưởng gián tiếp đến `segment_hash`.
- `domain_category`: **Không**
  - Chỉ có ở inference time từ `HybridDomainClassifier`.
- `context (prev/next segment text)`: **Không**
  - Không ghi vào segments artifact lúc crawl; chỉ infer script tự reconstruct theo thứ tự line khi chạy.

---

## 4. Gaps cho Pseudo-label Pipeline

### 4.1 Fields thiếu trong `segments.jsonl`

Thiếu các field quan trọng nếu muốn pseudo-label pipeline ổn định, self-learning-friendly:

- `html_tag_effective` (explicit field, hiện chỉ implicit qua hash seed mặc định `body`).
- `domain_category` (news/social/forum/unknown) tại thời điểm gán nhãn.
- `context_segment_hash` tại crawl artifact.
- `prev_text`/`next_text` hoặc context pointer ổn định.
- `source_modality` (page_text vs video_transcript) khi dùng merged pipeline.

### 4.2 Thông tin chỉ có tại inference time nhưng không persist về crawl artifact

Có trong `crawled_predictions.jsonl`/`page_level_results.json` nhưng không quay về source segment artifact:

- `effective_threshold`, `domain_category`, `decision_source`, `formality_score`, `og_types`, `html_tags`.
- `toxic_prob_adjusted`, `ai_learned`, `ai_learned_mode`, `learned_support`, `learned_agreement`.
- `context_segment_hash` (chỉ output infer).

Điều này làm pseudo-label builder phải phụ thuộc artifact job-level thay vì raw crawl canonical.

### 4.3 Potential hash inconsistency risks

1. **HTML-tag drift risk**
   - Crawl hash seed tag=`body`, inference/feedback thường dùng tag khác (`unknown`, `article`, ...).
   - Cùng text có thể sinh nhiều hash khác nhau theo stage.

2. **Merged-segment schema loss**
   - `build_merged_segments()` ghi `segments.jsonl` chỉ còn `{"text": ...}`.
   - Mất `segment_index/url_hash/segment_hash` của raw segments.
   - Infer vẫn chạy được (text-only) nhưng identity lineage yếu.

3. **Context-hash stability risk**
   - `context_segment_hash` phụ thuộc neighbor ordering.
   - Nếu transcript append order hoặc segmentation thay đổi giữa runs, hash thay đổi dù text target giữ nguyên.

4. **Feedback dedupe key mismatch risk**
   - DB dedupe dựa `COALESCE(context_segment_hash, segment_hash)` + tag.
   - Nếu upstream không gửi context hash hoặc gửi hash từ khác tag-mode, bản ghi học có thể phân mảnh.

---

## 5. Đề xuất minimal changes (chỉ liệt kê, chưa implement)

1. **File: `setup_and_crawl.py`**
   **Loại thay đổi:** thêm field
   Đề xuất: ghi explicit `html_tag_effective` vào từng row `segments.jsonl` (giá trị hiện tại vẫn `body` nếu chưa có detector).

2. **File: `setup_and_crawl.py`**
   **Loại thay đổi:** thêm field
   Đề xuất: thêm `source_modality` (`page_text`/`video_transcript`) khi tạo merged or transcript-origin rows (nếu crawl stage tạo được).

3. **File: `backend/app.py` (`build_merged_segments`)**
   **Loại thay đổi:** thay đổi logic
   Đề xuất: không ghi merged row dạng text-only; giữ tối thiểu identity fields (`url_hash`, `segment_index`, `segment_hash` hoặc `upstream_segment_hash`, `source_modality`).

4. **File: `infer_crawled_local.py`**
   **Loại thay đổi:** thay đổi logic (minimal)
   Đề xuất: khi input row đã có `segment_hash` + `html_tag_effective`, cho phép preserve/record alongside inferred hash để trace hash lineage (crawl-hash vs infer-hash).

5. **File: `backend/app.py` (`/api/feedback/segment` payload handling)**
   **Loại thay đổi:** thêm field / thay đổi logic nhẹ
   Đề xuất: ưu tiên nhận `segment_hash` từ client/infer output nếu có, đồng thời lưu `context_segment_hash`; fallback mới tự build hash.

6. **File: `domain_classifier.py`**
   **Loại thay đổi:** không cần sửa (cho mục tiêu audit này)
   Lý do: logic domain-threshold hiện đủ cho inference; gap chính nằm ở artifact lineage/persistence, không ở classifier.

7. **File: `backend/app.py` + pseudo-label script tương lai (`07_generate_pseudo_labels.py`)**
   **Loại thay đổi:** thêm field contract
   Đề xuất: chuẩn hoá một canonical segment identity contract dùng chung (`url_hash`, `segment_index`, `segment_hash_mode`, `segment_hash`, `context_segment_hash`, `html_tag_effective`, `domain_category_at_infer`).

---

## Audit metadata

- Audit timestamp (local): **2026-04-04T00:00:00 (date-level audit, generated on 2026-04-04)**
- Audited files:
  - `setup_and_crawl.py`
  - `infer_crawled_local.py`
  - `domain_classifier.py`
  - `backend/app.py` (crawl/merge/infer/feedback flows)
- Model used for audit writing: **Claude Sonnet 4.6 (`claude-sonnet-4-6`)**
- Script versions audited: **workspace current state at audit time (no code modifications in this task)**

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_crawled_local.py
Run PhoBERT sequence-classification inference on crawled segments.
Domain-aware thresholding: seg_threshold is adjusted per URL category
(news / social / forum / unknown) to reduce domain-bias false positives.

Example:
  python infer_crawled_local.py \
    --model_name phobert/v2 \
    --model_base_dir models/options \
    --data_dir data/raw/crawled_urls \
    --out_dir data/processed \
    --batch_size 8 \
    --max_length 256 \
    --page_threshold 0.25 \
    --seg_threshold 0.5

Threshold overrides per category (optional):
  --threshold_news 0.75 --threshold_social 0.50 --threshold_forum 0.62 --threshold_unknown 0.65

Notes:
- Local only: model loaded from local checkpoint
- Tokenizer from HF ("vinai/phobert-base") unless you pass --tokenizer_name
- Device auto: CUDA > MPS (Apple Silicon) > CPU
- domain_classifier.py must be in the same directory as this script
"""

import json
import argparse
import hashlib
import os
from typing import List, Dict, Any, Tuple, Optional

import joblib
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Hybrid domain-aware threshold classifier (same directory)
from domain_classifier import HybridDomainClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_OPTIONS_DIR = Path(
    os.getenv("VIETTOXIC_MODEL_OPTIONS_DIR", str(SCRIPT_DIR / "models" / "options"))
).expanduser()

MODEL_TYPES = {
    "phobert": {
        "required": ("config.json",),
        "required_any": ("model.safetensors", "pytorch_model.bin"),
    },
    "tfidf_lr": {
        "required": ("vectorizer.pkl", "model_lr.pkl"),
        "required_any": (),
    },
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_segments_jsonl(path: Path) -> List[Dict[str, Any]]:
    segs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text_raw = obj.get("text", "")
                text = text_raw.strip() if isinstance(text_raw, str) else ""
                if not text:
                    continue
                segs.append(
                    {
                        "text": text,
                        "segment_hash": obj.get("segment_hash"),
                        "html_tag_effective": obj.get("html_tag_effective"),
                    }
                )
            except json.JSONDecodeError:
                continue
    return segs


def iter_url_folders(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return [p for p in root_dir.iterdir() if p.is_dir()]


def normalize_segment_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def build_segment_hash(text: str, html_tag: str) -> str:
    base = f"{normalize_segment_text(text)}|{(html_tag or '').strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def build_context_segment_hash(
    prev_text: str,
    text: str,
    next_text: str,
    html_tag: str,
) -> str:
    base = "|".join([
        normalize_segment_text(prev_text),
        normalize_segment_text(text),
        normalize_segment_text(next_text),
        (html_tag or "").strip().lower(),
    ])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def apply_learned_prior(
    model_score: float,
    learned_stats: Optional[Dict[str, float]],
    alpha: float = 0.2,
    min_support: int = 3,
    min_agreement: float = 0.85,
    toxic_boost: float = 0.35,
    toxic_floor: float = 0.9,
) -> Tuple[float, bool, Optional[str], str]:
    if not learned_stats:
        return model_score, False, None, "none"

    toxic_count = int(learned_stats.get("toxic_count", 0))
    clean_count = int(learned_stats.get("clean_count", 0))
    support = toxic_count + clean_count
    if support < min_support:
        return model_score, False, None, "insufficient_support"

    dominant = max(toxic_count, clean_count)
    agreement = dominant / support if support else 0.0
    if agreement < min_agreement:
        return model_score, False, None, "conflict"

    prior_toxic = toxic_count / support
    learned_label = "toxic" if prior_toxic >= 0.5 else "clean"

    if learned_label == "toxic":
        # Toxic feedback is treated as strong evidence and receives a heavy penalty.
        boosted = model_score + toxic_boost * prior_toxic
        adjusted = max(model_score, max(toxic_floor, boosted))
        adjusted = max(0.0, min(1.0, adjusted))
        return adjusted, True, learned_label, "toxic_lock"

    adjusted = max(0.0, min(1.0, model_score + alpha * (prior_toxic - 0.5)))
    return adjusted, True, learned_label, "prior_applied"


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def list_model_types(model_root: Path) -> List[str]:
    if not model_root.exists():
        return []
    if not model_root.is_dir():
        raise NotADirectoryError(f"Model root path must be a directory: {model_root}")
    try:
        return sorted(
            [
                p.name
                for p in model_root.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
        )
    except PermissionError as exc:
        raise PermissionError(f"Permission denied while reading model root: {model_root}") from exc


def list_models_by_type(model_root: Path, model_type: str) -> List[str]:
    if model_type not in MODEL_TYPES:
        return []
    base_dir = model_root / model_type
    if not base_dir.exists():
        return []
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Model base path must be a directory: {base_dir}")
    try:
        return sorted(
            [
                p.name
                for p in base_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
        )
    except PermissionError as exc:
        raise PermissionError(f"Permission denied while reading model directory: {base_dir}") from exc


def list_all_models(model_root: Path) -> List[Dict[str, str]]:
    models: List[Dict[str, str]] = []
    for model_type in list_model_types(model_root):
        for name in list_models_by_type(model_root, model_type):
            models.append({
                "id": f"{model_type}/{name}",
                "type": model_type,
                "name": name,
            })
    return models


def get_default_model_id(model_root: Path) -> Optional[str]:
    phobert_models = list_models_by_type(model_root, "phobert")
    if phobert_models:
        if "v2" in phobert_models:
            return "phobert/v2"
        return f"phobert/{phobert_models[0]}"
    all_models = list_all_models(model_root)
    if not all_models:
        return None
    return all_models[0]["id"]


def validate_model_artifacts(model_type: str, model_dir: Path) -> None:
    requirements = MODEL_TYPES.get(model_type)
    if not requirements:
        raise ValueError(f"Unsupported model type: {model_type}")

    missing = [name for name in requirements["required"] if not (model_dir / name).exists()]
    required_any = requirements["required_any"]
    if required_any and not any((model_dir / name).exists() for name in required_any):
        missing.append(" or ".join(required_any))

    if missing:
        files = sorted([p.name for p in model_dir.iterdir() if p.is_file()])
        raise FileNotFoundError(
            f"Checkpoint folder missing: {', '.join(missing)}. Files found: {files}"
        )


def parse_model_id(model_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not model_id:
        return None, None
    if any(x in model_id for x in ("..", "\\")):
        raise ValueError(f"Invalid model name: {model_id}")
    if "/" not in model_id:
        return "phobert", model_id
    model_type, name = model_id.split("/", 1)
    if not model_type or not name or any(x in name for x in ("..", "/", "\\")):
        raise ValueError(f"Invalid model name: {model_id}")
    return model_type, name


def resolve_model_path(model_root: Path, model_id: Optional[str]) -> Tuple[str, str, Path]:
    if not model_id:
        model_id = get_default_model_id(model_root)

    if model_id is None:
        raise ValueError("No default model available")

    model_type, name = parse_model_id(model_id)
    if not model_type or not name:
        raise ValueError(f"Invalid model name: {model_id}")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    if any(x in name for x in ("..", "/", "\\")):
        raise ValueError(f"Invalid model name: {model_id}")

    base_dir = model_root / model_type
    models = list_models_by_type(model_root, model_type)
    if not models:
        raise ValueError(f"No models found under {base_dir}")
    if name not in models:
        raise ValueError(f"Model '{model_id}' not found. Available: {models}")

    model_path = base_dir / name
    if not model_path.is_dir():
        raise ValueError(f"Model '{model_id}' is not a directory under {base_dir}")

    validate_model_artifacts(model_type, model_path)
    return model_type, name, model_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_probs(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> List[float]:
    """Returns toxic prob for class 1."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1].detach().float().cpu().tolist()


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer_crawled(
    model_path: Optional[str],
    data_dir: str,
    out_dir: str,
    batch_size: int = 8,
    max_length: int = 256,
    page_threshold: float = 0.25,
    seg_threshold: float = 0.5,          # fallback / default threshold
    tokenizer_name: str = "vinai/phobert-base",
    device: str = "auto",
    fp16: bool = False,
    limit_pages: int = 0,
    quiet: bool = False,
    only_url_hashes: Optional[List[str]] = None,
    debug_force_prob: Optional[float] = None,
    model_name: Optional[str] = None,
    model_base_dir: Optional[str] = None,
    model_type: Optional[str] = None,
    # Per-category threshold overrides (None = use DomainClassifier defaults)
    threshold_news:    Optional[float] = None,
    threshold_social:  Optional[float] = None,
    threshold_forum:   Optional[float] = None,
    threshold_unknown: Optional[float] = None,
    formality_range: float = 0.15,
    override_threshold: float = 0.30,
    html_dir: Optional[str] = None,
    learned_feedback: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
) -> Dict[str, Any]:

    # ── Validate model path ──────────────────────────────────────────────
    if model_path:
        resolved_model_path = Path(model_path).expanduser().resolve()
        selected_model_name = resolved_model_path.name
        inferred_type = model_type
        if model_type is None:
            parts = resolved_model_path.parts
            if "options" in parts:
                idx = parts.index("options")
                if idx + 1 < len(parts):
                    inferred_type = parts[idx + 1]
        model_type = inferred_type or "phobert"
    else:
        base_dir = Path(model_base_dir).expanduser() if model_base_dir else MODEL_OPTIONS_DIR
        model_type, selected_model_name, resolved_model_path = resolve_model_path(base_dir, model_name)

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model path not found: {resolved_model_path}")
    if not resolved_model_path.is_dir():
        raise NotADirectoryError(f"Model path must be a directory: {resolved_model_path}")

    validate_model_artifacts(model_type, resolved_model_path)

    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Domain classifier ────────────────────────────────────────────────
    threshold_overrides: Dict[str, float] = {}
    if threshold_news    is not None: threshold_overrides["news"]    = threshold_news
    if threshold_social  is not None: threshold_overrides["social"]  = threshold_social
    if threshold_forum   is not None: threshold_overrides["forum"]   = threshold_forum
    if threshold_unknown is not None: threshold_overrides["unknown"] = threshold_unknown

    # Pass the CLI seg_threshold as the default unknown fallback when no override given
    if "unknown" not in threshold_overrides:
        threshold_overrides["unknown"] = seg_threshold

    domain_clf = HybridDomainClassifier(
        threshold_news=threshold_overrides.get("news", seg_threshold),
        threshold_social=threshold_overrides.get("social", seg_threshold),
        threshold_forum=threshold_overrides.get("forum", seg_threshold),
        threshold_unknown=threshold_overrides.get("unknown", seg_threshold),
        formality_range=formality_range,
        override_threshold=override_threshold,
    )

    if not quiet:
        print("[INFO] Domain-aware thresholds:")
        for cat, thr in domain_clf.thresholds.items():
            marker = " ← CLI default (--seg_threshold)" if cat == "unknown" and threshold_unknown is None else ""
            print(f"         {cat:<10} → {thr}{marker}")
        print(f"[INFO] Formality range: {formality_range}")
        print(f"[INFO] Override threshold: {override_threshold}")
        if html_dir:
            print(f"[INFO] HTML dir: {html_dir}")
        print(f"[INFO] Selected model: {model_type}/{selected_model_name}")
        print(f"[INFO] Model path: {resolved_model_path}")

    # ── Device ───────────────────────────────────────────────────────────
    device = pick_device(device)
    if not quiet:
        print(f"[INFO] Device: {device}")

    # ── Load model assets (skip in debug force-prob mode) ────────────────
    tokenizer = None
    model = None
    vectorizer = None
    if debug_force_prob is None:
        if model_type == "phobert":
            if not quiet:
                print("[INFO] Loading tokenizer:", tokenizer_name)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            if not quiet:
                print("[INFO] Loading model from:", str(resolved_model_path))
            config = AutoConfig.from_pretrained(str(resolved_model_path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(resolved_model_path), config=config, local_files_only=True
            )
            model.to(device)
            if fp16 and device.type == "cuda":
                model.half()
                if not quiet:
                    print("[INFO] Using FP16 on CUDA")
            model.eval()
        elif model_type == "tfidf_lr":
            if not quiet:
                print("[INFO] Loading TF-IDF vectorizer & LR model from:", str(resolved_model_path))
            vectorizer = joblib.load(resolved_model_path / "vectorizer.pkl")
            model = joblib.load(resolved_model_path / "model_lr.pkl")
            if not hasattr(model, "predict_proba"):
                raise ValueError("TF-IDF+LR model must support predict_proba")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:
        if debug_force_prob < 0.0 or debug_force_prob > 1.0:
            raise ValueError("debug_force_prob must be in [0.0, 1.0]")
        if not quiet:
            print(f"[INFO] Debug mode: force toxic_prob={debug_force_prob} for all segments")

    # ── Output paths ─────────────────────────────────────────────────────
    seg_out_path  = out_dir / "crawled_predictions.jsonl"
    page_out_path = out_dir / "page_level_results.json"
    csv_out_path  = out_dir / "page_level_results.csv"

    # ── Scan folders ─────────────────────────────────────────────────────
    url_folders = iter_url_folders(data_dir)
    if only_url_hashes:
        allow = set(only_url_hashes)
        url_folders = [p for p in url_folders if p.name in allow]
    if limit_pages and limit_pages > 0:
        url_folders = url_folders[:limit_pages]

    if not url_folders:
        print(f"[WARN] No url folders found in: {data_dir}")
        return {
            "page_results": [], "segment_results": [],
            "out_dir": str(out_dir),
            "page_out_path": str(page_out_path),
            "seg_out_path":  str(seg_out_path),
            "csv_out_path":  str(csv_out_path),
        }

    predictions_all: List[Dict[str, Any]] = []
    page_results:    List[Dict[str, Any]] = []

    # Category stats for end-of-run summary
    category_counts: Dict[str, int] = {}
    learned_feedback = learned_feedback or {}

    # ── Per-page processing ──────────────────────────────────────────────
    for folder in tqdm(url_folders, desc="Processing pages"):
        meta_path = folder / "meta.json"
        segs_path = folder / "segments.jsonl"

        if not (meta_path.exists() and segs_path.exists()):
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

        url = meta.get("url", "unknown")
        url_hash = folder.name

        segment_rows = load_segments_jsonl(segs_path)
        if not segment_rows:
            continue
        segments = [row["text"] for row in segment_rows]

        html_content = None
        if html_dir:
            html_path = Path(html_dir) / f"{url_hash}.html"
            if html_path.exists():
                try:
                    html_content = html_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    html_content = None
            else:
                alt_path = Path(html_dir) / url_hash / "raw.html"
                if alt_path.exists():
                    try:
                        html_content = alt_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        html_content = None

        # ── Domain-aware threshold (hybrid) ───────────────────────────
        threshold_info = domain_clf.get_threshold(url, segments, html_content, quiet=quiet)
        domain_category = threshold_info["domain_category"]
        effective_threshold = threshold_info["effective_threshold"]
        category_counts[domain_category] = category_counts.get(domain_category, 0) + 1

        if threshold_info.get("layer3_overrides") and not quiet:
            print(
                f"[WARN] Layer3 override for {url_hash}: {threshold_info.get('decision_source')}"
            )

        page_preds: List[Dict[str, Any]] = []

        # ── Batch inference ───────────────────────────────────────────
        for i in range(0, len(segments), batch_size):
            batch_texts = segments[i: i + batch_size]
            if debug_force_prob is None:
                if model_type == "phobert":
                    probs = predict_probs(batch_texts, tokenizer, model, device=device, max_length=max_length)
                else:
                    X = vectorizer.transform(batch_texts)
                    probs = model.predict_proba(X)[:, 1].tolist()
            else:
                probs = [float(debug_force_prob)] * len(batch_texts)
            for local_idx, (text, prob) in enumerate(zip(batch_texts, probs)):
                global_idx = i + local_idx
                prev_text = segments[global_idx - 1] if global_idx > 0 else ""
                next_text = segments[global_idx + 1] if global_idx + 1 < len(segments) else ""

                model_score = float(prob)
                segment_row = segment_rows[global_idx]

                artifact_html_tag_raw = segment_row.get("html_tag_effective")
                artifact_html_tag = artifact_html_tag_raw.strip().lower() if isinstance(artifact_html_tag_raw, str) else ""
                inferred_html_tag = ((threshold_info.get("html_tags") or ["unknown"])[0] or "").strip().lower()
                html_tag_key = artifact_html_tag or inferred_html_tag

                artifact_seg_hash_raw = segment_row.get("segment_hash")
                artifact_seg_hash = artifact_seg_hash_raw.strip() if isinstance(artifact_seg_hash_raw, str) else ""
                seg_hash = artifact_seg_hash or build_segment_hash(text, html_tag_key)
                context_hash = build_context_segment_hash(prev_text, text, next_text, html_tag_key)

                learned_stats = (
                    learned_feedback.get((context_hash, html_tag_key))
                    or learned_feedback.get((seg_hash, html_tag_key))
                    or learned_feedback.get((context_hash, ""))
                    or learned_feedback.get((seg_hash, ""))
                )

                adjusted_score, ai_learned, learned_label, learned_mode = apply_learned_prior(
                    model_score=model_score,
                    learned_stats=learned_stats,
                )
                label = 1 if adjusted_score > effective_threshold else 0

                page_preds.append({
                    "text": text,
                    "toxic_prob": round(model_score, 4),
                    "toxic_prob_adjusted": round(adjusted_score, 4),
                    "toxic_label": label,
                    "ai_learned": ai_learned,
                    "ai_learned_label": learned_label,
                    "ai_learned_mode": learned_mode,
                    "segment_hash": seg_hash,
                    "context_segment_hash": context_hash,
                    "html_tag_effective": html_tag_key,
                    "learned_support": int((learned_stats or {}).get("support", 0)),
                    "learned_agreement": round(float((learned_stats or {}).get("agreement", 0.0)), 4),
                })

        # ── Aggregate page-level ──────────────────────────────────────
        total_segs  = len(page_preds)
        toxic_count = sum(1 for p in page_preds if p["toxic_label"] == 1)
        toxic_ratio = toxic_count / total_segs if total_segs else 0.0
        page_toxic  = 1 if toxic_ratio > page_threshold else 0
        avg_prob    = sum(p["toxic_prob"] for p in page_preds) / total_segs if total_segs else 0.0

        top5 = sorted(page_preds, key=lambda x: x["toxic_prob"], reverse=True)[:5]

        page_results.append({
            "url_hash":             url_hash,
            "url":                  url,
            "seg_threshold_used":   effective_threshold,
            "effective_threshold":  effective_threshold,
            "struct_confidence":    threshold_info.get("struct_confidence"),
            "struct_source":        threshold_info.get("struct_source"),
            "formality_score":      threshold_info.get("formality_score"),
            "formality_delta":      threshold_info.get("formality_delta"),
            "layer3_overrides":     threshold_info.get("layer3_overrides"),
            "decision_source":      threshold_info.get("decision_source"),
            "og_types":             threshold_info.get("og_types"),
            "html_tags":            threshold_info.get("html_tags"),
            "method":               meta.get("method",  "unknown"),
            "status":               meta.get("status",  "unknown"),
            "total_segments":       total_segs,
            "toxic_segments":       toxic_count,
            "toxic_ratio":          round(toxic_ratio, 4),
            "page_toxic":           page_toxic,
            "avg_toxic_prob":       round(float(avg_prob), 4),
            "top5_toxic_segments":  [s["text"][:150] + "..." for s in top5],
        })

        for p in page_preds:
            predictions_all.append({
                "url_hash":           url_hash,
                "url":                url,
                "seg_threshold_used": effective_threshold,
                "formality_score":    threshold_info.get("formality_score"),
                "og_types":           threshold_info.get("og_types"),
                "html_tags":          threshold_info.get("html_tags"),
                "score":              p["toxic_prob"],
                "label":              p["toxic_label"],
                **p,
            })

    # ── Save outputs ─────────────────────────────────────────────────────
    with seg_out_path.open("w", encoding="utf-8") as f:
        for pred in predictions_all:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    df_pages = pd.DataFrame(page_results)
    page_out_path.write_text(
        df_pages.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    df_pages.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    # ── Summary ──────────────────────────────────────────────────────────
    if not quiet:
        print("\n" + "=" * 65)
        print(f"PAGE-LEVEL SUMMARY (max_length={max_length})")
        print("=" * 65)

        if len(df_pages) > 0:
            cols = ["url", "seg_threshold_used",
                    "total_segments", "toxic_ratio", "page_toxic", "avg_toxic_prob"]
            cols = [c for c in cols if c in df_pages.columns]
            print(df_pages[cols].to_string(index=False))

            print("\nMetrics tổng hợp:")
            print(f"  - Số trang processed    : {len(df_pages)}")
            print(f"  - Avg toxic_ratio        : {df_pages['toxic_ratio'].mean():.4f}")
            print(f"  - % trang toxic          : {(df_pages['page_toxic'] == 1).mean() * 100:.2f}%")
            print(f"  - Avg segments/page      : {df_pages['total_segments'].mean():.1f}")
        else:
            print("[WARN] No pages produced results.")

        print("=" * 65)
        print("\nKết quả đã lưu tại:")
        print(f"  - {seg_out_path}  (segment-level)")
        print(f"  - {page_out_path}  (page-level json)")
        print(f"  - {csv_out_path}   (page-level csv)")

    return {
        "page_results":     page_results,
        "segment_results":  predictions_all,
        "out_dir":          str(out_dir),
        "page_out_path":    str(page_out_path),
        "seg_out_path":     str(seg_out_path),
        "csv_out_path":     str(csv_out_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model id under --model_base_dir (e.g. phobert/v2, tfidf_lr/baseline)",
    )
    ap.add_argument(
        "--model_base_dir",
        type=str,
        default=str(MODEL_OPTIONS_DIR),
        help="Model root that contains type subdirectories (e.g. models/options)",
    )
    ap.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Explicit model directory path. If provided, overrides --model_name",
    )
    ap.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["phobert", "tfidf_lr"],
        help="Explicit model type when using --model_path",
    )
    ap.add_argument("--tokenizer_name",  type=str, default="vinai/phobert-base")
    ap.add_argument("--data_dir",        type=str, default="data/raw/crawled_urls")
    ap.add_argument("--out_dir",         type=str, default="data/processed")
    ap.add_argument("--max_length",      type=int, default=256)
    ap.add_argument("--batch_size",      type=int, default=8)

    # Threshold args
    ap.add_argument("--seg_threshold",   type=float, default=0.5,
                    help="Default/fallback seg threshold (also used for 'unknown' domains unless --threshold_unknown set)")
    ap.add_argument("--page_threshold",  type=float, default=0.25)
    ap.add_argument("--threshold_news",    type=float, default=None, help="Override threshold for news domains")
    ap.add_argument("--threshold_social",  type=float, default=None, help="Override threshold for social domains")
    ap.add_argument("--threshold_forum",   type=float, default=None, help="Override threshold for forum domains")
    ap.add_argument("--threshold_unknown", type=float, default=None, help="Override threshold for unknown domains")
    ap.add_argument("--formality_range", type=float, default=0.15,
                    help="Max threshold adjustment from formality score")
    ap.add_argument("--override_threshold", type=float, default=0.30,
                    help="Formality delta to trigger Layer 3 override")
    ap.add_argument("--html_dir", type=str, default=None,
                    help="Optional directory containing <url_hash>.html files")

    # Device & perf
    ap.add_argument("--device",      type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--fp16",        action="store_true")
    ap.add_argument("--limit_pages", type=int, default=0)
    ap.add_argument("--quiet",       action="store_true")
    ap.add_argument("--debug_force_prob", type=float, default=None,
                    help="Debug only: force toxic_prob for every segment (0..1), bypass model inference")

    args = ap.parse_args()

    infer_crawled(
        model_path      = args.model_path,
        model_name      = args.model_name,
        model_base_dir  = args.model_base_dir,
        model_type      = args.model_type,
        data_dir        = args.data_dir,
        out_dir         = args.out_dir,
        batch_size      = args.batch_size,
        max_length      = args.max_length,
        page_threshold  = args.page_threshold,
        seg_threshold   = args.seg_threshold,
        tokenizer_name  = args.tokenizer_name,
        device          = args.device,
        fp16            = args.fp16,
        limit_pages     = args.limit_pages,
        quiet           = args.quiet,
        debug_force_prob = args.debug_force_prob,
        threshold_news    = args.threshold_news,
        threshold_social  = args.threshold_social,
        threshold_forum   = args.threshold_forum,
        threshold_unknown = args.threshold_unknown,
        formality_range   = args.formality_range,
        override_threshold = args.override_threshold,
        html_dir          = args.html_dir,
    )


if __name__ == "__main__":
    main()

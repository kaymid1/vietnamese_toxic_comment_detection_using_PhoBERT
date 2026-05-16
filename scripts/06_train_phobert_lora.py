"""
PhoBERT Full Fine-tuning — Vietnamese Toxic Comment Detection
Refactored to match the LoRA script data source + export layout.
"""

# -----------------------
# Install
# -----------------------
# !pip -q install -U transformers datasets scikit-learn torch accelerate

import argparse
import os, json, random, shutil, time, uuid
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, log_loss,
    matthews_corrcoef, balanced_accuracy_score,
)

# -----------------------
# Logging
# -----------------------
t0 = time.time()

MLFLOW_ENABLED = os.environ.get("MLFLOW_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns/")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "viettoxic-phobert-lora")
POLICY_VERSION = os.environ.get("POLICY_VERSION", "policy-v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "phobert/v2")
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = uuid.uuid4().hex[:8]
RUN_ID = ""

mlflow = None
mlflow_active = False

def log(msg):
    print(f"[{time.time()-t0:8.1f}s] {msg}", flush=True)

def show_gpu():
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        log(f"VRAM free/total: {free/1e9:.2f}GB / {total/1e9:.2f}GB")
    else:
        log("No GPU detected (CPU only).")

os.environ["WANDB_DISABLED"] = "true"

# ================================================================
# Config — tune here
# ================================================================
parser = argparse.ArgumentParser(description="Train PhoBERT LoRA/full fine-tune with optional pseudo-label mixing")
parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print resolved config without training")
args, _ = parser.parse_known_args()

DATA_DIR = os.environ.get("DATA_DIR", "/content/drive/MyDrive/victsd_gold")
DATASET_PREFIX = os.environ.get("DATASET_PREFIX", "").strip()
DATASET_LAYOUT = os.environ.get("DATASET_LAYOUT", "auto").strip().lower()
MODEL_NAME = os.environ.get("MODEL_NAME", "vinai/phobert-base-v2").strip()
MAX_LENGTH = 256

PSEUDO_LABELS_DIR = os.environ.get("PSEUDO_LABELS_DIR", "").strip()
PSEUDO_LOSS_WEIGHT = float(os.environ.get("PSEUDO_LOSS_WEIGHT", "0.3"))
GOLD_DATA_DIR = os.environ.get("GOLD_DATA_DIR", "data/processed/victsd_gold")
GOLD_DATASET_PREFIX = os.environ.get("GOLD_DATASET_PREFIX", "").strip()
MAX_PSEUDO_RATIO = float(os.environ.get("MAX_PSEUDO_RATIO", "0.4"))
RUN_MANIFEST_DIR = os.environ.get("RUN_MANIFEST_DIR", "experiments/retrain_runs/")
MIXED_MODE = bool(PSEUDO_LABELS_DIR)

BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS     = 10
LR         = 2e-5
WEIGHT_DECAY  = 0.05
WARMUP_RATIO  = 0.08
MAX_GRAD_NORM = 1.0

LABEL_SMOOTHING     = 0.0
EARLY_STOP_PATIENCE = 4
HEAD_DROPOUT        = 0.1

USE_FOCAL   = True
FOCAL_GAMMA = 2.0

TOXIC_WEIGHT_SCALE = 0.5

SEED = int(os.environ.get("SEED", "42"))

run_dataset_tag = DATASET_PREFIX if DATASET_PREFIX else "victsd_gold"
RUN_ID = f"{run_dataset_tag}_{RUN_TIMESTAMP}_{RUN_SUFFIX}"
OUTPUT_BASE_ENV = os.environ.get("OUTPUT_BASE", "").strip()
OUTPUT_BASE = OUTPUT_BASE_ENV if OUTPUT_BASE_ENV else f"models/options/phobert/{RUN_ID}"
RESULTS_BASE = os.environ.get("RESULTS_BASE", f"results/phobert/{run_dataset_tag}")

if MLFLOW_ENABLED and not args.dry_run:
    try:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.start_run(run_name=RUN_ID)
        mlflow_active = True
    except Exception as exc:
        print(f"MLflow logging disabled: {exc}", flush=True)

PRIMARY_METRIC = "eval_f1_toxic"

THRESH_MIN  = 0.05
THRESH_MAX  = 0.95
THRESH_STEP = 0.01
THRESH_OBJECTIVES           = ["f1_toxic", "macro_f1"]
PRIMARY_THRESHOLD_OBJECTIVE = "f1_toxic"

N_ERROR_SAMPLES = 30

# Calibration
EPS              = 1e-12
N_BINS_ECE       = 10
USE_TEMP_SCALING = True
TEMP_LR          = 0.01
TEMP_MAX_ITERS   = 300

# ================================================================
# Seed + GPU
# ================================================================
log("Start. Setting seeds...")
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
show_gpu()

# ================================================================
# Dataset
# ================================================================
if PSEUDO_LOSS_WEIGHT <= 0:
    raise ValueError(f"PSEUDO_LOSS_WEIGHT must be > 0, got {PSEUDO_LOSS_WEIGHT}")
if not (0.0 <= MAX_PSEUDO_RATIO < 1.0):
    raise ValueError(f"MAX_PSEUDO_RATIO must be in [0, 1), got {MAX_PSEUDO_RATIO}")

log("Checking dataset files...")

pseudo_manifest = None
pseudo_manifest_path = None
if MIXED_MODE:
    gold_files = {
        split: (f"{GOLD_DATA_DIR}/{GOLD_DATASET_PREFIX}_{split}.jsonl" if GOLD_DATASET_PREFIX else f"{GOLD_DATA_DIR}/{split}.jsonl")
        for split in ["train", "validation", "test"]
    }
    for split, path in gold_files.items():
        assert os.path.exists(path), f"Missing gold {split} file: {path}"

    pseudo_accepted_path = f"{PSEUDO_LABELS_DIR}/accepted.jsonl"
    pseudo_manifest_path = f"{PSEUDO_LABELS_DIR}/manifest.json"
    assert os.path.exists(pseudo_accepted_path), f"Missing pseudo labels file: {pseudo_accepted_path}"
    assert os.path.exists(pseudo_manifest_path), f"Missing pseudo manifest file: {pseudo_manifest_path}"

    with open(pseudo_manifest_path, "r", encoding="utf-8") as f:
        pseudo_manifest = json.load(f)

    assert pseudo_manifest.get("batch_id"), "Pseudo manifest missing batch_id"
    assert pseudo_manifest.get("seed_model"), "Pseudo manifest missing seed_model"

    log("Loading gold + pseudo datasets ...")
    gold_dataset = load_dataset("json", data_files=gold_files)
    pseudo_dataset = load_dataset("json", data_files={"train": pseudo_accepted_path})["train"]
    if len(pseudo_dataset) == 0:
        raise ValueError("Pseudo labels accepted.jsonl is empty")

    pseudo_cols = pseudo_dataset.column_names
    for col in ["text", "toxicity"]:
        if col not in pseudo_cols:
            raise ValueError(f"Pseudo labels missing required column: {col}")

    keep_cols = ["text", "toxicity"]
    gold_train_base = gold_dataset["train"].remove_columns([c for c in gold_dataset["train"].column_names if c not in keep_cols])
    val_dataset = gold_dataset["validation"].remove_columns([c for c in gold_dataset["validation"].column_names if c not in keep_cols])
    test_dataset = gold_dataset["test"].remove_columns([c for c in gold_dataset["test"].column_names if c not in keep_cols])
    pseudo_base = pseudo_dataset.remove_columns([c for c in pseudo_dataset.column_names if c not in keep_cols])

    gold_train = gold_train_base.add_column("is_pseudo", [0] * len(gold_train_base))
    gold_train = gold_train.add_column("sample_weight", [1.0] * len(gold_train))

    pseudo_train = pseudo_base.add_column("is_pseudo", [1] * len(pseudo_base))
    pseudo_train = pseudo_train.add_column("sample_weight", [float(PSEUDO_LOSS_WEIGHT)] * len(pseudo_train))

    train_dataset = concatenate_datasets([gold_train, pseudo_train])

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })

    gold_data_dir_used = GOLD_DATA_DIR
    gold_prefix_used = GOLD_DATASET_PREFIX
    n_train_gold = len(gold_train)
    n_train_pseudo = len(pseudo_train)
else:
    split_names = ["train", "validation", "test"]

    def _resolve_split_paths(data_dir, dataset_prefix, layout):
        resolved = {}
        attempted = {}

        for split in split_names:
            if layout == "plain":
                candidates = [
                    os.path.join(data_dir, f"{split}.jsonl"),
                    os.path.join(data_dir, f"{dataset_prefix}_{split}.jsonl") if dataset_prefix else None,
                ]
            elif layout == "augmented":
                if not dataset_prefix:
                    raise ValueError("DATASET_PREFIX is required when DATASET_LAYOUT=augmented")
                candidates = [os.path.join(data_dir, f"{dataset_prefix}_{split}_augmented.jsonl")]
            else:  # auto
                candidates = [
                    os.path.join(data_dir, f"{split}.jsonl"),
                    os.path.join(data_dir, f"{dataset_prefix}_{split}.jsonl") if dataset_prefix else None,
                    os.path.join(data_dir, f"{dataset_prefix}_{split}_augmented.jsonl") if dataset_prefix else None,
                ]

            candidates = [c for c in candidates if c]
            attempted[split] = candidates
            found = next((p for p in candidates if os.path.exists(p)), None)
            if not found:
                attempted_msg = " | ".join(candidates)
                raise FileNotFoundError(f"Missing {split} file. Tried: {attempted_msg}")
            resolved[split] = found

        return resolved, attempted

    dataset_files, _ = _resolve_split_paths(DATA_DIR, DATASET_PREFIX, DATASET_LAYOUT)

    log("Loading dataset ...")
    dataset = load_dataset("json", data_files=dataset_files)

    keep_cols = ["text", "toxicity"]
    for split in split_names:
        dataset[split] = dataset[split].remove_columns([c for c in dataset[split].column_names if c not in keep_cols])

    dataset["train"] = dataset["train"].add_column("is_pseudo", [0] * len(dataset["train"]))
    dataset["train"] = dataset["train"].add_column("sample_weight", [1.0] * len(dataset["train"]))

    gold_data_dir_used = DATA_DIR
    gold_prefix_used = DATASET_PREFIX
    n_train_gold = len(dataset["train"])
    n_train_pseudo = 0

for split in ["validation", "test"]:
    if "is_pseudo" not in dataset[split].column_names:
        dataset[split] = dataset[split].add_column("is_pseudo", [0] * len(dataset[split]))
    if "sample_weight" not in dataset[split].column_names:
        dataset[split] = dataset[split].add_column("sample_weight", [1.0] * len(dataset[split]))

raw_text = {
    split: dataset[split]["text"] if "text" in dataset[split].column_names else None
    for split in ["validation", "test"]
}

log(f"Sizes: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")
log(f"Validation set: gold-only (n={len(dataset['validation'])} samples)")

if args.dry_run:
    log("Dry run mode enabled. Validating config and inputs only.")
    log(
        "Dry run config: "
        f"mode={'mixed' if MIXED_MODE else 'gold_only'} | "
        f"gold_data_dir={gold_data_dir_used} | gold_dataset_prefix={gold_prefix_used or '<none>'} | "
        f"pseudo_dir={PSEUDO_LABELS_DIR or '<unset>'} | pseudo_loss_weight={PSEUDO_LOSS_WEIGHT} | "
        f"max_pseudo_ratio={MAX_PSEUDO_RATIO} | run_manifest_dir={RUN_MANIFEST_DIR}"
    )
    if MIXED_MODE:
        log(
            "Dry run pseudo lineage: "
            f"batch_id={pseudo_manifest.get('batch_id')} | seed_model={pseudo_manifest.get('seed_model')} | "
            f"n_accepted_toxic={pseudo_manifest.get('n_accepted_toxic')} | "
            f"n_accepted_clean={pseudo_manifest.get('n_accepted_clean')}"
        )
        observed_ratio = n_train_pseudo / max(1, len(dataset["train"]))
        log(f"Dry run pseudo ratio (train): {observed_ratio:.4f} ({n_train_pseudo}/{len(dataset['train'])})")
    raise SystemExit(0)

os.makedirs(f"{OUTPUT_BASE}/checkpoints", exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(RUN_MANIFEST_DIR, exist_ok=True)

# ================================================================
# Tokenizer
# ================================================================
log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_batch(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

log("Tokenizing...")
tokenized_dataset = dataset.map(tokenize_batch, batched=True)
if "toxicity" in tokenized_dataset["train"].column_names:
    tokenized_dataset = tokenized_dataset.rename_column("toxicity", "labels")

allowed_feature_cols = {"input_ids", "attention_mask", "token_type_ids", "labels", "is_pseudo", "sample_weight"}
remove_cols = [c for c in tokenized_dataset["train"].column_names if c not in allowed_feature_cols]
if remove_cols:
    tokenized_dataset = tokenized_dataset.remove_columns(remove_cols)

tokenized_dataset.set_format("torch")
base_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

def data_collator(features):
    sample_weight = torch.tensor([float(f.pop("sample_weight", 1.0)) for f in features], dtype=torch.float)
    is_pseudo = torch.tensor([int(f.pop("is_pseudo", 0)) for f in features], dtype=torch.long)
    batch = base_data_collator(features)
    batch["sample_weight"] = sample_weight
    batch["is_pseudo"] = is_pseudo
    return batch

# ================================================================
# Base model + dropout tweaks
# ================================================================
log("Loading base model ...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

for attr, default in [
    ("classifier_dropout",         HEAD_DROPOUT),
    ("hidden_dropout_prob",        HEAD_DROPOUT),
    ("attention_probs_dropout_prob", HEAD_DROPOUT),
]:
    if hasattr(model.config, attr):
        current = getattr(model.config, attr)
        setattr(model.config, attr, max(float(current or 0), float(default)))

# ================================================================
# Class weights
# ================================================================
train_labels = torch.tensor(tokenized_dataset["train"]["labels"])
counts = torch.bincount(train_labels, minlength=2)
num_clean, num_toxic = counts[0].item(), counts[1].item()
log(f"Train counts -> clean={num_clean}, toxic={num_toxic}")
raw_toxic_weight = num_clean / max(1, num_toxic)
class_weights = torch.tensor([1.0, raw_toxic_weight * TOXIC_WEIGHT_SCALE], dtype=torch.float)
log(f"Class weights (scaled): {class_weights.tolist()}  raw={raw_toxic_weight:.4f} scale={TOXIC_WEIGHT_SCALE}")

# ================================================================
# Custom trainer with weighted / focal loss
# ================================================================
def focal_loss(logits, labels, weight=None, gamma=2.0, reduction="mean"):
    logp  = F.log_softmax(logits, dim=-1)
    p     = torch.exp(logp)
    pt    = p.gather(1, labels.unsqueeze(1)).squeeze(1)
    logpt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
    at    = weight.to(logits.device).gather(0, labels) if weight is not None else 1.0
    loss  = -at * (1 - pt) ** gamma * logpt
    if reduction == "none":
        return loss
    return loss.mean()

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing=0.0,
                 use_focal=False, focal_gamma=2.0, max_pseudo_ratio=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights   = class_weights
        self.label_smoothing = float(label_smoothing or 0.0)
        self.use_focal       = bool(use_focal)
        self.focal_gamma     = float(focal_gamma)
        self.max_pseudo_ratio = float(max_pseudo_ratio)

        self._samples_seen = 0
        self._pseudo_seen = 0
        self._samples_used = 0
        self._pseudo_used = 0
        self._pseudo_dropped = 0

    def pseudo_ratio_stats(self):
        seen_ratio = self._pseudo_seen / max(1, self._samples_seen)
        used_ratio = self._pseudo_used / max(1, self._samples_used)
        return {
            "samples_seen": int(self._samples_seen),
            "pseudo_seen": int(self._pseudo_seen),
            "samples_used": int(self._samples_used),
            "pseudo_used": int(self._pseudo_used),
            "pseudo_dropped": int(self._pseudo_dropped),
            "pseudo_ratio_seen": float(seen_ratio),
            "pseudo_ratio_used": float(used_ratio),
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        sample_weight = inputs.get("sample_weight", None)
        is_pseudo = inputs.get("is_pseudo", None)

        model_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "sample_weight", "is_pseudo"}}
        outputs = model(**model_inputs)
        logits = outputs["logits"]
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None

        keep_mask = torch.ones_like(labels, dtype=torch.bool)
        pseudo_count = 0
        dropped_count = 0

        if is_pseudo is not None:
            pseudo_mask = is_pseudo.to(labels.device).long() > 0
            pseudo_count = int(pseudo_mask.sum().item())
            gold_count = int((~pseudo_mask).sum().item())

            if model.training and self.max_pseudo_ratio < 1.0 and pseudo_count > 0:
                if gold_count <= 0:
                    max_pseudo_keep = 1
                else:
                    max_pseudo_keep = int((self.max_pseudo_ratio * gold_count) / max(1e-8, (1.0 - self.max_pseudo_ratio)))
                max_pseudo_keep = max(0, max_pseudo_keep)
                if pseudo_count > max_pseudo_keep:
                    pseudo_indices = torch.where(pseudo_mask)[0]
                    drop_n = pseudo_count - max_pseudo_keep
                    perm = torch.randperm(pseudo_indices.numel(), device=pseudo_indices.device)
                    drop_idx = pseudo_indices[perm[:drop_n]]
                    keep_mask[drop_idx] = False
                    dropped_count = int(drop_n)

        if not bool(keep_mask.any().item()):
            keep_mask[0] = True

        logits_kept = logits[keep_mask]
        labels_kept = labels[keep_mask]

        if self.use_focal:
            per_loss = focal_loss(logits_kept, labels_kept, weight=w, gamma=self.focal_gamma, reduction="none")
        elif self.label_smoothing > 0.0:
            lp = F.log_softmax(logits_kept, dim=-1)
            nll = F.nll_loss(lp, labels_kept, weight=w, reduction="none")
            smooth = -lp.mean(dim=-1)
            per_loss = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            per_loss = torch.nn.CrossEntropyLoss(weight=w, reduction="none")(logits_kept, labels_kept)

        if sample_weight is not None:
            sw = sample_weight.to(per_loss.device).float()
            if sw.dim() > 1:
                sw = sw.view(-1)
            sw = sw[keep_mask]
            loss = (per_loss * sw).sum() / sw.sum().clamp_min(1e-12)
        else:
            loss = per_loss.mean()

        used_count = int(keep_mask.sum().item())
        used_pseudo = pseudo_count - dropped_count
        if model.training:
            self._samples_seen += int(labels.shape[0])
            self._pseudo_seen += int(pseudo_count)
            self._samples_used += int(used_count)
            self._pseudo_used += int(max(0, used_pseudo))
            self._pseudo_dropped += int(dropped_count)

        return (loss, outputs) if return_outputs else loss

# ================================================================
# Metrics helpers
# ================================================================
def softmax_probs(logits):
    x = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + EPS)

def softmax_probs_temp(logits, temperature):
    return softmax_probs(logits / max(float(temperature), 1e-6))

def calibration_bins(y_true, prob_pos, n_bins=10):
    y_true = y_true.astype(int)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    out    = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i+1])
        mask   = (prob_pos >= lo) & (prob_pos < hi) if i < n_bins-1 else (prob_pos >= lo) & (prob_pos <= hi)
        count  = int(mask.sum())
        if count == 0:
            out.append({"bin": i, "lo": lo, "hi": hi, "count": 0,
                        "avg_conf": None, "accuracy": None, "gap": None})
            continue
        conf = float(prob_pos[mask].mean())
        acc  = float(((prob_pos[mask] >= 0.5).astype(int) == y_true[mask]).mean())
        out.append({"bin": i, "lo": lo, "hi": hi, "count": count,
                    "avg_conf": conf, "accuracy": acc, "gap": abs(acc - conf)})
    return out

def ece_score(y_true, prob_pos, n_bins=10):
    total = len(y_true)
    return float(sum(
        (r["count"]/total) * r["gap"]
        for r in calibration_bins(y_true, prob_pos, n_bins)
        if r["count"] > 0
    ))

def per_class_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)
    return {
        "precision_clean": float(p[0]), "recall_clean": float(r[0]),
        "f1_clean":        float(f[0]), "support_clean": int(s[0]),
        "precision_toxic": float(p[1]), "recall_toxic":  float(r[1]),
        "f1_toxic":        float(f[1]), "support_toxic":  int(s[1]),
    }

def confusion_stats(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "specificity": tn/max(1,tn+fp), "fpr": fp/max(1,fp+tn),
        "fnr": fn/max(1,fn+tp),         "npv": tn/max(1,tn+fn),
        "confusion_matrix": cm.tolist(),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    pm    = per_class_metrics(labels, preds)
    return {
        "accuracy":        float((preds == labels).mean()),
        "balanced_acc":    float(balanced_accuracy_score(labels, preds)),
        "macro_f1":        float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mcc":             float(matthews_corrcoef(labels, preds)),
        "f1_toxic":        pm["f1_toxic"], "f1_clean":        pm["f1_clean"],
        "precision_toxic": pm["precision_toxic"], "recall_toxic": pm["recall_toxic"],
    }

def evaluate_from_probs(prob_pos, labels, threshold=0.5):
    pred = (prob_pos >= threshold).astype(int)
    pm   = per_class_metrics(labels, pred)
    cs   = confusion_stats(labels, pred)
    roc_auc = pr_auc = ll = None
    try: roc_auc = float(roc_auc_score(labels, prob_pos))
    except Exception: pass
    try: pr_auc = float(average_precision_score(labels, prob_pos))
    except Exception: pass
    try: ll = float(log_loss(labels, np.stack([1-prob_pos, prob_pos], axis=1), labels=[0,1]))
    except Exception: pass
    brier  = float(np.mean((prob_pos - labels.astype(float))**2))
    ece    = ece_score(labels, prob_pos, n_bins=N_BINS_ECE)
    report = classification_report(labels, pred, labels=[0,1],
                                   target_names=["clean","toxic"], zero_division=0, digits=4)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float((pred==labels).mean()),
        "balanced_acc": float(balanced_accuracy_score(labels,pred)),
        "mcc": float(matthews_corrcoef(labels,pred)),
        "macro_f1":    float(f1_score(labels,pred,average="macro",    zero_division=0)),
        "micro_f1":    float(f1_score(labels,pred,average="micro",    zero_division=0)),
        "weighted_f1": float(f1_score(labels,pred,average="weighted", zero_division=0)),
        **pm, **cs,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "log_loss": ll,
        "brier": brier, "ece": ece,
        "classification_report": report,
        "calibration_bins": calibration_bins(labels, prob_pos, n_bins=N_BINS_ECE),
    }
    return metrics, pred

def evaluate_from_logits(logits, labels, threshold=0.5, temperature=None):
    probs    = softmax_probs_temp(logits, temperature) if temperature else softmax_probs(logits)
    prob_pos = probs[:, 1]
    metrics, preds = evaluate_from_probs(prob_pos, labels, threshold=threshold)
    if temperature is not None:
        metrics["temperature"] = float(temperature)
    return metrics, prob_pos, preds

def threshold_sweep(logits, labels, threshold_min=0.05, threshold_max=0.95,
                    threshold_step=0.01, temperature=None):
    rows = []
    for thr in np.arange(threshold_min, threshold_max+1e-9, threshold_step):
        m, _, _ = evaluate_from_logits(logits, labels, threshold=float(thr), temperature=temperature)
        rows.append({k: float(m[k]) for k in [
            "macro_f1","f1_toxic","precision_toxic","recall_toxic",
            "fpr","fnr","specificity","ece","brier",
        ]} | {"threshold": float(thr)})
    return rows, {obj: max(rows, key=lambda x: x[obj]).copy() for obj in THRESH_OBJECTIVES}

# ================================================================
# Temperature scaling
# ================================================================
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.zeros(1))
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-6)

def fit_temperature_scaler(logits_np, labels_np, lr=0.01, max_iters=300):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    logits  = torch.tensor(logits_np, dtype=torch.float32, device=device)
    labels  = torch.tensor(labels_np, dtype=torch.long,    device=device)
    scaler  = TemperatureScaler().to(device)
    opt     = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iters)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        temp = float(scaler.temperature.cpu().item())
        nll  = float(F.cross_entropy(scaler(logits), labels).cpu().item())
    return temp, nll

# ================================================================
# Training
# ================================================================
steps_per_epoch = max(1, len(tokenized_dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM))
EVAL_STEPS      = max(200, steps_per_epoch // 2)

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_BASE}/checkpoints",
    eval_strategy="steps",      eval_steps=EVAL_STEPS,
    save_strategy="steps",      save_steps=EVAL_STEPS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,  warmup_ratio=WARMUP_RATIO,
    max_grad_norm=MAX_GRAD_NORM,
    load_best_model_at_end=True,
    metric_for_best_model=PRIMARY_METRIC, greater_is_better=True,
    logging_steps=50, logging_first_step=True, save_total_limit=2,
    fp16=torch.cuda.is_available(), seed=SEED, report_to=[],
    remove_unused_columns=False,
)

trainer = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    class_weights=class_weights,
    label_smoothing=LABEL_SMOOTHING,
    use_focal=USE_FOCAL, focal_gamma=FOCAL_GAMMA,
    max_pseudo_ratio=MAX_PSEUDO_RATIO,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
)

log("Training full fine-tune...")
trainer.train()
log(f"Training finished. Best checkpoint: {trainer.state.best_model_checkpoint}")
pseudo_ratio_stats = trainer.pseudo_ratio_stats()
log(
    "Pseudo ratio stats: "
    f"seen={pseudo_ratio_stats['pseudo_ratio_seen']:.4f} "
    f"used={pseudo_ratio_stats['pseudo_ratio_used']:.4f} "
    f"dropped={pseudo_ratio_stats['pseudo_dropped']}"
)

# ================================================================
# Save best model
# ================================================================
best_model_path = OUTPUT_BASE
model.save_pretrained(best_model_path)
tokenizer.save_pretrained(best_model_path)
log(f"Saved model → {best_model_path}")

# ================================================================
# Inference
# ================================================================
log("Running prediction on validation / test...")
val_out     = trainer.predict(tokenized_dataset["validation"])
val_logits, val_labels = val_out.predictions, val_out.label_ids

test_out    = trainer.predict(tokenized_dataset["test"])
test_logits, test_labels = test_out.predictions, test_out.label_ids

# ================================================================
# Baseline eval
# ================================================================
test_pred_argmax = np.argmax(test_logits, axis=1)
pm_argmax        = per_class_metrics(test_labels, test_pred_argmax)
test_argmax_basic = {
    "accuracy":     float((test_pred_argmax==test_labels).mean()),
    "balanced_acc": float(balanced_accuracy_score(test_labels, test_pred_argmax)),
    "macro_f1":     float(f1_score(test_labels, test_pred_argmax, average="macro", zero_division=0)),
    "mcc":          float(matthews_corrcoef(test_labels, test_pred_argmax)),
    **pm_argmax,
    "confusion_matrix": confusion_matrix(test_labels, test_pred_argmax, labels=[0,1]).tolist(),
}
log(f"Test metrics (argmax): {test_argmax_basic}")

test_thr05_rich, _, _ = evaluate_from_logits(test_logits, test_labels, threshold=0.5)
log(
    f"Test @thr=0.50: macro_f1={test_thr05_rich['macro_f1']:.4f}, "
    f"f1_toxic={test_thr05_rich['f1_toxic']:.4f}, "
    f"roc_auc={test_thr05_rich['roc_auc']}, pr_auc={test_thr05_rich['pr_auc']}, "
    f"ece={test_thr05_rich['ece']:.4f}"
)

# ================================================================
# Threshold sweep (validation, raw)
# ================================================================
import csv, matplotlib.pyplot as plt

log("Sweeping thresholds on validation (raw)...")
threshold_rows = []
for thr in np.arange(THRESH_MIN, THRESH_MAX+1e-9, THRESH_STEP):
    thr = float(thr)
    m, _, _ = evaluate_from_logits(val_logits, val_labels, threshold=thr)
    row = {k: float(m[k]) for k in [
        "macro_f1","f1_toxic","precision_toxic","recall_toxic",
        "fpr","fnr","specificity","ece","brier",
    ]} | {"threshold": thr}
    threshold_rows.append(row)
    log(f"[VAL][RAW] thr={thr:.2f} f1_toxic={row['f1_toxic']:.4f} macro_f1={row['macro_f1']:.4f} "
        f"prec={row['precision_toxic']:.4f} rec={row['recall_toxic']:.4f} "
        f"fpr={row['fpr']:.4f} fnr={row['fnr']:.4f}")

val_best_by_objective = {obj: max(threshold_rows, key=lambda x: x[obj]).copy() for obj in THRESH_OBJECTIVES}
best_raw = val_best_by_objective[PRIMARY_THRESHOLD_OBJECTIVE]
log(f"Best RAW threshold ({PRIMARY_THRESHOLD_OBJECTIVE}): thr={best_raw['threshold']:.2f} "
    f"f1_toxic={best_raw['f1_toxic']:.4f} macro_f1={best_raw['macro_f1']:.4f}")

test_tuned_raw_rich, test_probs_pos_raw_tuned, test_pred_tuned_raw = evaluate_from_logits(
    test_logits, test_labels, threshold=float(best_raw["threshold"])
)
log(f"Test @raw thr={best_raw['threshold']:.2f}: macro_f1={test_tuned_raw_rich['macro_f1']:.4f} "
    f"f1_toxic={test_tuned_raw_rich['f1_toxic']:.4f} ece={test_tuned_raw_rich['ece']:.4f}")

with open(f"{RESULTS_BASE}/threshold_sweep_validation_raw.json","w",encoding="utf-8") as f:
    json.dump({"objective": PRIMARY_THRESHOLD_OBJECTIVE,
               "search": {"min": THRESH_MIN,"max": THRESH_MAX,"step": THRESH_STEP},
               "rows": threshold_rows, "best": best_raw,
               "best_by_objective": val_best_by_objective}, f, ensure_ascii=False, indent=2)

with open(f"{RESULTS_BASE}/threshold_sweep_validation_raw.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["threshold","macro_f1","f1_toxic","precision_toxic",
                                       "recall_toxic","fpr","fnr","specificity","ece","brier"])
    w.writeheader(); w.writerows(threshold_rows)

# Plots
thr_v = [r["threshold"]       for r in threshold_rows]
f1_v  = [r["f1_toxic"]        for r in threshold_rows]
pr_v  = [r["precision_toxic"] for r in threshold_rows]
rc_v  = [r["recall_toxic"]    for r in threshold_rows]
fp_v  = [r["fpr"]             for r in threshold_rows]
fn_v  = [r["fnr"]             for r in threshold_rows]

for fname, ys, lbls, title in [
    ("threshold_vs_f1_toxic.png",         [f1_v],           ["F1_toxic"],                      "Threshold vs F1_toxic"),
    ("threshold_vs_precision_recall.png", [pr_v, rc_v],     ["Precision_toxic","Recall_toxic"], "Threshold vs Precision/Recall"),
    ("threshold_vs_fpr_fnr.png",          [fp_v, fn_v],     ["FPR","FNR"],                     "Threshold vs FPR/FNR"),
]:
    plt.figure(figsize=(8,5))
    for y, lbl in zip(ys, lbls):
        kw = {"marker": "o", "markersize": 3} if len(ys) == 1 else {}
        plt.plot(thr_v, y, label=lbl, **kw)
    if len(ys) > 1:
        plt.legend()
    plt.axvline(best_raw["threshold"], linestyle="--", color="gray")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{RESULTS_BASE}/{fname}", dpi=200); plt.show()

# ================================================================
# Temperature scaling
# ================================================================
temperature_result = val_scaled_rows = val_scaled_best_by_objective = None
val_scaled_best = test_tuned_scaled_rich = test_probs_pos_scaled_tuned = test_pred_tuned_scaled = None

if USE_TEMP_SCALING:
    log("Fitting temperature scaler...")
    best_temperature, val_temp_nll = fit_temperature_scaler(
        val_logits, val_labels, lr=TEMP_LR, max_iters=TEMP_MAX_ITERS
    )
    log(f"Temperature: T={best_temperature:.6f}, val_nll={val_temp_nll:.6f}")

    raw_p    = softmax_probs(val_logits)[:,1]
    scaled_p = softmax_probs_temp(val_logits, best_temperature)[:,1]
    temperature_result = {
        "temperature":      float(best_temperature),
        "val_nll_after":    float(val_temp_nll),
        "val_ece_before":   ece_score(val_labels, raw_p,    n_bins=N_BINS_ECE),
        "val_ece_after":    ece_score(val_labels, scaled_p, n_bins=N_BINS_ECE),
        "val_brier_before": float(np.mean((raw_p    - val_labels.astype(float))**2)),
        "val_brier_after":  float(np.mean((scaled_p - val_labels.astype(float))**2)),
    }
    log(f"ECE: {temperature_result['val_ece_before']:.4f} → {temperature_result['val_ece_after']:.4f}  "
        f"Brier: {temperature_result['val_brier_before']:.4f} → {temperature_result['val_brier_after']:.4f}")

    val_scaled_rows, val_scaled_best_by_objective = threshold_sweep(
        val_logits, val_labels,
        threshold_min=THRESH_MIN, threshold_max=THRESH_MAX, threshold_step=THRESH_STEP,
        temperature=best_temperature,
    )
    val_scaled_best = val_scaled_best_by_objective[PRIMARY_THRESHOLD_OBJECTIVE]

    test_tuned_scaled_rich, test_probs_pos_scaled_tuned, test_pred_tuned_scaled = evaluate_from_logits(
        test_logits, test_labels,
        threshold=float(val_scaled_best["threshold"]),
        temperature=float(best_temperature),
    )
    log(f"Test @scaled thr={val_scaled_best['threshold']:.2f} T={best_temperature:.4f}: "
        f"macro_f1={test_tuned_scaled_rich['macro_f1']:.4f} "
        f"f1_toxic={test_tuned_scaled_rich['f1_toxic']:.4f} "
        f"ece={test_tuned_scaled_rich['ece']:.4f}")

    with open(f"{RESULTS_BASE}/threshold_sweep_validation_scaled.json","w",encoding="utf-8") as f:
        json.dump({"temperature": float(best_temperature), "rows": val_scaled_rows,
                   "best_by_objective": val_scaled_best_by_objective}, f, ensure_ascii=False, indent=2)

# ================================================================
# Deployment selection
# ================================================================
use_scaled_for_deploy = False
deploy_temperature    = None
deploy_threshold      = float(best_raw["threshold"])
deploy_mode           = "raw_threshold"

if temperature_result is not None and val_scaled_best is not None:
    if temperature_result["val_ece_after"] <= temperature_result["val_ece_before"]:
        use_scaled_for_deploy = True
        deploy_temperature    = float(temperature_result["temperature"])
        deploy_threshold      = float(val_scaled_best["threshold"])
        deploy_mode           = "temperature_scaled_threshold"

log(f"Deployment: mode={deploy_mode}, thr={deploy_threshold:.4f}, T={deploy_temperature}")

final_test_rich  = test_tuned_scaled_rich  if use_scaled_for_deploy else test_tuned_raw_rich
final_test_probs = test_probs_pos_scaled_tuned if use_scaled_for_deploy else test_probs_pos_raw_tuned
final_test_preds = test_pred_tuned_scaled   if use_scaled_for_deploy else test_pred_tuned_raw

log(f"FINAL: macro_f1={final_test_rich['macro_f1']:.4f} f1_toxic={final_test_rich['f1_toxic']:.4f} "
    f"precision={final_test_rich['precision_toxic']:.4f} recall={final_test_rich['recall_toxic']:.4f} "
    f"fpr={final_test_rich['fpr']:.4f} fnr={final_test_rich['fnr']:.4f} "
    f"ece={final_test_rich['ece']:.4f} brier={final_test_rich['brier']:.4f}")

# ================================================================
# Save calibration package
# ================================================================
threshold_path = f"{best_model_path}/threshold.json"
with open(threshold_path,"w",encoding="utf-8") as f:
    json.dump({"version": "phobert_full",
               "selection_metric": PRIMARY_THRESHOLD_OBJECTIVE,
               "deployment_mode": deploy_mode,
               "threshold": float(deploy_threshold),
               "temperature": deploy_temperature,
               "validation_best_raw": best_raw,
               "validation_best_scaled": val_scaled_best}, f, ensure_ascii=False, indent=2)

temperature_path = f"{best_model_path}/temperature_scaling.json"
with open(temperature_path,"w",encoding="utf-8") as f:
    json.dump({"enabled": bool(USE_TEMP_SCALING), "result": temperature_result},
              f, ensure_ascii=False, indent=2)

with open(f"{RESULTS_BASE}/calibration_summary.json","w",encoding="utf-8") as f:
    json.dump({"primary_threshold_objective": PRIMARY_THRESHOLD_OBJECTIVE,
               "deploy_mode": deploy_mode, "deploy_threshold": float(deploy_threshold),
               "deploy_temperature": deploy_temperature,
               "temperature_result": temperature_result,
               "raw_best_by_objective": val_best_by_objective,
               "scaled_best_by_objective": val_scaled_best_by_objective}, f, ensure_ascii=False, indent=2)

# ================================================================
# Error analysis
# ================================================================
test_texts = raw_text["test"] or ["<NO_TEXT>"] * len(test_labels)
fn_idx = np.where((test_labels==1) & (final_test_preds==0))[0]
fp_idx = np.where((test_labels==0) & (final_test_preds==1))[0]
rng    = np.random.default_rng(SEED)
fn_pick = rng.choice(fn_idx, size=min(N_ERROR_SAMPLES,len(fn_idx)), replace=False) if len(fn_idx) else []
fp_pick = rng.choice(fp_idx, size=min(N_ERROR_SAMPLES,len(fp_idx)), replace=False) if len(fp_idx) else []

def pack_samples(idxs, kind):
    out = [{"index": int(i), "type": kind, "text": test_texts[int(i)],
            "label_true": int(test_labels[int(i)]), "prob_toxic": float(final_test_probs[int(i)]),
            "pred": int(final_test_preds[int(i)])} for i in idxs]
    out.sort(key=lambda x: x["prob_toxic"], reverse=(kind=="FP"))
    return out

with open(f"{RESULTS_BASE}/error_analysis.json","w",encoding="utf-8") as f:
    json.dump({"deployment_mode": deploy_mode, "threshold_used": float(deploy_threshold),
               "temperature_used": deploy_temperature,
               "counts": {"FN": int(len(fn_idx)), "FP": int(len(fp_idx)),
                          "total_test": int(len(test_labels))},
               "FN_samples": pack_samples(fn_pick,"FN"),
               "FP_samples": pack_samples(fp_pick,"FP")}, f, ensure_ascii=False, indent=2)

# ================================================================
# Full metrics bundle
# ================================================================
results = {
    "dataset_version": run_dataset_tag, "model": MODEL_NAME,
    "env": {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
    "config": {
        "MODEL_NAME": MODEL_NAME, "MAX_LENGTH": MAX_LENGTH,
        "BATCH_SIZE": BATCH_SIZE, "GRAD_ACCUM": GRAD_ACCUM,
        "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO, "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE, "head_dropout": HEAD_DROPOUT,
        "use_focal": USE_FOCAL, "focal_gamma": FOCAL_GAMMA,
        "primary_metric": PRIMARY_METRIC,
        "threshold_objectives": THRESH_OBJECTIVES,
        "primary_threshold_objective": PRIMARY_THRESHOLD_OBJECTIVE,
        "ece_bins": N_BINS_ECE, "use_temp_scaling": USE_TEMP_SCALING,
        "temp_lr": TEMP_LR, "temp_max_iters": TEMP_MAX_ITERS,
        "training_type": "pseudo_label_augmented" if MIXED_MODE else "gold_only",
        "pseudo_loss_weight": PSEUDO_LOSS_WEIGHT,
        "max_pseudo_ratio": MAX_PSEUDO_RATIO,
    },
    "train_label_counts": {"clean": int(num_clean), "toxic": int(num_toxic)},
    "train_composition": {
        "n_gold": int(n_train_gold),
        "n_pseudo": int(n_train_pseudo),
        "n_total": int(len(dataset["train"])),
        "pseudo_ratio_raw": float(n_train_pseudo / max(1, len(dataset["train"]))),
        "pseudo_ratio_stats": pseudo_ratio_stats,
    },
    "class_weights": [float(x) for x in class_weights.tolist()],
    "best_model_checkpoint": trainer.state.best_model_checkpoint,
    "test_argmax_basic":       test_argmax_basic,
    "test_threshold_0p5_rich": test_thr05_rich,
    "threshold_tuning_raw": {
        "search": {"min": THRESH_MIN,"max": THRESH_MAX,"step": THRESH_STEP},
        "best_by_objective": val_best_by_objective, "threshold_file": threshold_path,
    },
    "temperature_scaling": {
        "enabled": bool(USE_TEMP_SCALING), "result": temperature_result,
        "temperature_file": temperature_path,
    },
    "test_tuned_raw_threshold_rich":    test_tuned_raw_rich,
    "test_tuned_scaled_threshold_rich": test_tuned_scaled_rich,
    "deployment_selection": {"mode": deploy_mode,
                             "threshold": float(deploy_threshold),
                             "temperature": deploy_temperature},
    "final_test_rich": final_test_rich,
}
with open(f"{RESULTS_BASE}/metrics.json","w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(f"{RESULTS_BASE}/report_thr0p5.txt","w",encoding="utf-8") as f:
    f.write(test_thr05_rich["classification_report"])
with open(f"{RESULTS_BASE}/report_final.txt","w",encoding="utf-8") as f:
    f.write(final_test_rich["classification_report"])

log("Done. Saved model + metrics + calibration + error analysis.")

best_eval_f1_toxic = None
best_eval_macro_f1 = None
best_epoch = None
for row in trainer.state.log_history:
    if "eval_f1_toxic" in row:
        v = float(row["eval_f1_toxic"])
        if best_eval_f1_toxic is None or v > best_eval_f1_toxic:
            best_eval_f1_toxic = v
            best_eval_macro_f1 = float(row.get("eval_macro_f1")) if row.get("eval_macro_f1") is not None else best_eval_macro_f1
            best_epoch = int(row.get("epoch")) if row.get("epoch") is not None else best_epoch

training_manifest = {
    "run_id": RUN_ID,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "training_type": "pseudo_label_augmented" if MIXED_MODE else "gold_only",
    "gold_data": {
        "dir": gold_data_dir_used,
        "prefix": gold_prefix_used,
        "n_train": int(n_train_gold),
        "n_val": int(len(dataset["validation"])),
        "n_test": int(len(dataset["test"])),
    },
    "pseudo_labels": {
        "batch_id": pseudo_manifest.get("batch_id"),
        "batch_manifest_path": pseudo_manifest_path,
        "seed_model": pseudo_manifest.get("seed_model"),
        "n_accepted_toxic": int(pseudo_manifest.get("n_accepted_toxic", 0)),
        "n_accepted_clean": int(pseudo_manifest.get("n_accepted_clean", 0)),
        "pseudo_loss_weight": float(PSEUDO_LOSS_WEIGHT),
        "max_pseudo_ratio": float(MAX_PSEUDO_RATIO),
    } if MIXED_MODE else None,
    "hyperparams": {
        "base_model": MODEL_NAME,
        "epochs": int(EPOCHS),
        "lr": float(LR),
        "batch_size": int(BATCH_SIZE),
        "max_length": int(MAX_LENGTH),
        "seed": int(SEED),
    },
    "results": {
        "best_epoch": best_epoch,
        "best_eval_f1_toxic": best_eval_f1_toxic,
        "best_eval_macro_f1": best_eval_macro_f1,
        "test_f1_toxic": float(final_test_rich.get("f1_toxic")) if final_test_rich.get("f1_toxic") is not None else None,
        "test_macro_f1": float(final_test_rich.get("macro_f1")) if final_test_rich.get("macro_f1") is not None else None,
    },
    "promotion_status": "pending",
}

training_manifest_path = f"{best_model_path}/training_manifest.json"
with open(training_manifest_path, "w", encoding="utf-8") as f:
    json.dump(training_manifest, f, ensure_ascii=False, indent=2)

run_manifest_path = f"{RUN_MANIFEST_DIR.rstrip('/')}/{RUN_ID}.json"
shutil.copyfile(training_manifest_path, run_manifest_path)

# ================================================================
# Metadata export (drop-in at end)
# ================================================================
MODEL_ID = MODEL_VERSION
DATASET_VERSION = run_dataset_tag
IS_BASELINE = True

# --- training curve from trainer log_history ---
curve = []
for row in trainer.state.log_history:
    if "epoch" in row and "loss" in row:
        curve_row = {
            "epoch": row.get("epoch"),
            "loss": row.get("loss"),
        }
        if "eval_macro_f1" in row:
            curve_row["f1"] = row.get("eval_macro_f1")
        if "eval_f1_toxic" in row:
            curve_row["f1_toxic"] = row.get("eval_f1_toxic")
        curve.append(curve_row)

run_config = {
    "run_id": RUN_ID,
    "model_name": MODEL_ID,
    "dataset_version": DATASET_VERSION,
    "model_version": MODEL_VERSION,
    "policy_version": POLICY_VERSION,
    "created_at": datetime.now().isoformat(),
    "is_baseline": IS_BASELINE,
    "hyperparameters": {
        "MODEL_NAME": MODEL_NAME,
        "MAX_LENGTH": MAX_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "GRAD_ACCUM": GRAD_ACCUM,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "head_dropout": HEAD_DROPOUT,
        "use_focal": USE_FOCAL,
        "focal_gamma": FOCAL_GAMMA,
        "primary_metric": PRIMARY_METRIC,
    },
}

metrics_payload = {}
if isinstance(results, dict) and "final_test_rich" in results:
    metrics_payload = results["final_test_rich"]
elif isinstance(results, dict) and "test_argmax_basic" in results:
    metrics_payload = results["test_argmax_basic"]

metrics_out = {
    "macro_f1": metrics_payload.get("macro_f1"),
    "f1_toxic": metrics_payload.get("f1_toxic"),
    "precision": metrics_payload.get("precision_toxic"),
    "recall": metrics_payload.get("recall_toxic"),
    "accuracy": metrics_payload.get("accuracy"),
}

training_type = "pseudo_label_augmented" if MIXED_MODE else "gold_only"
pseudo_batch_id = pseudo_manifest.get("batch_id") if MIXED_MODE and pseudo_manifest else "none"

def _build_epoch_mlflow_metrics(log_history):
    rows = []
    latest_train_loss = None
    for entry in log_history:
        step = entry.get("step")
        step = int(step) if isinstance(step, (int, float)) else None

        if entry.get("loss") is not None:
            latest_train_loss = float(entry["loss"])
            rows.append((step, {"train_loss": latest_train_loss}))

        if any(k in entry for k in ["eval_f1_toxic", "eval_macro_f1", "eval_loss"]):
            metrics = {}
            if entry.get("eval_f1_toxic") is not None:
                metrics["eval_f1_toxic"] = float(entry["eval_f1_toxic"])
            if entry.get("eval_macro_f1") is not None:
                metrics["eval_macro_f1"] = float(entry["eval_macro_f1"])
            if entry.get("eval_loss") is not None:
                metrics["eval_loss"] = float(entry["eval_loss"])
            if latest_train_loss is not None:
                metrics["train_loss"] = latest_train_loss
            if metrics:
                rows.append((step, metrics))
    return rows

epoch_mlflow_metrics = _build_epoch_mlflow_metrics(trainer.state.log_history)

mlflow_params = {
    "training_type": training_type,
    "base_model": MODEL_NAME,
    "epochs": int(EPOCHS),
    "lr": float(LR),
    "batch_size": int(BATCH_SIZE),
    "max_length": int(MAX_LENGTH),
    "seed": int(SEED),
    "gold_n_train": int(n_train_gold),
    "gold_n_val": int(len(dataset["validation"])),
}
if MIXED_MODE and pseudo_manifest:
    mlflow_params.update({
        "pseudo_loss_weight": float(PSEUDO_LOSS_WEIGHT),
        "max_pseudo_ratio": float(MAX_PSEUDO_RATIO),
        "pseudo_batch_id": str(pseudo_manifest.get("batch_id", "")),
        "pseudo_seed_model": str(pseudo_manifest.get("seed_model", "")),
        "pseudo_n_accepted_toxic": int(pseudo_manifest.get("n_accepted_toxic", 0)),
        "pseudo_n_accepted_clean": int(pseudo_manifest.get("n_accepted_clean", 0)),
    })

mlflow_tags = {
    "mlflow.runName": RUN_ID,
    "training_type": training_type,
    "pseudo_label_batch": pseudo_batch_id,
}

with open(f"{best_model_path}/run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)

with open(f"{best_model_path}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)

with open(f"{best_model_path}/training_curve.json", "w", encoding="utf-8") as f:
    json.dump(curve, f, ensure_ascii=False, indent=2)

if mlflow_active and mlflow is not None:
    try:
        try:
            mlflow.set_tags(mlflow_tags)
        except Exception:
            pass

        for k, v in mlflow_params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        for idx, (step, metric_row) in enumerate(epoch_mlflow_metrics):
            try:
                effective_step = step if step is not None else idx
                for mk, mv in metric_row.items():
                    mlflow.log_metric(mk, mv, step=effective_step)
            except Exception:
                pass

        final_mlflow_metrics = {
            "test_f1_toxic": float(final_test_rich.get("f1_toxic") or 0.0),
            "test_macro_f1": float(final_test_rich.get("macro_f1") or 0.0),
        }
        for k, v in final_mlflow_metrics.items():
            try:
                mlflow.log_metric(k, v)
            except Exception:
                pass

        try:
            mlflow.log_artifact(training_manifest_path)
        except Exception:
            pass
    except Exception as exc:
        print(f"MLflow logging disabled: {exc}", flush=True)
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass

# ================================================================
# ZIP EXPORT
# ================================================================
zip_names = {
    "best": f"best_model_full_{run_dataset_tag}",
    "results": f"results_full_{run_dataset_tag}",
}
for name, path in [
    (zip_names["best"], best_model_path),
    (zip_names["results"], RESULTS_BASE),
]:
    if os.path.exists(path):
        log(f"Zipping {path} → {name}.zip ...")
        shutil.make_archive(name, "zip", path)

log("All zips created.")

try:
    from google.colab import files
    files.download(f"{zip_names['best']}.zip")
    files.download(f"{zip_names['results']}.zip")
    log("Download triggered.")
except Exception:
    log("Not running in Colab, skip files.download().")

print(f"Model saved to {best_model_path}", flush=True)
print(f"Run manifest: {run_manifest_path}", flush=True)
print("Promotion status: pending — run Phase E promotion check", flush=True)

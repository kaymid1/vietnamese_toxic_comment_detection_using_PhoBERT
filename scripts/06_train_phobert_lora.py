"""
PhoBERT LoRA Fine-tuning — Vietnamese Toxic Comment Detection
Refactored from original full fine-tuning script to use PEFT / LoRA.

Key changes vs original:
  - peft library: LoraConfig + get_peft_model wrap the base model
  - Only LoRA adapters + classifier head are trained (~1-3% of params)
  - FREEZE_EPOCHS removed (LoRA already handles encoder freezing)
  - merge_and_unload() saves a standard HF model — no peft dep at inference
  - lora_adapter/ saved separately (small, reusable for future experiments)
  - OUTPUT_BASE / RESULTS_BASE renamed to *_lora so both runs coexist
  - LR bumped to 2e-4 (LoRA works better with higher LR than full fine-tuning)
  - All evaluation / threshold / temperature-scaling logic unchanged
"""

# -----------------------
# Install
# -----------------------
# !pip -q install -U transformers datasets scikit-learn torch accelerate peft

import os, json, random, time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model
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
DATA_DIR = "/content/drive/MyDrive/victsd"
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256

BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS     = 10
LR         = 2e-4       # higher than full fine-tuning (typical for LoRA)
WEIGHT_DECAY  = 0.05
WARMUP_RATIO  = 0.08
MAX_GRAD_NORM = 1.0

LABEL_SMOOTHING     = 0.0
EARLY_STOP_PATIENCE = 4
HEAD_DROPOUT        = 0.1

USE_FOCAL   = True  # False giảm nhạy FP ở clean
FOCAL_GAMMA = 2.0

TOXIC_WEIGHT_SCALE = 0.5  # <1.0 để giảm thiên lệch toxic

SEED = 42

OUTPUT_BASE  = "models/phobert_lora"   # keeps separate from original run
RESULTS_BASE = "results/phobert_lora"

PRIMARY_METRIC = "f1_toxic"

THRESH_MIN  = 0.05
THRESH_MAX  = 0.95
THRESH_STEP = 0.01
THRESH_OBJECTIVES           = ["f1_toxic", "macro_f1"]
PRIMARY_THRESHOLD_OBJECTIVE = "f1_toxic"

N_ERROR_SAMPLES = 30

# Calibration
EPS            = 1e-12
N_BINS_ECE     = 10
USE_TEMP_SCALING = True
TEMP_LR        = 0.01
TEMP_MAX_ITERS = 300

# ---- LoRA --------------------------------------------------------
# PhoBERT is RoBERTa-based; attention proj names: query, key, value, dense.
# query+value is the standard efficient choice (original LoRA paper).
# For more capacity: raise LORA_R (32/64) or add "key","dense" to target_modules.
LORA_R               = 32          # tăng từ 16 → 32
LORA_ALPHA           = 64          # giữ alpha = 2*r
LORA_DROPOUT         = 0.
LORA_TARGET_MODULES  = ["query", "key", "value", "dense"]  # thêm key + dense
LORA_BIAS            = "none"
# classifier is outside the backbone — list in modules_to_save to keep it trainable
LORA_MODULES_TO_SAVE = ["classifier"]
# ------------------------------------------------------------------

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
log("Checking dataset files...")
for fname in [
    "train_augmented.jsonl",
    "validation_augmented.jsonl",
    "test_augmented.jsonl",
]:
    assert os.path.exists(f"{DATA_DIR}/{fname}"), f"Missing {fname}"

os.makedirs(f"{OUTPUT_BASE}/checkpoints", exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/best",        exist_ok=True)
os.makedirs(RESULTS_BASE,                 exist_ok=True)

log("Loading dataset ...")
dataset = load_dataset("json", data_files={
    "train":      f"{DATA_DIR}/train_augmented.jsonl",
    "validation": f"{DATA_DIR}/validation_augmented.jsonl",
    "test":       f"{DATA_DIR}/test_augmented.jsonl",
})

raw_text = {
    split: dataset[split]["text"] if "text" in dataset[split].column_names else None
    for split in ["validation", "test"]
}
log(f"Sizes: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")

# ================================================================
# Tokenizer
# ================================================================
log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_batch(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

log("Tokenizing...")
tokenized_dataset = dataset.map(tokenize_batch, batched=True)
cols_to_remove = [c for c in ["text", "meta"] if c in tokenized_dataset["train"].column_names]
if cols_to_remove:
    tokenized_dataset = tokenized_dataset.remove_columns(cols_to_remove)
if "label" in tokenized_dataset["train"].column_names:
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# ================================================================
# Base model + dropout tweaks
# ================================================================
log("Loading base model ...")
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

for attr, default in [
    ("classifier_dropout",         HEAD_DROPOUT),
    ("hidden_dropout_prob",        HEAD_DROPOUT),
    ("attention_probs_dropout_prob", HEAD_DROPOUT),
]:
    if hasattr(base_model.config, attr):
        current = getattr(base_model.config, attr)
        setattr(base_model.config, attr, max(float(current or 0), float(default)))

# ================================================================
# Apply LoRA
# ================================================================
log("Applying LoRA adapter...")

lora_cfg = LoraConfig(
    task_type       = TaskType.SEQ_CLS,
    r               = LORA_R,
    lora_alpha      = LORA_ALPHA,
    lora_dropout    = LORA_DROPOUT,
    target_modules  = LORA_TARGET_MODULES,
    bias            = LORA_BIAS,
    modules_to_save = LORA_MODULES_TO_SAVE,
    inference_mode  = False,
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
log(f"LoRA trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

lora_config_dict = {
    "r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
    "target_modules": LORA_TARGET_MODULES, "bias": LORA_BIAS,
    "modules_to_save": LORA_MODULES_TO_SAVE, "learning_rate": LR,
}

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
def focal_loss(logits, labels, weight=None, gamma=2.0):
    logp  = F.log_softmax(logits, dim=-1)
    p     = torch.exp(logp)
    pt    = p.gather(1, labels.unsqueeze(1)).squeeze(1)
    logpt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
    at    = weight.to(logits.device).gather(0, labels) if weight is not None else 1.0
    return (-at * (1 - pt) ** gamma * logpt).mean()

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing=0.0,
                 use_focal=False, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights   = class_weights
        self.label_smoothing = float(label_smoothing or 0.0)
        self.use_focal       = bool(use_focal)
        self.focal_gamma     = float(focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits  = outputs["logits"]
        w       = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if self.use_focal:
            loss = focal_loss(logits, labels, weight=w, gamma=self.focal_gamma)
        elif self.label_smoothing > 0.0:
            lp     = F.log_softmax(logits, dim=-1)
            nll    = F.nll_loss(lp, labels, weight=w, reduction="mean")
            smooth = -lp.mean(dim=-1).mean()
            loss   = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            loss = torch.nn.CrossEntropyLoss(weight=w)(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ================================================================
# Metrics helpers  (identical to original)
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
)

log("Training with LoRA adapters...")
trainer.train()
log(f"Training finished. Best checkpoint: {trainer.state.best_model_checkpoint}")

# ================================================================
# Merge LoRA → save standalone model (no peft dep at inference)
# ================================================================
best_model_path = f"{OUTPUT_BASE}/best"
adapter_path    = f"{OUTPUT_BASE}/lora_adapter"
os.makedirs(adapter_path, exist_ok=True)

log("Merging LoRA adapters into base weights...")
merged_model = model.merge_and_unload()       # fuses delta weights → plain HF model
merged_model.save_pretrained(best_model_path)
tokenizer.save_pretrained(best_model_path)

model.save_pretrained(adapter_path)           # adapter_config.json + small adapter weights
log(f"Saved merged model  → {best_model_path}")
log(f"Saved LoRA adapter  → {adapter_path}")

with open(f"{RESULTS_BASE}/lora_config.json", "w", encoding="utf-8") as f:
    json.dump(lora_config_dict, f, ensure_ascii=False, indent=2)

# ================================================================
# Inference (use merged model)
# ================================================================
log("Running prediction on validation / test...")
trainer.model = merged_model

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
    json.dump({"version": "phobert_lora",
               "selection_metric": PRIMARY_THRESHOLD_OBJECTIVE,
               "deployment_mode": deploy_mode,
               "threshold": float(deploy_threshold),
               "temperature": deploy_temperature,
               "validation_best_raw": best_raw,
               "validation_best_scaled": val_scaled_best,
               "lora_config": lora_config_dict}, f, ensure_ascii=False, indent=2)

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
               "scaled_best_by_objective": val_scaled_best_by_objective,
               "lora_config": lora_config_dict}, f, ensure_ascii=False, indent=2)

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
    "dataset_version": "victsd", "model": MODEL_NAME,
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
        "lora": lora_config_dict,
    },
    "train_label_counts": {"clean": int(num_clean), "toxic": int(num_toxic)},
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

log("Done. Saved merged model + LoRA adapter + metrics + calibration + error analysis.")



# ================================================================
# Metadata export (drop-in at end)
# ================================================================
import json
from datetime import datetime

MODEL_ID = "phobert/lora_v1"        # ví dụ: phobert/lora_v1
DATASET_VERSION = "victsd_vihsd"
IS_BASELINE = False

# --- training curve from trainer log_history ---
curve = []
for row in trainer.state.log_history:
    if "epoch" in row and "loss" in row:
        curve_row = {
            "epoch": row.get("epoch"),
            "loss": row.get("loss"),
        }
        # nếu có eval metrics thì thêm vào
        if "eval_macro_f1" in row:
            curve_row["f1"] = row.get("eval_macro_f1")
        if "eval_f1_toxic" in row:
            curve_row["f1_toxic"] = row.get("eval_f1_toxic")
        curve.append(curve_row)

run_config = {
    "run_id": f"{MODEL_ID}_run",
    "model_name": MODEL_ID,
    "dataset_version": DATASET_VERSION,
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
        "lora": lora_config_dict,
    },
}

# --- metrics.json ---
# ưu tiên final_test_rich nếu có
metrics_payload = {}
if isinstance(results, dict) and "final_test_rich" in results:
    metrics_payload = results["final_test_rich"]
elif isinstance(results, dict) and "test_argmax_basic" in results:
    metrics_payload = results["test_argmax_basic"]

# fallback nếu bạn muốn map thẳng từ final_test_rich
# expected keys: macro_f1, f1_toxic, precision_toxic, recall_toxic, accuracy
metrics_out = {
    "macro_f1": metrics_payload.get("macro_f1"),
    "f1_toxic": metrics_payload.get("f1_toxic"),
    "precision": metrics_payload.get("precision_toxic"),
    "recall": metrics_payload.get("recall_toxic"),
    "accuracy": metrics_payload.get("accuracy"),
}

# write files into merged model folder
with open(f"{best_model_path}/run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)

with open(f"{best_model_path}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)

with open(f"{best_model_path}/training_curve.json", "w", encoding="utf-8") as f:
    json.dump(curve, f, ensure_ascii=False, indent=2)

# ================================================================
# ZIP EXPORT
# ================================================================
import shutil
from google.colab import files

for name, path in [
    ("best_model_lora",   best_model_path),
    ("lora_adapter",      adapter_path),
    ("results_lora",      RESULTS_BASE),
]:
    if os.path.exists(path):
        log(f"Zipping {path} → {name}.zip ...")
        shutil.make_archive(name, "zip", path)

log("All zips created.")
files.download("best_model_lora.zip")
files.download("lora_adapter.zip")
log("Download triggered.")
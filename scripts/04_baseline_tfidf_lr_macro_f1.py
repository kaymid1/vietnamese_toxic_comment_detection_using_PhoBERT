import json
import os
import random
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ================================================================
# Config — match LoRA script data source / export layout
# ================================================================
DATA_DIR = os.environ.get("DATA_DIR", "/content/drive/MyDrive/victsd_gold")
DATASET_PREFIX = os.environ.get("DATASET_PREFIX", "")
SEED = int(os.environ.get("SEED", "42"))

MODEL_ID = "tfidf_lr/baseline-macro-f1"
run_dataset_tag = DATASET_PREFIX if DATASET_PREFIX else "victsd_gold"
DATASET_VERSION = run_dataset_tag
IS_BASELINE = True

OUTPUT_BASE = os.environ.get("OUTPUT_BASE", f"models/tfidf_lr/{run_dataset_tag}")
RESULTS_BASE = os.environ.get("RESULTS_BASE", f"results/baseline/{run_dataset_tag}")

random.seed(SEED)
np.random.seed(SEED)

# ================================================================
# Dataset
# ================================================================
split_names = ["train", "validation", "test"]

def _resolve_split_paths(data_dir, dataset_prefix):
    resolved = {}
    for split in split_names:
        candidates = [
            os.path.join(data_dir, f"{split}.jsonl"),
            os.path.join(data_dir, f"{dataset_prefix}_{split}.jsonl") if dataset_prefix else None,
            os.path.join(data_dir, f"{dataset_prefix}_{split}_augmented.jsonl") if dataset_prefix else None,
        ]
        candidates = [c for c in candidates if c]
        found = next((p for p in candidates if os.path.exists(p)), None)
        if not found:
            attempted = " | ".join(candidates)
            raise FileNotFoundError(f"Missing {split} file. Tried: {attempted}")
        resolved[split] = found
    return resolved

dataset_files = _resolve_split_paths(DATA_DIR, DATASET_PREFIX)

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)

print("Loading dataset ...")
dataset = load_dataset("json", data_files=dataset_files)

train_texts = [ex["text"] for ex in dataset["train"]]
train_labels = [ex["toxicity"] for ex in dataset["train"]]

val_texts = [ex["text"] for ex in dataset["validation"]]
val_labels = [ex["toxicity"] for ex in dataset["validation"]]

test_texts = [ex["text"] for ex in dataset["test"]]
test_labels = [ex["toxicity"] for ex in dataset["test"]]

print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

# ================================================================
# TF-IDF Vectorizer
# ================================================================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=False,
    max_features=None,
    min_df=1,
    token_pattern=r"(?u)\b\w+\b",
)

X_train = vectorizer.fit_transform(train_texts)
X_val   = vectorizer.transform(val_texts)
X_test  = vectorizer.transform(test_texts)

# ================================================================
# Logistic Regression
# ================================================================
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=SEED,
    n_jobs=-1,
)

model.fit(X_train, train_labels)

# ================================================================
# Evaluation
# ================================================================
def evaluate(split_name, y_true, X):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    f1_toxic = f1_score(y_true, y_pred, pos_label=1)
    f1_clean = f1_score(y_true, y_pred, pos_label=0)

    print(f"\n=== {split_name.upper()} RESULTS ===")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"Macro-F1: {macro_f1:.4f} | F1_toxic: {f1_toxic:.4f} | F1_clean: {f1_clean:.4f}")

    return {
        "macro_f1": macro_f1,
        "f1_toxic": f1_toxic,
        "f1_clean": f1_clean,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "prob_pos": y_prob.tolist(),
    }

val_metrics = evaluate("validation", val_labels, X_val)
test_metrics = evaluate("test", test_labels, X_test)

# ================================================================
# Save results
# ================================================================
results = {
    "dataset_version": DATASET_VERSION,
    "model": "TFIDF_LR",
    "validation": val_metrics,
    "test": test_metrics,
}

with open(f"{RESULTS_BASE}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Save vectorizer & model
import joblib
joblib.dump(vectorizer, f"{RESULTS_BASE}/vectorizer.pkl")
joblib.dump(model, f"{RESULTS_BASE}/model_lr.pkl")

# ================================================================
# Metadata export (run_config + metrics + curve)
# ================================================================
from datetime import datetime

curve = []
# baseline does not log epochs; keep empty curve for consistency

run_config = {
    "run_id": f"{MODEL_ID}_run",
    "model_name": MODEL_ID,
    "dataset_version": DATASET_VERSION,
    "created_at": datetime.now().isoformat(),
    "is_baseline": IS_BASELINE,
    "hyperparameters": {
        "ngram_range": [1, 2],
        "lowercase": False,
        "min_df": 1,
        "class_weight": "balanced",
        "max_iter": 1000,
    },
}

metrics_payload = test_metrics
metrics_out = {
    "macro_f1": metrics_payload.get("macro_f1"),
    "f1_toxic": metrics_payload.get("f1_toxic"),
    "precision": metrics_payload.get("classification_report", {}).get("1", {}).get("precision"),
    "recall": metrics_payload.get("classification_report", {}).get("1", {}).get("recall"),
    "accuracy": metrics_payload.get("classification_report", {}).get("accuracy"),
}

os.makedirs(f"{OUTPUT_BASE}/best", exist_ok=True)
with open(f"{OUTPUT_BASE}/best/run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_BASE}/best/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_BASE}/best/training_curve.json", "w", encoding="utf-8") as f:
    json.dump(curve, f, ensure_ascii=False, indent=2)

# ================================================================
# ZIP EXPORT
# ================================================================
import shutil

zip_names = {
    "best": f"best_model_tfidf_lr_{run_dataset_tag}",
    "results": f"results_tfidf_lr_{run_dataset_tag}",
}
ZIP_OUTPUT_DIR = os.environ.get("ZIP_OUTPUT_DIR", DATA_DIR)
os.makedirs(ZIP_OUTPUT_DIR, exist_ok=True)

for name, path in [
    (zip_names["best"], f"{OUTPUT_BASE}/best"),
    (zip_names["results"], RESULTS_BASE),
]:
    if os.path.exists(path):
        zip_base = os.path.join(ZIP_OUTPUT_DIR, name)
        print(f"Zipping {path} → {zip_base}.zip ...")
        shutil.make_archive(zip_base, "zip", path)

print(f"All zips created in {ZIP_OUTPUT_DIR}")

print(f"\n✅ Baseline TF-IDF + LR hoàn tất – results lưu tại {RESULTS_BASE}/")

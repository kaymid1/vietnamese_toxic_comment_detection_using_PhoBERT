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
DATA_DIR = "/content/drive/MyDrive/victsd"
SEED = 42

MODEL_ID = "tfidf_lr/baseline"
DATASET_VERSION = "victsd_vihsd"
IS_BASELINE = True

OUTPUT_BASE  = "models/tfidf_lr"
RESULTS_BASE = "results/baseline"

random.seed(SEED)
np.random.seed(SEED)

# ================================================================
# Dataset
# ================================================================
for fname in [
    "train_augmented.jsonl",
    "validation_augmented.jsonl",
    "test_augmented.jsonl",
]:
    assert os.path.exists(f"{DATA_DIR}/{fname}"), f"Missing {fname}"

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)

print("Loading dataset ...")
dataset = load_dataset("json", data_files={
    "train":      f"{DATA_DIR}/train_augmented.jsonl",
    "validation": f"{DATA_DIR}/validation_augmented.jsonl",
    "test":       f"{DATA_DIR}/test_augmented.jsonl",
})

train_texts = [ex["text"] for ex in dataset["train"]]
train_labels = [ex["label"] for ex in dataset["train"]]

val_texts = [ex["text"] for ex in dataset["validation"]]
val_labels = [ex["label"] for ex in dataset["validation"]]

test_texts = [ex["text"] for ex in dataset["test"]]
test_labels = [ex["label"] for ex in dataset["test"]]

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

for name, path in [
    ("best_model_tfidf_lr", f"{OUTPUT_BASE}/best"),
    ("results_tfidf_lr", RESULTS_BASE),
]:
    if os.path.exists(path):
        print(f"Zipping {path} → {name}.zip ...")
        shutil.make_archive(name, "zip", path)

print("All zips created.")
try:
    from google.colab import files
    files.download("best_model_tfidf_lr.zip")
    files.download("results_tfidf_lr.zip")
    print("Download triggered.")
except Exception:
    print("Not running in Colab, skip files.download().")

print(f"\n✅ Baseline TF-IDF + LR hoàn tất – results lưu tại {RESULTS_BASE}/")

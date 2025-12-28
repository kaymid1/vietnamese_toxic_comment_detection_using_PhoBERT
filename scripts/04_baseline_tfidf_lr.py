import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from datasets import load_dataset
import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/processed/victsd_v1"
SEED = 42
np.random.seed(SEED)

# Load dataset
dataset = load_dataset("json", data_files={
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/validation.jsonl",
    "test": f"{DATA_DIR}/test.jsonl"
})

train_texts = [ex["text"] for ex in dataset["train"]]
train_labels = [ex["label"] for ex in dataset["train"]]

val_texts = [ex["text"] for ex in dataset["validation"]]
val_labels = [ex["label"] for ex in dataset["validation"]]

test_texts = [ex["text"] for ex in dataset["test"]]
test_labels = [ex["label"] for ex in dataset["test"]]

print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

# -----------------------------
# TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=False,          # giữ nguyên case như preprocessing
    max_features=None,        # không giới hạn để fair comparison
    min_df=1,
    token_pattern=r"(?u)\b\w+\b"  # đơn giản, phù hợp tiếng Việt không space-based
)

X_train = vectorizer.fit_transform(train_texts)
X_val   = vectorizer.transform(val_texts)
X_test  = vectorizer.transform(test_texts)

# -----------------------------
# Logistic Regression
# -----------------------------
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=SEED,
    n_jobs=-1
)

model.fit(X_train, train_labels)

# -----------------------------
# Evaluation
# -----------------------------
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
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

# Eval trên validation trước
val_metrics = evaluate("validation", val_labels, X_val)

# Final eval trên test
test_metrics = evaluate("test", test_labels, X_test)

# -----------------------------
# Save results
# -----------------------------
os.makedirs("results/baseline", exist_ok=True)

with open("results/baseline/metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset_version": "victsd_v1",
        "model": "TFIDF_LR",
        "validation": val_metrics,
        "test": test_metrics
    }, f, ensure_ascii=False, indent=2)

# Save vectorizer & model nếu cần (optional)
import joblib
joblib.dump(vectorizer, "results/baseline/vectorizer.pkl")
joblib.dump(model, "results/baseline/model_lr.pkl")

print("\n✅ Baseline TF-IDF + LR hoàn tất – results lưu tại results/baseline/")
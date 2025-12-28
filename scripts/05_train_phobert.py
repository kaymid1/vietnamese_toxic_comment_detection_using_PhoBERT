import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch
import mlflow

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/processed/victsd_v1"
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
SEED = 42

os.makedirs("models/phobert", exist_ok=True)
os.makedirs("results/phobert", exist_ok=True)

# MLflow experiment
mlflow.set_experiment("Vietnamese Toxic Comment Detection")
mlflow.start_run(run_name="phobert_base_victsd_v1")

mlflow.log_param("dataset_version", "victsd_v1")
mlflow.log_param("model", MODEL_NAME)
mlflow.log_param("max_length", MAX_LENGTH)
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("epochs", EPOCHS)
mlflow.log_param("learning_rate", LR)

# Load dataset
dataset = load_dataset("json", data_files={
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/validation.jsonl",
    "test": f"{DATA_DIR}/test.jsonl"
})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize_batch, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text", "meta"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# Class weights for imbalanced dataset
train_labels = torch.tensor(tokenized_dataset["train"]["labels"])
label_counts = torch.bincount(train_labels, minlength=2)
num_clean = label_counts[0].item()
num_toxic = label_counts[1].item()
weight_clean = 1.0
weight_toxic = num_clean / num_toxic
class_weights = torch.tensor([weight_clean, weight_toxic], dtype=torch.float)

# Custom Trainer with weighted CrossEntropyLoss
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    macro_f1 = f1_score(labels, preds, average="macro")
    f1_toxic = f1_score(labels, preds, pos_label=1)
    f1_clean = f1_score(labels, preds, pos_label=0)
    
    return {
        "macro_f1": macro_f1,
        "f1_toxic": f1_toxic,
        "f1_clean": f1_clean
    }

# Training args
training_args = TrainingArguments(
    output_dir="models/phobert/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    seed=SEED,
    logging_dir="models/phobert/logs",
    report_to=[]  # disable wandb
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

# Train
trainer.train()

# Save best model + tokenizer
best_model_path = "models/phobert/best"
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)

# Final eval on test
test_results = trainer.predict(tokenized_dataset["test"])
test_metrics = compute_metrics((test_results.predictions, test_results.label_ids))

y_pred = np.argmax(test_results.predictions, axis=1)
y_true = test_results.label_ids

class_report = classification_report(y_true, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred).tolist()

# Log to MLflow
mlflow.log_metrics({
    "test_macro_f1": test_metrics["macro_f1"],
    "test_f1_toxic": test_metrics["f1_toxic"],
    "test_f1_clean": test_metrics["f1_clean"]
})

# Save results
results = {
    "dataset_version": "victsd_v1",
    "model": "PhoBERT-base",
    "test_metrics": test_metrics,
    "classification_report": class_report,
    "confusion_matrix": conf_matrix
}
with open("results/phobert/metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

mlflow.end_run()

print("\n✅ PhoBERT fine-tune hoàn tất!")
print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f} | F1_toxic: {test_metrics['f1_toxic']:.4f}")

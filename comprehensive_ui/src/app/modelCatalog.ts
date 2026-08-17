export const PHOBERT_V2_FINETUNED_ID = "phobert/phobert_v2_finetuned";
export const HIDDEN_UI_MODEL_IDS = new Set(["tfidf_lr/baseline_tfidf"]);

export const isHiddenUiModel = (modelId: string | null | undefined): boolean =>
  Boolean(modelId && HIDDEN_UI_MODEL_IDS.has(modelId));

const BUILTIN_MODEL_LABELS: Record<string, string> = {
  "tfidf_lr/baseline_tfidf": "TF-IDF + Logistic Regression",
  "phobert/baseline": "PhoBERT v1 Baseline",
  [PHOBERT_V2_FINETUNED_ID]: "PhoBERT v2 Fine-tuned",
  "phobert/phobert_lora_4.7": "PhoBERT v2 Fine-tuned",
};

export const getModelLabel = (
  modelId: string | null | undefined,
  apiLabels?: Record<string, string>,
): string => {
  if (!modelId) return "";
  return apiLabels?.[modelId] || BUILTIN_MODEL_LABELS[modelId] || modelId;
};

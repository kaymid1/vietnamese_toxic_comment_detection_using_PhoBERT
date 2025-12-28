# PhoBERT-base Optimization Log – ViCTSD (victsd_v1)

**Dataset version**: data/processed/victsd_v1  
**Model**: vinai/phobert-base  
**Task**: Binary Toxic Comment Classification  
**Main metric**: Macro-F1 (best model theo macro-F1 trên validation)  
**Baseline reference**: TF-IDF + LR → Test Macro-F1: 0.7043 | F1_toxic: 0.4844

## Experiment Table

| Exp ID | Date       | Description / Changes                                | Epochs | Batch | LR   | Val Macro-F1 | Val F1_toxic | Test Macro-F1 | Test F1_toxic | Test F1_clean | Notes |
|--------|------------|------------------------------------------------------|--------|-------|------|--------------|--------------|---------------|---------------|---------------|-------|
| exp-01 | 2025-12-29 | Baseline PhoBERT (CrossEntropy, no weight)           | 5      | 16    | 2e-5 | 0.7279       | 0.5121       | **0.7171**    | **0.4928**    | ~0.9414       | Validation table cho thấy F1_toxic tăng đều qua các epoch |
| exp-02 | YYYY-MM-DD | [Mô tả thay đổi tiếp theo, ví dụ: Focal Loss γ=2]    | -      | -     | -    | -            | -            | -             | -             | -             | -     |
| exp-03 | YYYY-MM-DD | ...                                                  | -      | -     | -    | -            | -            | -             | -             | -             | -     |

## Validation Progress (exp-01 detail)

| Epoch | Training Loss | Validation Loss | Macro F1 | F1 Toxic | F1 Clean |
|-------|---------------|-----------------|----------|----------|----------|
| 1     | No log        | 0.280807        | 0.695697 | 0.457275 | 0.934118 |
| 2     | 0.306600      | 0.325590        | 0.656579 | 0.366559 | 0.946598 |
| 3     | 0.212600      | 0.338753        | 0.703992 | 0.460674 | 0.947311 |
| 4     | 0.155000      | 0.403896        | 0.711724 | 0.477212 | 0.946237 |
| 5     | 0.103700      | 0.454167        | 0.727874 | 0.512077 | 0.943670 |

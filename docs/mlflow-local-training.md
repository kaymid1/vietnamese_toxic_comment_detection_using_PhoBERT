# Local training MLflow contract

Local training tracking is opt-in and uses the Phase 2B.1 HTTP server. It never
falls back to `mlruns/`, a file URI, or direct SQLite tracking.

Start the server from the repository root:

```powershell
python -m backend.mlflow_server
```

Enable tracking for a local training process:

```powershell
$env:MLFLOW_ENABLED = "true"
```

`MLFLOW_TRACKING_URI` has highest precedence. When it is unset, the client URI
is built from `MLFLOW_SERVER_HOST` (default `127.0.0.1`) and
`MLFLOW_SERVER_PORT` (default `5000`). An enabled client checks the server health
before loading training data and exits with startup instructions when the
server is unavailable. Set `MLFLOW_ENABLED=false` to run without tracking.

Local experiment names follow
`viettoxic-local-<family/version>-<training-workflow>-<objective>`. The four
workflows remain separate: PhoBERT v1 full fine-tune macro-F1, PhoBERT v2 full
fine-tune toxic-F1, PhoBERT v2 full fine-tune macro-F1, and the PhoBERT v2
adaptive retrain/finetune macro-F1 workflow. `MLFLOW_EXPERIMENT_NAME` may
explicitly override the selected name.

Every local run records model family, actual training mode, dataset, script,
run/config ID, base model, local platform, and a parent model only when the
finetune initialization metadata proves that lineage.

The Kaggle notebook remains on its existing Phase 2B.1-independent tracking
backend. Kaggle cannot reach a developer machine through `127.0.0.1`; its remote
tracking design is deferred to Phase 2B.2B.

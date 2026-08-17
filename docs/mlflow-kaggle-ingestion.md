# Kaggle MLflow post-run ingestion

Kaggle must not connect to the developer machine's `127.0.0.1` MLflow server.
It runs in an isolated network and may keep a temporary SQLite-backed MLflow
store only as non-canonical execution evidence.

The supported flow is:

```text
Kaggle training
  -> mlflow_run_evidence.json inside the model ZIP
  -> existing webhook output download
  -> existing artifact checksum/serving validation
  -> evidence schema, path, size, and SHA-256 validation
  -> canonical local MLflow HTTP API
  -> local canonical run ID recorded in feedback.db
  -> existing candidate/promotion lifecycle continues independently
```

The version 1 evidence manifest records source job/run identity, experiment and
run names, proven training initialization, separate training/tracking/artifact
statuses, safe scalar params/metrics/tags, timestamps, provenance, and every
ingested artifact. Artifact identities are ZIP-relative POSIX paths. Each entry
contains its role, required flag, byte size, and SHA-256. Absolute Windows,
macOS/Linux, file, SQLite, URL, traversal, symlink, duplicate, missing, and
unmanifested paths are rejected before MLflow is contacted. Sensitive keys and
machine-specific scalar values are not exported.

Status meanings are independent:

- `training_status`: `success` or `failed`.
- `tracking_status`: `complete`, `partial`, `failed`, or `disabled` for the
  isolated source tracking evidence.
- `artifact_status`: `complete`, `partial`, or `missing`.

A successful finetune must prove `initialization_mode=existing_model_artifact`
and retain its `parent_model_id`. If the wrapper fails before initialization can
be proven, failure evidence uses `initialization_mode=unconfirmed`; it does not
claim that finetuning actually started.

A successful model with failed source tracking remains a successful model. A
canonical MLflow outage records a retriable ingestion failure but does not
delete the downloaded evidence ZIP, registered candidate, or serving artifact.
The next completion-status poll safely retries.

The operational database stores one mapping per `(source_job_id,
source_run_id)`, plus the evidence checksum and canonical local MLflow run ID.
Exact retries return the existing run. Different evidence for an existing
source identity is a conflict requiring operator review. MLflow is also
searched by source-identity tags so a completed API write can be recovered if a
process stopped before the mapping was committed.

An `ingesting` reservation deliberately blocks concurrent duplicate writers. A
normal API/server failure is changed to `failed` with `retriable=1` and is
retried on the next completion-status poll. A process crash while the
reservation itself is `ingesting` requires operator review before retry, because
automatically stealing that reservation could create a duplicate run.

Canonical runs are reconstructed only through supported MLflow APIs. They use
`viettoxic-kaggle-<model-family>` experiments and preserve these relationship
tags:

```text
execution_origin = kaggle
ingestion_mode = post_run
source_job_id
source_run_id
source_evidence_sha256
evidence_schema_version
source_training_status
source_tracking_status
source_artifact_status
```

No Kaggle or local MLflow SQLite tables are copied, merged, or edited. The
legacy repository-root `mlflow.db` remains immutable. The application model
registry and production slots remain authoritative for serving.

The Phase 2B.2A local scripts still contain pre-existing best-effort catches for
some mid-run logging calls. That local-only observability hardening remains a
separate follow-up; this Kaggle contract never converts incomplete source
tracking into `tracking_status=complete`.

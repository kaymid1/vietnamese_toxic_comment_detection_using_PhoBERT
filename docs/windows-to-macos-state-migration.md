# Windows to macOS state migration

Phase 2C.1 provides local, auditable migration tooling. It does not perform the real migration. Generated bundles are private, ignored migration material and must never be committed.

## Safety architecture

```text
real Windows state (read-only)
        |
        v
SQLite backup snapshot of feedback.db
        |
        v
portable-reference conversion on the snapshot only
        |
        v
versioned checksummed directory bundle
        |
        v
Mac import dry-run
        |
        v
validated target staging -> explicit promotion -> target verification
```

The exporter runs `PRAGMA integrity_check` before backup, after backup, and after portable-reference conversion. It records a logical SQLite digest, size, and mtime before and after export. If the running application changes `feedback.db` concurrently, the report identifies a valid point-in-time snapshot; it never restores or overwrites the live database.

Never run `python -m backend.portability_migration --apply` against the real `data/processed/feedback/feedback.db`. The bundle tool invokes the existing migration logic only with its staging snapshot path.

## Bundle contract (schema version 1)

The transfer format is a directory, avoiding a risky multi-gigabyte ZIP:

```text
viettoxic-state-bundle/
|-- manifest.json
|-- checksums.sha256
|-- application/
|   `-- feedback.db
|-- models/
|   `-- options/...
|-- persistent_artifacts/
|   |-- runtime/...
|   `-- data/...
|-- mlflow/
|   |-- active/                 # omitted when not initialized
|   |   |-- mlflow.db
|   |   `-- artifacts/...
|   `-- legacy_evidence/
|       |-- legacy_mlflow_inventory.json
|       |-- legacy_mlflow_artifacts.json
|       `-- legacy_mlflow_checksums.json
`-- metadata/
    |-- source_inventory.json
    `-- environment_requirements.json
```

`manifest.json` records schema/project identity, source platform and Git commit, creation time, timestamp-independent content identity, component status, Python/MLflow/schema compatibility, sensitivity flags, state counts, and per-file sizes and SHA-256 values. `checksums.sha256` covers the manifest and every payload/metadata file. All member identities are normalized relative POSIX paths.

Import treats a bundle as untrusted. It rejects unsupported schemas, malformed manifests, missing or unexpected files, duplicate members/destinations, size or SHA mismatch, absolute/drive/UNC/traversal paths, symlinks, corrupt SQLite, incompatible application tables, invalid model contracts, and unsafe MLflow references.

## Component policy

| Component | Export | Transform | Import | Reason |
| --- | --- | --- | --- | --- |
| `feedback.db` | SQLite backup API | Managed path fields converted on snapshot only | Integrity/schema/count verified | Preserves operational state without touching the live DB |
| `models/options` | Full tree | None; per-file checksum | Staged as one model-options component | Preserve every inference-ready model; no binary rewriting |
| Required managed runtime/data references | Referenced items only | Stored beneath logical scheme/path | Resolved beneath target roots | Preserve rollback/promotion/finetune dependencies without copying `.runtime` wholesale |
| Active MLflow | Optional safe DB snapshot plus artifact tree | No SQL rewriting | Installed at `APP_DATA_DIR/mlflow` | Preserve canonical runs only when no RUNNING run or machine path exists |
| Legacy MLflow | Three deterministic evidence JSON files | None | Evidence only | Never merges or activates the immutable root `mlflow.db` |
| `.env.local` and credentials | Excluded | Variable names/descriptions only | Operator recreates manually | Secret values must not enter the bundle |
| Browser state | Excluded | None | None | Not required for backend correctness |
| Logs/cache/build/temp/Kaggle workspaces | Excluded | None | None | Disposable runtime state, not application state |

## Persistent artifact classification

Path-bearing application fields are scanned after snapshot conversion.

- `runtime://model_registry/...` is `persistent-required` and is included when present.
- `runtime://kaggle_real_jobs/...` is ephemeral and is not copied automatically.
- Model references are covered by full `models/options` preservation.
- Required `data://...` artifacts are included, except state owned by the dedicated MLflow component.
- Missing required paths are reported as `MISSING_REFERENCED_ARTIFACT`.
- Unmanaged paths such as `D:\Downloads\...` or `/Users/...` are reported as `EXTERNAL_REFERENCE_REQUIRES_REVIEW` and are never copied automatically.

`.runtime/model_registry` is logically persistent despite its physical location. Moving it under `APP_DATA_DIR` remains technical debt and is deliberately outside Phase 2C.1.

## MLflow handling

The legacy repository-root `mlflow.db` remains immutable historical evidence. It is never merged with the active store.

If `APP_DATA_DIR/mlflow/mlflow.db` does not exist, the manifest records `not_initialized`. If it exists, export refuses while any run is `RUNNING` or when experiments/runs contain machine-specific `file://`, drive-letter, or `/Users/...` artifact metadata. A safe store is snapshotted with SQLite backup, its artifact tree is copied and checksummed, and both source DB and artifact-tree signatures are checked again before bundle activation.

## Commands available in Phase 2C.1

Read-only source inventory (does not hash/copy the 7.5 GB model tree):

```powershell
python -m backend.state_bundle export --dry-run
```

Inspect or verify an existing disposable bundle:

```powershell
python -m backend.state_bundle inspect <bundle>
python -m backend.state_bundle verify <bundle>
```

Import dry-run is the default and performs zero target writes:

```bash
python -m backend.state_bundle import <bundle> \
  --target-data-dir /Users/<operator>/VietToxicData \
  --target-runtime-dir /Users/<operator>/VietToxicRuntime \
  --target-model-options-dir /Users/<operator>/VietToxicModels \
  --dry-run
```

The report shows source identity, component/checksum status, target paths, collisions, database/model/MLflow status, missing environment requirements, planned actions, and the warning when imported automation is enabled. No service is started.

## Collision, backup, and rollback behavior

Apply refuses any existing target component with `TARGET_STATE_EXISTS`. There is no merge behavior. Explicit replacement requires `--replace-existing`; existing SQLite is backed up through the SQLite backup API and directories/files are copied to a separate backup root before promotion.

All incoming components are copied to target-side staging and validated first. Promotion records each original component in rollback staging. If a later promotion or target verification fails, newly promoted state is removed and original components are restored in reverse order. Successful replacement retains the durable backup for operator recovery.

Keep backend, frontend, MLflow, webhook, and automation services stopped during import. The importer never starts them and never changes imported automation settings.

## Future Phase 2C.2 procedure — do not execute in Phase 2C.1

1. Stop all Windows writers and record final integrity fingerprints.
2. Create the final private directory bundle:

   ```powershell
   python -m backend.state_bundle export --output state_exports\viettoxic-final
   python -m backend.state_bundle verify state_exports\viettoxic-final
   ```

3. Transfer the directory without modifying its contents, preserving all files.
4. On macOS, inspect and verify before any import:

   ```bash
   python -m backend.state_bundle inspect /path/to/viettoxic-final
   python -m backend.state_bundle verify /path/to/viettoxic-final
   ```

5. Configure target roots, then perform import dry-run:

   ```bash
   export APP_DATA_DIR=/Users/<operator>/VietToxicData
   export APP_RUNTIME_DIR=/Users/<operator>/VietToxicRuntime
   export VIETTOXIC_MODEL_OPTIONS_DIR=/Users/<operator>/VietToxicModels
   python -m backend.state_bundle import /path/to/viettoxic-final --dry-run
   ```

6. Resolve every collision, missing requirement, external reference, and automation warning. Recreate `.env.local` files manually; never copy secret values through the bundle.
7. With services still stopped, apply only after reviewing the dry-run:

   ```bash
   python -m backend.state_bundle import /path/to/viettoxic-final --apply
   python -m backend.state_bundle verify-target /path/to/viettoxic-final
   ```

8. Start the MLflow server explicitly if the imported manifest contains active MLflow. Verify its historical run, metric, and artifact access.
9. Start the backend, verify model selection/production slots/review/training state, then start the frontend. Review automation configuration before enabling any watcher, webhook, or Kaggle workflow.

The operator must choose explicit backup/replacement flags if target state already exists. Do not use `--replace-existing` without reviewing and retaining the reported backup.

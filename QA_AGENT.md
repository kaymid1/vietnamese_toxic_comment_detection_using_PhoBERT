# QA Agent

## Purpose
QA Agent is a pragmatic pre-merge/pre-deploy gate. It does not replace manual developer testing; it enforces a minimum automated check set based on affected code areas.

## Responsibilities
1. Detect changed areas from git diff / changed file list.
2. Select matching smoke/regression checks.
3. Run required checks in deterministic order.
4. Report concise results in a fixed format.

## Affected-Area Mapping (V1)
- `backend/app.py`, `backend/**`, `infer_crawled_local.py`, `comment_crawl.py`
  - Run backend smoke/regression pytest suite.
- `comprehensive_ui/src/**`
  - Run frontend production build.
  - Run backend API contract checks for MLflow store payloads.
- `kaggle/**`, `scripts/publish_kaggle_*`, MLflow-related backend paths
  - Run Kaggle/MLflow smoke tests (dry-run, status, artifact checks).
- `docker-compose.yml`, `backend/Dockerfile`, `comprehensive_ui/Dockerfile`, `requirements*.txt`
  - Run dependency/import/build sanity checks.

## Standard QA Command
- PowerShell: `./scripts/qa_check.ps1`
- Bash: `./scripts/qa_check.sh`

Optional mode:
- `--only-affected` / `-OnlyAffected` to run only checks mapped to changed files (still runs Python compile sanity first).

## Baseline Check Sequence
1. `python -m py_compile backend/app.py`
2. Backend pytest smoke/regression (`tests/test_backend_smoke.py`, `tests/test_mlflow_kaggle.py`, `tests/test_api_contract_mlflow_store.py`)
3. Frontend production build (`npm run build` in `comprehensive_ui`)
4. Targeted checks by affected-area mapping (when only-affected mode is enabled)

## Pass / Fail Criteria
- **Pass**: all required checks for the selected scope complete with exit code `0`.
- **Fail**: any required check fails or exits non-zero.
- **Skipped**: checks not required by mapping or explicitly skipped by mode/flags.

## Test Safety Rules
- Tests must not mutate real project runtime data.
- Backend tests must use temporary DB/runtime/data folders.
- Kaggle tests must use local simulated artifacts for normal QA.
- External cloud credentials are never required for V1 QA pass.

## QA Report Format (Fixed)
Use this exact section order after each QA run:

1. `Changed Areas`
2. `Tests Run`
3. `Pass/Fail`
4. `Skipped Checks`
5. `Residual Risks`

Example skeleton:

```text
Changed Areas:
- backend/app.py
- comprehensive_ui/src/hooks/useMlflowStore.ts

Tests Run:
- python -m py_compile backend/app.py
- pytest tests/test_backend_smoke.py tests/test_mlflow_kaggle.py tests/test_api_contract_mlflow_store.py -q
- npm run build (comprehensive_ui)

Pass/Fail:
- PASS: py_compile
- PASS: pytest
- PASS: frontend build

Skipped Checks:
- none

Residual Risks:
- No browser E2E in V1 (Playwright planned for V2)
- Real Kaggle cloud execution not covered by default QA
```

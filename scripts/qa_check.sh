#!/usr/bin/env bash
set -euo pipefail

ONLY_AFFECTED=0
SKIP_FRONTEND=0
SKIP_BACKEND_TESTS=0

for arg in "$@"; do
  case "$arg" in
    --only-affected) ONLY_AFFECTED=1 ;;
    --skip-frontend) SKIP_FRONTEND=1 ;;
    --skip-backend-tests) SKIP_BACKEND_TESTS=1 ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

run_step() {
  local label="$1"
  shift
  echo "==> ${label}"
  "$@"
}

PYTHON_BIN="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
fi

get_changed_files() {
  local diff_files
  diff_files="$(git diff --name-only --diff-filter=ACMRTUXB HEAD~1 HEAD 2>/dev/null || true)"
  if [[ -n "${diff_files}" ]]; then
    printf '%s\n' "${diff_files}"
    return 0
  fi
  git status --porcelain 2>/dev/null | awk '{print substr($0,4)}' || true
}

has_pattern_match() {
  local patterns=("$@")
  while IFS= read -r file; do
    [[ -z "${file}" ]] && continue
    for pattern in "${patterns[@]}"; do
      if [[ "${file}" =~ ${pattern} ]]; then
        return 0
      fi
    done
  done < "${CHANGED_FILE_TMP}"
  return 1
}

CHANGED_FILE_TMP="$(mktemp)"
trap 'rm -f "${CHANGED_FILE_TMP}"' EXIT
get_changed_files > "${CHANGED_FILE_TMP}"

backend_patterns=('^backend/' '^infer_crawled_local\.py$' '^comment_crawl\.py$')
frontend_patterns=('^comprehensive_ui/src/')
kaggle_patterns=('^kaggle/' '^scripts/publish_kaggle_' '^backend/.*mlflow' '^backend/app\.py$')
deps_patterns=('^docker-compose\.yml$' '^backend/Dockerfile$' '^comprehensive_ui/Dockerfile$' '^requirements.*\.txt$' '^requirements-base\.txt$' '^requirements-ml\.txt$')

has_backend=0
has_frontend=0
has_kaggle=0
has_deps=0

has_pattern_match "${backend_patterns[@]}" && has_backend=1 || true
has_pattern_match "${frontend_patterns[@]}" && has_frontend=1 || true
has_pattern_match "${kaggle_patterns[@]}" && has_kaggle=1 || true
has_pattern_match "${deps_patterns[@]}" && has_deps=1 || true

run_backend_tests=1
run_frontend_build=1

if [[ "${SKIP_BACKEND_TESTS}" -eq 1 ]]; then
  run_backend_tests=0
elif [[ "${ONLY_AFFECTED}" -eq 1 && "${has_backend}" -eq 0 && "${has_kaggle}" -eq 0 && "${has_deps}" -eq 0 ]]; then
  run_backend_tests=0
fi

if [[ "${SKIP_FRONTEND}" -eq 1 ]]; then
  run_frontend_build=0
elif [[ "${ONLY_AFFECTED}" -eq 1 && "${has_frontend}" -eq 0 && "${has_deps}" -eq 0 ]]; then
  run_frontend_build=0
fi

run_step "Python compile sanity" "${PYTHON_BIN}" -m py_compile backend/app.py

if [[ "${run_backend_tests}" -eq 1 ]]; then
  run_step \
    "Backend QA pytest" \
    "${PYTHON_BIN}" -m pytest -q \
      tests/test_backend_smoke.py \
      tests/test_mlflow_kaggle.py \
      tests/test_api_contract_mlflow_store.py \
      tests/test_frontend_i18n_encoding.py
else
  echo "==> Skip backend pytest (not affected)"
fi

if [[ "${run_frontend_build}" -eq 1 ]]; then
  pushd comprehensive_ui >/dev/null
  run_step "Frontend production build" npm run build
  popd >/dev/null
else
  echo "==> Skip frontend build (not affected)"
fi

echo "QA checks completed."

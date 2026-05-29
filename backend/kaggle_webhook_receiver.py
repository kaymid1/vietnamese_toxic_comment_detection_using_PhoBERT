import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.system_settings import (
    DEFAULT_SETTINGS_DB_PATH,
    get_bool_setting as get_runtime_bool_setting,
    get_int_setting as get_runtime_int_setting,
    get_setting as get_runtime_setting,
)


BASE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = BASE_DIR / ".runtime"
STATE_PATH = RUNTIME_DIR / "kaggle_webhook_jobs.json"

# Load local env files so running this service standalone still picks up backend config.
load_dotenv(BASE_DIR / "backend" / ".env.local", override=False)
load_dotenv(BASE_DIR / ".env.local", override=False)

WEBHOOK_MODE = os.getenv("KAGGLE_WEBHOOK_MODE", "mock").strip().lower()
AUTO_COMPLETE_SECONDS = max(0, int(os.getenv("KAGGLE_WEBHOOK_AUTO_COMPLETE_SECONDS", "90")))
DEFAULT_ARTIFACT_URI = os.getenv("KAGGLE_WEBHOOK_DEFAULT_ARTIFACT_URI", "").strip()
DEFAULT_ARTIFACT_CHECKSUM = os.getenv("KAGGLE_WEBHOOK_DEFAULT_ARTIFACT_CHECKSUM", "").strip()

REAL_KERNEL_OWNER = os.getenv("KAGGLE_KERNEL_OWNER", "").strip()
REAL_KERNEL_SLUG = os.getenv("KAGGLE_KERNEL_SLUG", "").strip()
REAL_KERNEL_TITLE = os.getenv("KAGGLE_KERNEL_TITLE", "VietComment Analyzer MLflow Retrain").strip()
REAL_ACCELERATOR = os.getenv("KAGGLE_KERNEL_ACCELERATOR", "NvidiaTeslaT4").strip()
REAL_PRIVATE = os.getenv("KAGGLE_KERNEL_PRIVATE", "true").strip().lower() in {"1", "true", "yes", "on"}
REAL_PUSH_TIMEOUT_SEC = max(30, int(os.getenv("KAGGLE_KERNEL_PUSH_TIMEOUT_SEC", "600")))
REAL_STATUS_TIMEOUT_SEC = max(10, int(os.getenv("KAGGLE_KERNEL_STATUS_TIMEOUT_SEC", "60")))
REAL_OUTPUT_TIMEOUT_SEC = max(30, int(os.getenv("KAGGLE_KERNEL_OUTPUT_TIMEOUT_SEC", "900")))
REAL_PULL_TIMEOUT_SEC = max(15, int(os.getenv("KAGGLE_KERNEL_PULL_TIMEOUT_SEC", "120")))
REAL_OUTPUT_PATTERN = os.getenv(
    "KAGGLE_REAL_ARTIFACT_FILE_PATTERN",
    r".*\.(zip|tar\.gz|tgz|joblib|bin|pt)$",
).strip()
REAL_ARTIFACT_URI_TEMPLATE = os.getenv("KAGGLE_REAL_ARTIFACT_URI_TEMPLATE", "").strip()
REAL_DATASET_SOURCES = os.getenv("KAGGLE_KERNEL_DATASET_SOURCES", "").strip()
REAL_PUSH_RETRY_ATTEMPTS = max(1, int(os.getenv("KAGGLE_PUSH_RETRY_ATTEMPTS", "4")))
REAL_PUSH_RETRY_DELAY_SEC = max(1, int(os.getenv("KAGGLE_PUSH_RETRY_DELAY_SEC", "5")))
REAL_BUNDLE_URL = os.getenv("KAGGLE_REAL_BUNDLE_URL", "").strip()
REAL_BUNDLE_URL_TEMPLATE = os.getenv("KAGGLE_REAL_BUNDLE_URL_TEMPLATE", "").strip()
REAL_TEST_MODE = os.getenv("KAGGLE_REAL_TEST_MODE", "smoke").strip() or "smoke"
REAL_IMPORT_API_URL = os.getenv("KAGGLE_REAL_IMPORT_API_URL", "").strip()
REAL_IMPORT_API_TOKEN = os.getenv("KAGGLE_REAL_IMPORT_API_TOKEN", "").strip()
REAL_IMPORT_ARTIFACT_PATH = os.getenv("KAGGLE_REAL_IMPORT_ARTIFACT_PATH", "").strip()
REAL_IMPORT_NOTES = os.getenv("KAGGLE_REAL_IMPORT_NOTES", "Kaggle real run").strip()
REAL_IMPORT_REQUIRED = os.getenv("KAGGLE_REAL_IMPORT_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
REAL_NOTEBOOK_SOURCE = Path(
    os.getenv(
        "KAGGLE_REAL_NOTEBOOK_SOURCE",
        str(BASE_DIR / "kaggle" / "notebooks" / "mlflow_retrain" / "viettoxic_mlflow_retrain.py"),
    ).strip()
)

_LOCK = threading.Lock()

app = FastAPI(title="VietComment Analyzer Kaggle Webhook Receiver")


def _setting(key: str, default: str = "") -> str:
    return str(get_runtime_setting(key, default, db_path=DEFAULT_SETTINGS_DB_PATH) or "")


def _int_setting(key: str, default: int, min_value: Optional[int] = None) -> int:
    return get_runtime_int_setting(key, default, db_path=DEFAULT_SETTINGS_DB_PATH, min_value=min_value)


def _bool_setting(key: str, default: bool = False) -> bool:
    return get_runtime_bool_setting(key, default, db_path=DEFAULT_SETTINGS_DB_PATH)


def _webhook_mode() -> str:
    mode = _setting("KAGGLE_WEBHOOK_MODE", WEBHOOK_MODE).strip().lower()
    return mode if mode in {"mock", "real"} else "mock"


class TriggerRequest(BaseModel):
    run_id: str
    batch_id: Optional[str] = None
    model_kind: Optional[str] = None
    training_mode: Optional[str] = None
    base_model: Optional[str] = None
    requested_at: Optional[str] = None
    notebook_url: Optional[str] = None


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    _ensure_runtime_dir()
    if not STATE_PATH.exists():
        return {"jobs": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            jobs = data.get("jobs")
            if isinstance(jobs, dict):
                return {"jobs": jobs}
    except Exception:
        pass
    return {"jobs": {}}


def _save_state(state: Dict[str, Any]) -> None:
    _ensure_runtime_dir()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _status_for_mock_job(job: Dict[str, Any]) -> str:
    raw = str(job.get("status") or "running").strip().lower()
    if raw in {"queued", "running", "completed", "failed"}:
        status = raw
    else:
        status = "running"

    if status in {"completed", "failed"}:
        return status

    created_at_ts = float(job.get("created_at_ts") or time.time())
    if AUTO_COMPLETE_SECONDS > 0 and (time.time() - created_at_ts) >= AUTO_COMPLETE_SECONDS:
        return "completed"
    return status


def _build_subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    kaggle_username = _setting("KAGGLE_USERNAME", "").strip()
    kaggle_key = _setting("KAGGLE_KEY", "").strip()
    if kaggle_username:
        env["KAGGLE_USERNAME"] = kaggle_username
    if kaggle_key:
        env["KAGGLE_KEY"] = kaggle_key
    # Force UTF-8 for Python-based CLIs (including kaggle) to avoid Windows cp1252/charmap failures.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run_cmd(cmd: list[str], *, cwd: Optional[Path] = None, timeout: Optional[int] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_build_subprocess_env(),
        timeout=timeout,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = err or out or f"exit_code={proc.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(cmd)} | {detail}")
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_bundle_url(batch_id: Optional[str], run_id: str) -> str:
    if REAL_BUNDLE_URL_TEMPLATE:
        try:
            return REAL_BUNDLE_URL_TEMPLATE.format(batch_id=batch_id or "", run_id=run_id).strip()
        except Exception:
            return REAL_BUNDLE_URL
    if "{" in REAL_BUNDLE_URL and "}" in REAL_BUNDLE_URL:
        try:
            return REAL_BUNDLE_URL.format(batch_id=batch_id or "", run_id=run_id).strip()
        except Exception:
            return REAL_BUNDLE_URL
    return REAL_BUNDLE_URL


def _resolve_owner_slug(notebook_url: Optional[str]) -> tuple[str, str]:
    owner = _setting("KAGGLE_KERNEL_OWNER", REAL_KERNEL_OWNER).strip()
    slug = _setting("KAGGLE_KERNEL_SLUG", REAL_KERNEL_SLUG).strip()
    if owner and slug:
        return owner, slug

    url = (notebook_url or "").strip()
    if url:
        m = re.search(r"/code/([^/]+)/([^/?#]+)", url)
        if m:
            owner = owner or m.group(1).strip()
            slug = slug or m.group(2).strip()

    if not owner or not slug:
        raise RuntimeError(
            "Kaggle kernel owner/slug is missing. Set KAGGLE_KERNEL_OWNER + KAGGLE_KERNEL_SLUG or pass notebook_url with /code/<owner>/<slug>."
        )
    return owner, slug


def _slugify_title(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s


def _build_real_script_content(payload: TriggerRequest) -> str:
    notebook_source = Path(_setting("KAGGLE_REAL_NOTEBOOK_SOURCE", str(REAL_NOTEBOOK_SOURCE)).strip())
    if not notebook_source.exists():
        raise RuntimeError(f"Notebook source not found: {notebook_source}")

    requested_model_kind = (payload.model_kind or "phobert").strip().lower()
    configured_test_mode = _setting("KAGGLE_REAL_TEST_MODE", REAL_TEST_MODE).strip() or "smoke"
    resolved_test_mode = "smoke" if requested_model_kind == "lr_smoke" else "phobert"
    if configured_test_mode in {"validate", "smoke"} and requested_model_kind == "lr_smoke":
        resolved_test_mode = configured_test_mode
    source_text = notebook_source.read_text(encoding="utf-8-sig").replace("\ufeff", "").replace("ï»¿", "")
    env_overrides = {
        "VIETTOXIC_TEST_MODE": resolved_test_mode,
        "VIETTOXIC_BUNDLE_URL": _resolve_bundle_url(payload.batch_id, payload.run_id),
        "VIETTOXIC_RUN_NAME": payload.run_id,
        "VIETTOXIC_IMPORT_API_URL": REAL_IMPORT_API_URL,
        "VIETTOXIC_IMPORT_API_TOKEN": REAL_IMPORT_API_TOKEN,
        "VIETTOXIC_IMPORT_ARTIFACT_PATH": REAL_IMPORT_ARTIFACT_PATH,
        "VIETTOXIC_IMPORT_NOTES": REAL_IMPORT_NOTES,
        "VIETTOXIC_IMPORT_REQUIRED": "true" if REAL_IMPORT_REQUIRED else "false",
        "VIETTOXIC_MODEL_KIND": requested_model_kind,
        "VIETTOXIC_TRAINING_MODE": payload.training_mode or "",
        "VIETTOXIC_BASE_MODEL": payload.base_model or "",
    }
    preamble = (
        "# Auto-generated by kaggle_webhook_receiver.py\n"
        "import os\n"
        f"_env = {json.dumps(env_overrides, ensure_ascii=False)}\n"
        "for _k, _v in _env.items():\n"
        "    if _v is not None and str(_v).strip() != '':\n"
        "        os.environ[_k] = str(_v)\n\n"
    )
    return preamble + source_text


def _script_to_ipynb(script_text: str) -> str:
    lines = script_text.splitlines(keepends=True)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=2)


def _attach_phobert_train_script(job_dir: Path) -> Optional[Path]:
    source = BASE_DIR / "scripts" / "06_train_phobert_lora_macro_f1_finetune.py"
    if not source.exists():
        return None
    target = job_dir / "scripts" / "train_phobert.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _build_kernel_metadata(
    owner: str,
    slug: str,
    code_file: str,
    *,
    title: Optional[str] = None,
    kernel_type: str = "script",
    dataset_sources: Optional[list[str]] = None,
    competition_sources: Optional[list[str]] = None,
    kernel_sources: Optional[list[str]] = None,
    model_sources: Optional[list[str]] = None,
) -> Dict[str, Any]:
    kernel_title = _setting("KAGGLE_KERNEL_TITLE", REAL_KERNEL_TITLE).strip()
    accelerator = _setting("KAGGLE_KERNEL_ACCELERATOR", REAL_ACCELERATOR).strip()
    private = _bool_setting("KAGGLE_KERNEL_PRIVATE", REAL_PRIVATE)
    metadata: Dict[str, Any] = {
        "id": f"{owner}/{slug}",
        "title": title or kernel_title or slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": kernel_type,
        "is_private": private,
        "enable_gpu": accelerator.lower() != "none",
        "enable_internet": True,
        "dataset_sources": dataset_sources or [],
        "competition_sources": competition_sources or [],
        "kernel_sources": kernel_sources or [],
        "model_sources": model_sources or [],
    }
    if accelerator and accelerator.lower() != "none":
        metadata["accelerator"] = accelerator
    return metadata


def _parse_csv_list(value: str) -> list[str]:
    items = [part.strip() for part in (value or "").split(",")]
    return [item for item in items if item]


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _fetch_existing_kernel_metadata(kernel_ref: str, job_dir: Path) -> Optional[Dict[str, Any]]:
    pull_dir = job_dir / "_kernel_pull"
    pull_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_cmd(
            ["kaggle", "kernels", "pull", kernel_ref, "-p", str(pull_dir), "-m", "-q"],
            timeout=REAL_PULL_TIMEOUT_SEC,
        )
    except Exception:
        return None

    meta_path = pull_dir / "kernel-metadata.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _normalize_kernel_status(raw: str) -> str:
    s = (raw or "").strip().lower()
    if any(token in s for token in ("error", "failed", "cancelled", "canceled")):
        return "failed"
    if any(token in s for token in ("complete", "completed", "success", "succeeded")):
        return "completed"
    if any(token in s for token in ("queued", "pending", "running", "committing")):
        return "running"
    return "running"


def _extract_push_error(push_output: str) -> Optional[str]:
    text = (push_output or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if "kernel push error" in lowered:
        for line in reversed(text.splitlines()):
            if "kernel push error" in line.lower():
                return line.strip()
        return "Kernel push failed."
    if "notebook not found" in lowered:
        return "Kernel push failed: Notebook not found."
    if "404" in lowered and "not found" in lowered:
        return "Kernel push failed: Kaggle returned 404 Not Found."
    return None


def _is_editor_type_change_error(push_error: Optional[str]) -> bool:
    text = (push_error or "").strip().lower()
    return "cannot change the editor type" in text


def _is_kaggle_push_conflict(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return (
        ("409" in lowered and "conflict" in lowered)
        or ("savekernel" in lowered and "conflict" in lowered)
        or ("another version of this kernel is currently being created" in lowered)
    )


def _run_kaggle_push_with_retry(job_dir: Path) -> str:
    cmd = ["kaggle", "kernels", "push", "-p", str(job_dir)]
    logs: list[str] = []
    retry_attempts = _int_setting("KAGGLE_PUSH_RETRY_ATTEMPTS", REAL_PUSH_RETRY_ATTEMPTS, min_value=1)
    retry_delay_sec = _int_setting("KAGGLE_PUSH_RETRY_DELAY_SEC", REAL_PUSH_RETRY_DELAY_SEC, min_value=1)

    for attempt in range(1, retry_attempts + 1):
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_build_subprocess_env(),
            timeout=REAL_PUSH_TIMEOUT_SEC,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        combined = "\n".join(part for part in (out, err) if part).strip()
        logs.append(f"[push-try:{attempt}] {combined or '(no output)'}")

        if proc.returncode == 0:
            return "\n\n".join(logs)

        # Let caller parse known logical push errors (e.g., editor type mismatch).
        if _extract_push_error(combined):
            return "\n\n".join(logs)

        if _is_kaggle_push_conflict(combined) and attempt < retry_attempts:
            time.sleep(retry_delay_sec * attempt)
            continue

        detail = combined or f"exit_code={proc.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(cmd)} | {detail}")

    raise RuntimeError(f"Command failed after retries: {' '.join(cmd)}")


def _pick_artifact_file(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists() or not output_dir.is_dir():
        return None
    candidates = [p for p in output_dir.rglob("*") if p.is_file()]
    if not candidates:
        return None

    pattern = None
    try:
        if REAL_OUTPUT_PATTERN:
            pattern = re.compile(REAL_OUTPUT_PATTERN, re.IGNORECASE)
    except re.error:
        pattern = None

    if pattern is not None:
        filtered = [p for p in candidates if pattern.match(p.name)]
        if filtered:
            candidates = filtered

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_artifact_uri(path: Path, job: Dict[str, Any]) -> str:
    if REAL_ARTIFACT_URI_TEMPLATE:
        try:
            return REAL_ARTIFACT_URI_TEMPLATE.format(
                job_id=job.get("job_id") or "",
                run_id=job.get("run_id") or "",
                file_name=path.name,
                file_path=str(path.resolve()),
            )
        except Exception:
            pass
    return f"file://{path.resolve().as_posix()}"


def _trigger_mock(payload: TriggerRequest) -> Dict[str, Any]:
    job_id = f"mock_{uuid.uuid4().hex[:12]}"
    now = time.time()

    job = {
        "mode": "mock",
        "job_id": job_id,
        "run_id": payload.run_id,
        "batch_id": payload.batch_id,
        "training_mode": payload.training_mode,
        "model_kind": payload.model_kind,
        "base_model": payload.base_model,
        "notebook_url": payload.notebook_url,
        "requested_at": payload.requested_at,
        "created_at_ts": now,
        "status": "running",
        "artifact_uri": DEFAULT_ARTIFACT_URI or None,
        "artifact_checksum": DEFAULT_ARTIFACT_CHECKSUM or None,
        "error_message": None,
    }

    with _LOCK:
        state = _load_state()
        state["jobs"][job_id] = job
        _save_state(state)

    return {
        "accepted": True,
        "job_id": job_id,
        "status": "running",
        "message": "Mock Kaggle job accepted",
    }


def _trigger_real(payload: TriggerRequest) -> Dict[str, Any]:
    if shutil.which("kaggle") is None:
        raise HTTPException(status_code=500, detail="kaggle CLI not found in PATH")

    owner, slug = _resolve_owner_slug(payload.notebook_url)
    kernel_ref = f"{owner}/{slug}"
    job_id = f"real_{uuid.uuid4().hex[:12]}"
    job_dir = RUNTIME_DIR / "kaggle_real_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    kernel_meta = _fetch_existing_kernel_metadata(kernel_ref, job_dir)
    existing_kernel_type = str((kernel_meta or {}).get("kernel_type") or "").strip().lower()
    existing_code_file = str((kernel_meta or {}).get("code_file") or "").strip()
    existing_title = str((kernel_meta or {}).get("title") or "").strip()
    existing_dataset_sources = _coerce_string_list((kernel_meta or {}).get("dataset_sources"))
    existing_competition_sources = _coerce_string_list((kernel_meta or {}).get("competition_sources"))
    existing_kernel_sources = _coerce_string_list((kernel_meta or {}).get("kernel_sources"))
    existing_model_sources = _coerce_string_list((kernel_meta or {}).get("model_sources"))

    env_dataset_sources = _parse_csv_list(_setting("KAGGLE_KERNEL_DATASET_SOURCES", REAL_DATASET_SOURCES))
    dataset_sources = env_dataset_sources or existing_dataset_sources
    metadata_title = existing_title or _setting("KAGGLE_KERNEL_TITLE", REAL_KERNEL_TITLE).strip() or slug
    if _slugify_title(metadata_title) != slug:
        metadata_title = slug

    script_content = _build_real_script_content(payload)
    if (payload.model_kind or "phobert").strip().lower() == "phobert":
        _attach_phobert_train_script(job_dir)
    prefer_notebook = existing_kernel_type == "notebook" or existing_code_file.lower().endswith(".ipynb")

    def _write_kernel_payload(as_notebook: bool) -> tuple[str, str]:
        if as_notebook:
            code_file_local = existing_code_file if existing_code_file.lower().endswith(".ipynb") else f"{slug}.ipynb"
            code_content_local = _script_to_ipynb(script_content)
            kernel_type_local = "notebook"
        else:
            code_file_local = (
                existing_code_file
                if existing_code_file and not existing_code_file.lower().endswith(".ipynb")
                else "viettoxic_mlflow_retrain.py"
            )
            code_content_local = script_content
            kernel_type_local = "script"

        code_path_local = job_dir / code_file_local
        code_path_local.write_text(code_content_local, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                _build_kernel_metadata(
                    owner,
                    slug,
                    code_file_local,
                    title=metadata_title,
                    kernel_type=kernel_type_local,
                    dataset_sources=dataset_sources,
                    competition_sources=existing_competition_sources,
                    kernel_sources=existing_kernel_sources,
                    model_sources=existing_model_sources,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return code_file_local, kernel_type_local

    metadata_path = job_dir / "kernel-metadata.json"
    first_code_file, first_kernel_type = _write_kernel_payload(prefer_notebook)
    push_logs: list[str] = []

    first_push_stdout = ""
    try:
        first_push_stdout = _run_kaggle_push_with_retry(job_dir)
        push_error = _extract_push_error(first_push_stdout)
    except Exception as exc:
        first_push_stdout = str(exc)
        push_error = _extract_push_error(first_push_stdout) or first_push_stdout
    push_logs.append(f"[attempt:{first_kernel_type}] {first_push_stdout}")
    final_code_file = first_code_file
    final_kernel_type = first_kernel_type

    if _is_editor_type_change_error(push_error):
        retry_notebook = not prefer_notebook
        retry_code_file, retry_kernel_type = _write_kernel_payload(retry_notebook)
        retry_push_stdout = ""
        try:
            retry_push_stdout = _run_kaggle_push_with_retry(job_dir)
            push_error = _extract_push_error(retry_push_stdout)
        except Exception as exc:
            retry_push_stdout = str(exc)
            push_error = _extract_push_error(retry_push_stdout) or retry_push_stdout
        push_logs.append(f"[attempt:{retry_kernel_type}] {retry_push_stdout}")
        final_code_file = retry_code_file
        final_kernel_type = retry_kernel_type

    push_stdout = "\n\n".join(push_logs)

    now = time.time()
    job = {
        "mode": "real",
        "job_id": job_id,
        "run_id": payload.run_id,
        "batch_id": payload.batch_id,
        "training_mode": payload.training_mode,
        "model_kind": payload.model_kind,
        "base_model": payload.base_model,
        "notebook_url": payload.notebook_url,
        "requested_at": payload.requested_at,
        "created_at_ts": now,
        "status": "running",
        "artifact_uri": None,
        "artifact_checksum": None,
        "error_message": None,
        "owner": owner,
        "slug": slug,
        "kernel_ref": kernel_ref,
        "work_dir": str(job_dir),
        "code_file": final_code_file,
        "kernel_type": final_kernel_type,
        "push_error": push_error,
        "push_stdout": push_stdout[-4000:],
    }
    if push_error:
        job["status"] = "failed"
        job["error_message"] = push_error

    with _LOCK:
        state = _load_state()
        state["jobs"][job_id] = job
        _save_state(state)

    if push_error:
        return {
            "accepted": True,
            "job_id": job_id,
            "status": "failed",
            "message": f"Real Kaggle job failed at push: {push_error}",
        }

    return {
        "accepted": True,
        "job_id": job_id,
        "status": "running",
        "message": f"Real Kaggle job submitted via kernels push ({kernel_ref})",
    }


def _status_mock(job_id: str, job: Dict[str, Any], jobs: Dict[str, Any]) -> Dict[str, Any]:
    current_status = _status_for_mock_job(job)
    if current_status != job.get("status"):
        job["status"] = current_status
        if current_status == "completed":
            if not job.get("artifact_uri"):
                job["artifact_uri"] = f"mock://kaggle-artifacts/{job.get('run_id') or job_id}/model.zip"
            if not job.get("artifact_checksum"):
                job["artifact_checksum"] = uuid.uuid4().hex
        jobs[job_id] = job

    return {
        "job_id": job_id,
        "run_id": job.get("run_id"),
        "status": job.get("status") or "running",
        "artifact_uri": job.get("artifact_uri"),
        "artifact_checksum": job.get("artifact_checksum"),
        "error_message": job.get("error_message"),
    }


def _status_real(job_id: str, job: Dict[str, Any], jobs: Dict[str, Any]) -> Dict[str, Any]:
    status = str(job.get("status") or "running").strip().lower()
    kernel_ref = str(job.get("kernel_ref") or "").strip()
    push_error = str(job.get("push_error") or "").strip()
    if not push_error and "push_error" not in job:
        # Backward compatibility for jobs created before push_error field existed.
        push_error = _extract_push_error(str(job.get("push_stdout") or "")) or ""

    if not kernel_ref:
        job["status"] = "failed"
        job["error_message"] = "Missing kernel_ref in job state."
        jobs[job_id] = job
    elif push_error:
        job["status"] = "failed"
        job["error_message"] = push_error
        jobs[job_id] = job
    elif status not in {"completed", "failed"}:
        try:
            status_stdout = _run_cmd(
                ["kaggle", "kernels", "status", kernel_ref],
                timeout=REAL_STATUS_TIMEOUT_SEC,
            )
            normalized = _normalize_kernel_status(status_stdout)
            job["last_status_stdout"] = status_stdout[-4000:]
            job["status"] = normalized
            if normalized == "failed":
                job["error_message"] = f"Kaggle run failed ({kernel_ref}). status={status_stdout.strip()}"
            if normalized == "completed" and not job.get("artifact_uri"):
                work_dir = Path(str(job.get("work_dir") or RUNTIME_DIR))
                output_dir = work_dir / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                _run_cmd(
                    ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir), "-o", "-q"],
                    timeout=REAL_OUTPUT_TIMEOUT_SEC,
                )
                artifact_file = _pick_artifact_file(output_dir)
                if artifact_file is None:
                    job["error_message"] = f"Kaggle completed but no output artifact found in {output_dir}"
                    job["status"] = "failed"
                else:
                    job["artifact_checksum"] = _sha256_file(artifact_file)
                    job["artifact_uri"] = _resolve_artifact_uri(artifact_file, job)
        except Exception as exc:
            job["status"] = "failed"
            job["error_message"] = str(exc)
        jobs[job_id] = job

    return {
        "job_id": job_id,
        "run_id": job.get("run_id"),
        "status": job.get("status") or "running",
        "artifact_uri": job.get("artifact_uri"),
        "artifact_checksum": job.get("artifact_checksum"),
        "error_message": job.get("error_message"),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "mode": _webhook_mode()}


@app.post("/kaggle/trigger")
def kaggle_trigger(payload: TriggerRequest) -> Dict[str, Any]:
    mode = _webhook_mode()
    if mode == "real":
        return _trigger_real(payload)
    return _trigger_mock(payload)


@app.get("/kaggle/status")
def kaggle_status(job_id: str = Query(..., min_length=1)) -> Dict[str, Any]:
    with _LOCK:
        state = _load_state()
        jobs = state.get("jobs", {})
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            return {"job_id": job_id, "status": "failed", "error_message": "job_id not found"}

        mode = str(job.get("mode") or "mock").strip().lower()
        if mode == "real":
            payload = _status_real(job_id, job, jobs)
        else:
            payload = _status_mock(job_id, job, jobs)

        state["jobs"] = jobs
        _save_state(state)
        return payload

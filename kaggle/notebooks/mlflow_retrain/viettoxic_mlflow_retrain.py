# %% [markdown]
# # VietComment Analyzer MLflow Retrain (Mirror)
#
# File nay duoc version trong repo de maintain de hon.
# Khi can cap nhat notebook Kaggle:
# 1) Sua file nay
# 2) Publish len Kaggle bang script `scripts/publish_kaggle_kernel.ps1`
#
# Luu y:
# - Day la mirror source cho Kaggle Kernel.
# - Ban co the dung script `.py` hoac copy cell vao notebook UI tren Kaggle.

# %%
import json
import os
import pathlib
import subprocess
import sys
import time
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

WORKDIR = pathlib.Path("/kaggle/working/viettoxic")
WORKDIR.mkdir(parents=True, exist_ok=True)

BUNDLE_URL = os.getenv("VIETTOXIC_BUNDLE_URL", "").strip()
BUNDLE_ZIP = WORKDIR / "mlflow_bundle.zip"
BUNDLE_DIR = WORKDIR / "bundle"
SEED = int(os.getenv("VIETTOXIC_SEED", "42"))

TEST_MODE = os.getenv("VIETTOXIC_TEST_MODE", "smoke").strip().lower()
SMOKE_MAX_TRAIN = int(os.getenv("VIETTOXIC_SMOKE_MAX_TRAIN", "4000"))
SMOKE_MAX_VAL = int(os.getenv("VIETTOXIC_SMOKE_MAX_VAL", "1200"))
SMOKE_MAX_TEST = int(os.getenv("VIETTOXIC_SMOKE_MAX_TEST", "1200"))

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(WORKDIR / "mlruns"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "viettoxic-kaggle-retrain-smoke")
RUN_NAME = os.getenv("VIETTOXIC_RUN_NAME", f"kaggle_smoke_{time.strftime('%Y%m%d_%H%M%S')}")

IMPORT_API_URL = os.getenv("VIETTOXIC_IMPORT_API_URL", "").strip()
IMPORT_API_TOKEN = os.getenv("VIETTOXIC_IMPORT_API_TOKEN", "").strip()
IMPORT_ARTIFACT_PATH = os.getenv("VIETTOXIC_IMPORT_ARTIFACT_PATH", "").strip()
IMPORT_NOTES = os.getenv("VIETTOXIC_IMPORT_NOTES", "Kaggle retrain smoke test")
IMPORT_REQUIRED = os.getenv("VIETTOXIC_IMPORT_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
DATASET_ROOT_OVERRIDE = os.getenv("VIETTOXIC_DATASET_ROOT", "").strip()
BUNDLE_DOWNLOAD_REQUIRED = os.getenv("VIETTOXIC_BUNDLE_DOWNLOAD_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
MODEL_KIND = os.getenv("VIETTOXIC_MODEL_KIND", "phobert").strip().lower()
TRAINING_MODE = os.getenv("VIETTOXIC_TRAINING_MODE", "finetune").strip().lower()
BASE_MODEL = os.getenv("VIETTOXIC_BASE_MODEL", "").strip()
PHOBERT_DRY_RUN = os.getenv("VIETTOXIC_PHOBERT_DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "on"}
PHOBERT_MODEL_FALLBACK = os.getenv("VIETTOXIC_PHOBERT_MODEL_FALLBACK", "vinai/phobert-base-v2").strip()
PHOBERT_PSEUDO_LOSS_WEIGHT = os.getenv("VIETTOXIC_PSEUDO_LOSS_WEIGHT", "0.3").strip()
PHOBERT_MAX_PSEUDO_RATIO = os.getenv("VIETTOXIC_MAX_PSEUDO_RATIO", "0.3").strip()
SPLIT_NAME_CANDIDATES: dict[str, list[str]] = {
    "train": ["train.jsonl", "train_augmented.jsonl"],
    "validation": ["validation.jsonl", "validation_augmented.jsonl", "val.jsonl", "dev.jsonl"],
    "test": ["test.jsonl", "test_augmented.jsonl"],
}


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_python_package(module_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(module_name)
    except ImportError:
        pkg = pip_name or module_name
        print(f"Installing missing package: {pkg}")
        run([sys.executable, "-m", "pip", "install", "-q", pkg])


def download_bundle_if_configured() -> None:
    if not BUNDLE_URL:
        print("VIETTOXIC_BUNDLE_URL is empty -> skip bundle download.")
        return
    print(f"Downloading bundle from: {BUNDLE_URL}")
    try:
        with urlopen(BUNDLE_URL) as response:  # nosec B310
            BUNDLE_ZIP.write_bytes(response.read())
        print(f"Saved: {BUNDLE_ZIP}")

        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
            zf.extractall(BUNDLE_DIR)
        print(f"Extracted to: {BUNDLE_DIR}")
    except (HTTPError, URLError, zipfile.BadZipFile) as exc:
        msg = (
            f"Bundle download/extract failed ({exc}). "
            "Will continue and try resolving dataset from local paths (/kaggle/input or VIETTOXIC_DATASET_ROOT)."
        )
        if BUNDLE_DOWNLOAD_REQUIRED:
            raise RuntimeError(msg) from exc
        print(f"[WARN] {msg}")


def _build_split_paths(base_dir: pathlib.Path, nested: bool) -> dict[str, pathlib.Path]:
    root = base_dir / "dataset" / "victsd_gold" if nested else base_dir
    return {
        "train": root / "train.jsonl",
        "validation": root / "validation.jsonl",
        "test": root / "test.jsonl",
    }


def _choose_shortest_path(paths: list[pathlib.Path]) -> pathlib.Path:
    return sorted(paths, key=lambda p: (len(p.parts), len(str(p))))[0]


def _discover_split_paths(root: pathlib.Path) -> dict[str, pathlib.Path] | None:
    if not root.exists() or not root.is_dir():
        return None

    found: dict[str, pathlib.Path] = {}
    for split_name, file_names in SPLIT_NAME_CANDIDATES.items():
        selected: pathlib.Path | None = None
        for file_name in file_names:
            matches = [p for p in root.rglob(file_name) if p.is_file()]
            if matches:
                selected = _choose_shortest_path(matches)
                break
        if selected is None:
            return None
        found[split_name] = selected
    return found


def resolve_dataset_paths() -> tuple[dict[str, pathlib.Path], str]:
    candidate_roots: list[pathlib.Path] = []
    if DATASET_ROOT_OVERRIDE:
        candidate_roots.append(pathlib.Path(DATASET_ROOT_OVERRIDE))
    candidate_roots.append(BUNDLE_DIR)

    kaggle_input_root = pathlib.Path("/kaggle/input")
    if kaggle_input_root.exists():
        for child in kaggle_input_root.iterdir():
            if child.is_dir():
                candidate_roots.extend([child, child / "victsd_gold", child / "dataset", child / "dataset" / "victsd_gold"])

    # Remove duplicates but keep original order.
    seen: set[pathlib.Path] = set()
    unique_roots: list[pathlib.Path] = []
    for root in candidate_roots:
        if root not in seen:
            seen.add(root)
            unique_roots.append(root)

    attempted: list[pathlib.Path] = []
    for root in unique_roots:
        clean_paths = _build_split_paths(root, nested=False)
        if all(path.exists() for path in clean_paths.values()):
            return clean_paths, f"clean_victsd_gold@{root}"
        attempted.extend(clean_paths.values())

        full_paths = _build_split_paths(root, nested=True)
        if all(path.exists() for path in full_paths.values()):
            return full_paths, f"full_bundle@{root}"
        attempted.extend(full_paths.values())

        discovered = _discover_split_paths(root)
        if discovered is not None:
            return discovered, f"auto_discovered@{root}"

    attempted_str = "\n".join(f"- {p}" for p in attempted)
    input_dirs = []
    kaggle_input_root = pathlib.Path("/kaggle/input")
    if kaggle_input_root.exists():
        input_dirs = sorted(str(p) for p in kaggle_input_root.iterdir() if p.is_dir())
    input_dirs_str = "\n".join(f"- {p}" for p in input_dirs) if input_dirs else "- (none)"
    raise FileNotFoundError(
        "Could not resolve dataset files for smoke/validate run.\n"
        "Provide VIETTOXIC_BUNDLE_URL, or mount dataset under /kaggle/input, or set VIETTOXIC_DATASET_ROOT.\n"
        "Visible /kaggle/input directories:\n"
        f"{input_dirs_str}\n"
        f"Tried:\n{attempted_str}"
    )


def parse_binary_label(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed in (0, 1) else None


def load_jsonl_dataset(path: pathlib.Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if skipped <= 3:
                    print(f"[WARN] Invalid JSON at {path}:{line_no}")
                continue

            if not isinstance(item, dict):
                skipped += 1
                continue
            text = item.get("text")
            toxicity = parse_binary_label(item.get("toxicity", item.get("label")))
            constructiveness = parse_binary_label(item.get("constructiveness"))
            if not isinstance(text, str) or not text.strip():
                skipped += 1
                continue
            if toxicity is None:
                skipped += 1
                continue
            row: dict[str, int | str] = {"text": text.strip(), "toxicity": toxicity}
            if constructiveness is not None:
                row["constructiveness"] = constructiveness
            rows.append(row)
    if skipped:
        print(f"[WARN] Skipped {skipped} malformed rows in {path}")
    return rows


def downsample_rows(rows: list[dict[str, int | str]], max_rows: int) -> list[dict[str, int | str]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    import random

    rng = random.Random(SEED)
    copied = rows[:]
    rng.shuffle(copied)
    return copied[:max_rows]


def evaluate_split(name: str, y_true: list[int], y_pred: list[int], positive_name: str = "toxic") -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f"f1_{positive_name}": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_negative": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }
    if positive_name == "toxic":
        metrics["f1_clean"] = metrics["f1_negative"]
    print(f"[{name}] {json.dumps(metrics, ensure_ascii=False)}")
    return metrics


def zip_dir(source_dir: pathlib.Path, zip_path: pathlib.Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(source_dir)))


def _common_parent(paths: list[pathlib.Path]) -> pathlib.Path:
    resolved = [str(path.parent) for path in paths]
    return pathlib.Path(os.path.commonpath(resolved))


def resolve_bundle_member(*parts: str) -> pathlib.Path:
    return BUNDLE_DIR.joinpath(*parts)


def resolve_phobert_train_script() -> pathlib.Path:
    src_dir = pathlib.Path(__file__).resolve().parent
    candidates = [
        resolve_bundle_member("scripts", "train_phobert.py"),
        WORKDIR / "scripts" / "train_phobert.py",
        src_dir / "scripts" / "train_phobert.py",
        pathlib.Path("/kaggle/src/scripts/train_phobert.py"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "PhoBERT training script not found in bundle. Expected scripts/train_phobert.py. "
        "Export a full_bundle from the MLflow admin page before triggering phobert mode."
    )


def resolve_phobert_model_name() -> str:
    bundled_base = resolve_bundle_member("base_model")
    if TRAINING_MODE == "finetune" and (bundled_base / "config.json").exists():
        return str(bundled_base)
    if BASE_MODEL and not BASE_MODEL.startswith("phobert/"):
        return BASE_MODEL
    return PHOBERT_MODEL_FALLBACK


def run_phobert_retrain() -> dict:
    ensure_python_package("transformers")
    ensure_python_package("datasets")
    ensure_python_package("sklearn", "scikit-learn")
    ensure_python_package("accelerate")
    ensure_python_package("matplotlib")
    if MLFLOW_ENABLED:
        ensure_python_package("mlflow")

    dataset_paths, bundle_profile = resolve_dataset_paths()
    train_script = resolve_phobert_train_script()
    dataset_root = _common_parent([dataset_paths["train"], dataset_paths["validation"], dataset_paths["test"]])
    pseudo_dir = resolve_bundle_member("pseudo")
    has_pseudo = (pseudo_dir / "accepted.jsonl").exists() and (pseudo_dir / "manifest.json").exists()
    artifact_dir = WORKDIR / "artifacts" / RUN_NAME
    output_base = artifact_dir / "model"
    results_base = artifact_dir / "results"
    manifests_dir = artifact_dir / "manifests"
    zip_output_dir = WORKDIR / "phobert_zips"
    for path in [artifact_dir, output_base, results_base, manifests_dir, zip_output_dir]:
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(dataset_root),
            "DATASET_LAYOUT": "plain",
            "GOLD_DATA_DIR": str(dataset_root),
            "MODEL_NAME": resolve_phobert_model_name(),
            "TRAINING_MODE": TRAINING_MODE if TRAINING_MODE in {"finetune", "retrain"} else "finetune",
            "OUTPUT_BASE": str(output_base),
            "RESULTS_BASE": str(results_base),
            "RUN_MANIFEST_DIR": str(manifests_dir),
            "ZIP_OUTPUT_DIR": str(zip_output_dir),
            "MODEL_VERSION": f"phobert/{RUN_NAME}",
            "MLFLOW_ENABLED": "true" if MLFLOW_ENABLED else "false",
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME.replace("smoke", "phobert"),
            "PSEUDO_LOSS_WEIGHT": PHOBERT_PSEUDO_LOSS_WEIGHT,
            "MAX_PSEUDO_RATIO": PHOBERT_MAX_PSEUDO_RATIO,
        }
    )
    if has_pseudo:
        env["PSEUDO_LABELS_DIR"] = str(pseudo_dir)

    dry_cmd = [sys.executable, str(train_script), "--dry-run"]
    print("PhoBERT dry-run command:", " ".join(dry_cmd))
    subprocess.run(dry_cmd, check=True, env=env)
    if PHOBERT_DRY_RUN:
        return {
            "status": "ok",
            "mode": "phobert_dry_run",
            "run_name": RUN_NAME,
            "bundle_profile": bundle_profile,
            "dataset_root": str(dataset_root),
            "pseudo_enabled": has_pseudo,
            "model_name": env["MODEL_NAME"],
        }

    train_cmd = [sys.executable, str(train_script)]
    print("PhoBERT train command:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True, env=env)

    artifact_zip_path = zip_output_dir / "best_model_full_victsd_gold.zip"
    if not artifact_zip_path.exists():
        candidates = sorted(zip_output_dir.glob("best_model_full_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            artifact_zip_path = WORKDIR / f"{RUN_NAME}_phobert_model.zip"
            zip_dir(output_base, artifact_zip_path)
        else:
            artifact_zip_path = candidates[0]

    import_response = maybe_import_artifact(RUN_NAME, str(artifact_zip_path))
    summary = {
        "status": "ok",
        "mode": "phobert",
        "run_name": RUN_NAME,
        "training_mode": TRAINING_MODE,
        "bundle_profile": bundle_profile,
        "dataset_root": str(dataset_root),
        "pseudo_enabled": has_pseudo,
        "model_name": env["MODEL_NAME"],
        "artifact_zip_local": str(artifact_zip_path),
        "import_response": import_response,
    }
    (artifact_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def maybe_import_artifact(run_name: str, artifact_path: str) -> dict | None:
    if not IMPORT_API_URL:
        return None

    payload = {
        "run_name": run_name,
        "artifact_path": IMPORT_ARTIFACT_PATH or artifact_path,
        "notes": IMPORT_NOTES,
    }
    headers = {"Content-Type": "application/json"}
    if IMPORT_API_TOKEN:
        headers["Authorization"] = f"Bearer {IMPORT_API_TOKEN}"

    req = Request(
        IMPORT_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(req, timeout=45) as resp:  # nosec B310
            raw = (resp.read() or b"").decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {"status": "empty_response"}
    except HTTPError as exc:
        detail = (exc.read() or b"").decode("utf-8", errors="replace")[:500]
        msg = f"Import artifact failed with HTTP {exc.code}: {detail}"
        if IMPORT_REQUIRED:
            raise RuntimeError(msg) from exc
        print(f"[WARN] {msg}")
        return {"status": "error", "detail": msg}
    except (URLError, json.JSONDecodeError) as exc:
        msg = f"Import artifact failed: {exc}"
        if IMPORT_REQUIRED:
            raise RuntimeError(msg) from exc
        print(f"[WARN] {msg}")
        return {"status": "error", "detail": msg}


def run_smoke_retrain() -> dict:
    ensure_python_package("sklearn", "scikit-learn")
    ensure_python_package("joblib")
    if MLFLOW_ENABLED:
        ensure_python_package("mlflow")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib

    dataset_paths, bundle_profile = resolve_dataset_paths()
    print(f"Resolved dataset profile: {bundle_profile}")
    print(f"Dataset files: {dataset_paths}")

    train_rows = downsample_rows(load_jsonl_dataset(dataset_paths["train"]), SMOKE_MAX_TRAIN)
    val_rows = downsample_rows(load_jsonl_dataset(dataset_paths["validation"]), SMOKE_MAX_VAL)
    test_rows = downsample_rows(load_jsonl_dataset(dataset_paths["test"]), SMOKE_MAX_TEST)

    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Dataset is empty after loading/downsampling.")

    y_train = [int(item["toxicity"]) for item in train_rows]
    if len(set(y_train)) < 2:
        raise RuntimeError("Train split has <2 classes; cannot train LogisticRegression.")

    x_train = [str(item["text"]) for item in train_rows]
    x_val = [str(item["text"]) for item in val_rows]
    x_test = [str(item["text"]) for item in test_rows]
    y_val = [int(item["toxicity"]) for item in val_rows]
    y_test = [int(item["toxicity"]) for item in test_rows]
    has_constructiveness = all(
        "constructiveness" in item
        for split_rows in (train_rows, val_rows, test_rows)
        for item in split_rows
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=600,
        random_state=SEED,
        n_jobs=-1,
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    print("Training smoke model...")
    model.fit(x_train_vec, y_train)
    val_pred = model.predict(x_val_vec)
    test_pred = model.predict(x_test_vec)

    val_metrics = evaluate_split("validation_toxicity", y_val, val_pred.tolist(), positive_name="toxic")
    test_metrics = evaluate_split("test_toxicity", y_test, test_pred.tolist(), positive_name="toxic")

    constructiveness_payload = None
    constructiveness_model = None
    if has_constructiveness:
        y_train_constructiveness = [int(item["constructiveness"]) for item in train_rows]
        y_val_constructiveness = [int(item["constructiveness"]) for item in val_rows]
        y_test_constructiveness = [int(item["constructiveness"]) for item in test_rows]

        if len(set(y_train_constructiveness)) >= 2:
            constructiveness_model = LogisticRegression(
                class_weight="balanced",
                max_iter=600,
                random_state=SEED,
                n_jobs=-1,
            )
            print("Training constructiveness smoke model...")
            constructiveness_model.fit(x_train_vec, y_train_constructiveness)
            val_constructiveness_pred = constructiveness_model.predict(x_val_vec)
            test_constructiveness_pred = constructiveness_model.predict(x_test_vec)
            constructiveness_payload = {
                "validation": evaluate_split(
                    "validation_constructiveness",
                    y_val_constructiveness,
                    val_constructiveness_pred.tolist(),
                    positive_name="constructive",
                ),
                "test": evaluate_split(
                    "test_constructiveness",
                    y_test_constructiveness,
                    test_constructiveness_pred.tolist(),
                    positive_name="constructive",
                ),
                "label_counts": {
                    "train": {
                        "non_constructive": int(y_train_constructiveness.count(0)),
                        "constructive": int(y_train_constructiveness.count(1)),
                    },
                    "validation": {
                        "non_constructive": int(y_val_constructiveness.count(0)),
                        "constructive": int(y_val_constructiveness.count(1)),
                    },
                    "test": {
                        "non_constructive": int(y_test_constructiveness.count(0)),
                        "constructive": int(y_test_constructiveness.count(1)),
                    },
                },
            }
        else:
            constructiveness_payload = {
                "skipped": True,
                "reason": "train split has <2 constructiveness classes",
                "label_counts": {
                    "train": {
                        "non_constructive": int(y_train_constructiveness.count(0)),
                        "constructive": int(y_train_constructiveness.count(1)),
                    },
                },
            }
    else:
        constructiveness_payload = {
            "skipped": True,
            "reason": "constructiveness field missing from at least one row",
        }

    artifact_dir = WORKDIR / "artifacts" / RUN_NAME
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model_lr.joblib"
    constructiveness_model_path = artifact_dir / "model_constructiveness_lr.joblib"
    vectorizer_path = artifact_dir / "vectorizer.joblib"
    metrics_path = artifact_dir / "metrics.json"
    summary_path = artifact_dir / "run_summary.json"
    artifact_zip_path = WORKDIR / f"{RUN_NAME}.zip"

    joblib.dump(model, model_path)
    if constructiveness_model is not None:
        joblib.dump(constructiveness_model, constructiveness_model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics_payload = {
        "run_name": RUN_NAME,
        "mode": "smoke_retrain_tfidf_lr_multitask",
        "bundle_profile": bundle_profile,
        "tracking_uri": MLFLOW_TRACKING_URI if MLFLOW_ENABLED else None,
        "sizes": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "tasks": ["toxicity", "constructiveness"],
        "validation": val_metrics,
        "test": test_metrics,
        "toxicity": {
            "validation": val_metrics,
            "test": test_metrics,
            "label_counts": {
                "train": {"clean": int(y_train.count(0)), "toxic": int(y_train.count(1))},
                "validation": {"clean": int(y_val.count(0)), "toxic": int(y_val.count(1))},
                "test": {"clean": int(y_test.count(0)), "toxic": int(y_test.count(1))},
            },
        },
        "constructiveness": constructiveness_payload,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow_logged = False
    if MLFLOW_ENABLED:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=RUN_NAME):
            for key, value in metrics_payload["sizes"].items():
                mlflow.log_param(f"size_{key}", int(value))
            mlflow.log_param("bundle_profile", bundle_profile)
            mlflow.log_param("mode", "smoke_retrain_tfidf_lr_multitask")
            mlflow.log_param("tasks", "toxicity,constructiveness")
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_toxicity_{k}", float(v))
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_toxicity_{k}", float(v))
            if constructiveness_payload and not constructiveness_payload.get("skipped"):
                for k, v in constructiveness_payload["validation"].items():
                    mlflow.log_metric(f"val_constructiveness_{k}", float(v))
                for k, v in constructiveness_payload["test"].items():
                    mlflow.log_metric(f"test_constructiveness_{k}", float(v))
            mlflow.log_artifact(str(model_path))
            if constructiveness_model is not None:
                mlflow.log_artifact(str(constructiveness_model_path))
            mlflow.log_artifact(str(vectorizer_path))
            mlflow.log_artifact(str(metrics_path))
        mlflow_logged = True

    summary_payload = {
        "status": "ok",
        "run_name": RUN_NAME,
        "mlflow_logged": mlflow_logged,
        "artifact_dir": str(artifact_dir),
        "artifact_zip_local": str(artifact_zip_path),
        "import_api_url": IMPORT_API_URL or None,
        "import_artifact_path": IMPORT_ARTIFACT_PATH or str(artifact_zip_path),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_dir(artifact_dir, artifact_zip_path)
    import_response = maybe_import_artifact(RUN_NAME, str(artifact_zip_path))
    if import_response is not None:
        summary_payload["import_response"] = import_response
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        **summary_payload,
        "metrics": metrics_payload,
        "bundle_dir_exists": BUNDLE_DIR.exists(),
        "bundle_url_set": bool(BUNDLE_URL),
    }


def main() -> None:
    print("Python:", sys.version)
    print("Workdir:", WORKDIR)
    print("Test mode:", TEST_MODE)
    download_bundle_if_configured()

    if TEST_MODE == "validate":
        dataset_paths, bundle_profile = resolve_dataset_paths()
        status = {
            "status": "ok",
            "mode": "validate",
            "bundle_profile": bundle_profile,
            "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
            "bundle_url_set": bool(BUNDLE_URL),
            "bundle_dir_exists": BUNDLE_DIR.exists(),
        }
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    if TEST_MODE == "smoke":
        status = run_smoke_retrain()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    if TEST_MODE == "phobert":
        status = run_phobert_retrain()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unsupported VIETTOXIC_TEST_MODE={TEST_MODE}. Use 'validate', 'smoke', or 'phobert'.")


if __name__ == "__main__":
    main()

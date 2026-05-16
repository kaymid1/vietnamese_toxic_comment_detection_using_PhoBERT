# %% [markdown]
# # VietToxic MLflow Retrain (Mirror)
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
import zipfile
from urllib.request import urlopen

WORKDIR = pathlib.Path("/kaggle/working/viettoxic")
WORKDIR.mkdir(parents=True, exist_ok=True)

BUNDLE_URL = os.getenv("VIETTOXIC_BUNDLE_URL", "").strip()
BUNDLE_ZIP = WORKDIR / "mlflow_bundle.zip"
BUNDLE_DIR = WORKDIR / "bundle"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_bundle_if_configured() -> None:
    if not BUNDLE_URL:
        print("VIETTOXIC_BUNDLE_URL is empty -> skip bundle download.")
        return
    print(f"Downloading bundle from: {BUNDLE_URL}")
    with urlopen(BUNDLE_URL) as response:  # nosec B310
        BUNDLE_ZIP.write_bytes(response.read())
    print(f"Saved: {BUNDLE_ZIP}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
        zf.extractall(BUNDLE_DIR)
    print(f"Extracted to: {BUNDLE_DIR}")


def main() -> None:
    print("Python:", sys.version)
    print("Workdir:", WORKDIR)
    download_bundle_if_configured()

    # TODO:
    # - Dat training flow that su dung bundle/data cua ban.
    # - Neu can, pip install them package tai day.
    # - Chay train va ghi artifact vao /kaggle/working hoac /kaggle/output.
    #
    # Vi du:
    # run([sys.executable, "-m", "pip", "install", "-r", "requirements-ml.txt"])
    # run([sys.executable, "train.py", "--config", str(BUNDLE_DIR / "config" / "training_config.yaml")])
    #
    # Sau khi run xong, ban co the upload artifact qua API backend:
    # POST /api/mlflow/manual/import-artifact
    status = {
        "status": "ok",
        "bundle_url_set": bool(BUNDLE_URL),
        "bundle_dir_exists": BUNDLE_DIR.exists(),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


from datasets import load_dataset # type: ignore
import os

OUTPUT_DIR = "data/raw/victsd"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = load_dataset("tarudesu/ViCTSD")

for split in ["train", "validation", "test"]:
    ds[split].to_json(
        f"{OUTPUT_DIR}/{split}.jsonl", #jsonl -> good for nlp training, stream, diff.
        orient="records",
        lines=True,
        force_ascii=False
    )

print("✅ Export raw ViCTSD done")

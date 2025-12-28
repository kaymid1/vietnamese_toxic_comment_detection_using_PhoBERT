import json
import os

INPUT_DIR = "data/raw/victsd"
OUTPUT_DIR = "data/processed/victsd_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    return " ".join(text.strip().split())

def process_file(split: str):
    input_path = f"{INPUT_DIR}/{split}.jsonl"
    output_path = f"{OUTPUT_DIR}/{split}.jsonl"

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)

            text = clean_text(item["Comment"])
            label = int(item["Toxicity"])

            if not text:
                continue  # bỏ comment rỗng

            record = {
                "text": text,
                "label": label,
                "meta": {
                    "source": "ViCTSD"
                }
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

for split in ["train", "validation", "test"]:
    process_file(split)

print("✅ Preprocess + normalize done (victsd_v1)")

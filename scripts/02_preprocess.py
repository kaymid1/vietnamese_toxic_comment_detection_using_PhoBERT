import json
import os
import re

INPUT_DIR = "data/raw/victsd"
OUTPUT_DIR = "data/processed/victsd_v1"  # version này sẽ là baseline preprocessing
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    text = text.strip()
    
    # 1. Normalize unicode (rất quan trọng với tiếng Việt từ Facebook/Youtube)
    # Ví dụ: "tên" → "tên", "học" → "học"
    import unicodedata
    text = unicodedata.normalize("NFC", text) # chuẩn hóa về dạng composed
    
    # 2. Normalize whitespace (loại bỏ thừa, tab, multiple spaces)
    text = " ".join(text.split())
    
    # 3. Optional: lowercase → KHÔNG NÊN làm với PhoBERT
    # PhoBERT là case-sensitive và được pretrain trên text gốc tiếng Việt (có chữ hoa đầu câu)
    # → giữ nguyên case để tận dụng subword tốt hơn
    
    # 4. Không remove punctuation mạnh
    # Giữ lại ! ? . , " ' để bảo toàn sarcasm, cảm xúc (e.g. "đẹp vãi!!!" vs "đẹp")
    
    # 5. Optional nhẹ: xử lý emoji → giữ nguyên
    # Emoji thường mang ý nghĩa toxic (😂 khi troll, 😡 khi chửi)
    # PhoBERT tokenizer xử lý được emoji
    
    # 6. Optional: normalize teencode thường gặp (nếu muốn experiment)
    # Chỉ áp dụng nếu error analysis cho thấy nhiều teencode bị miss
    # Ví dụ: thay "k" → "không", "dc" → "được", "ns" → "nói" ...
    # Nhưng giai đoạn đầu → chưa cần, sẽ test ablation sau
    
    return text

def process_file(split: str):
    input_path = f"{INPUT_DIR}/{split}.jsonl"
    output_path = f"{OUTPUT_DIR}/{split}.jsonl"
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            item = json.loads(line)
            raw_text = item.get("Comment", "") or item.get("comment", "")  # phòng trường hợp key khác
            toxicity = int(item["Toxicity"])

            text = clean_text(raw_text)

            if not text:  # bỏ comment rỗng sau clean
                continue

            record = {
                "text": text,
                "toxicity": toxicity,
                "meta": {
                    "source": "ViCTSD",
                    "original_length": len(raw_text),
                    "processed_length": len(text),
                    # Nếu dataset có thêm topic/constructiveness thì giữ lại ở đây
                }
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"✅ Processed {split}.jsonl")

for split in ["train", "validation", "test"]:
    process_file(split)

print("✅ Preprocessing victsd_v1 hoàn tất – version này sẽ dùng cho baseline + PhoBERT đầu tiên")
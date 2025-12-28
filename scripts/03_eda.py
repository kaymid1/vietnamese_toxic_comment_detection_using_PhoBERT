# scripts/03_eda.py
import os
import re
import json
import argparse
from collections import Counter, defaultdict

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹ0-9_]+", re.UNICODE)

# Một stoplist nhỏ, không “đụng” dấu tiếng Việt nhiều quá.
# Bạn có thể mở rộng sau, nhưng EDA ban đầu đừng xóa mạnh tay.
VI_STOP = {
    "và","là","thì","mà","có","cho","của","một","những","các","đã","đang","sẽ","với","tôi","tao",
    "mày","bạn","anh","em","chị","ông","bà","nó","họ","chúng","chúng_ta","đây","đó","ở","khi","đến",
    "được","không","cũng","rất","quá","này","kia","ấy","như","vậy","thế","hay","vì","do","nên"
}

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def tokenize(text: str):
    # giữ tiếng Việt có dấu bằng regex range À-ỹ
    tokens = TOKEN_RE.findall(text.lower())
    return tokens

def ngrams(tokens, n=2):
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_bar(counter: Counter, title: str, out_path: str, topk: int = 20):
    items = counter.most_common(topk)
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel("Count")
    save_plot(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/victsd_v1", help="Folder containing train/validation/test jsonl")
    ap.add_argument("--out_dir", default="reports/eda", help="Output folder for EDA report/plots")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all, else cap samples per split for speed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
    stats = {}

    # Global counters per split
    for split, filename in splits.items():
        path = os.path.join(args.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        label_counter = Counter()
        lengths = []
        empty_count = 0

        tok_counter_by_label = {0: Counter(), 1: Counter()}
        bigram_counter_by_label = {0: Counter(), 1: Counter()}

        n = 0
        for row in read_jsonl(path):
            if args.max_samples and n >= args.max_samples:
                break
            n += 1

            text = (row.get("text") or "").strip()
            label = int(row.get("label"))

            if not text:
                empty_count += 1
                continue

            label_counter[label] += 1
            toks = tokenize(text)
            lengths.append(len(toks))

            # token stats (lọc stopwords nhẹ)
            toks_f = [t for t in toks if t not in VI_STOP and len(t) > 1]
            tok_counter_by_label[label].update(toks_f)

            # bigrams (lọc stopwords nhẹ)
            bi = ngrams(toks_f, n=2)
            bigram_counter_by_label[label].update(bi)

        lengths_arr = np.array(lengths, dtype=np.int32) if lengths else np.array([], dtype=np.int32)

        stats[split] = {
            "num_rows_read": n,
            "num_non_empty": int(sum(label_counter.values())),
            "num_empty_skipped": int(empty_count),
            "label_counts": dict(label_counter),
            "length_tokens": {
                "min": int(lengths_arr.min()) if lengths_arr.size else 0,
                "p50": int(np.percentile(lengths_arr, 50)) if lengths_arr.size else 0,
                "p90": int(np.percentile(lengths_arr, 90)) if lengths_arr.size else 0,
                "p95": int(np.percentile(lengths_arr, 95)) if lengths_arr.size else 0,
                "max": int(lengths_arr.max()) if lengths_arr.size else 0,
                "mean": float(lengths_arr.mean()) if lengths_arr.size else 0.0,
            },
            "top_tokens_clean": tok_counter_by_label[0].most_common(args.topk),
            "top_tokens_toxic": tok_counter_by_label[1].most_common(args.topk),
            "top_bigrams_clean": bigram_counter_by_label[0].most_common(args.topk),
            "top_bigrams_toxic": bigram_counter_by_label[1].most_common(args.topk),
        }

    # Plot 1: label distribution (train)
    train_labels = stats["train"]["label_counts"]
    plt.figure(figsize=(6, 4))
    xs = ["clean(0)", "toxic(1)"]
    ys = [train_labels.get(0, 0), train_labels.get(1, 0)]
    plt.bar(xs, ys)
    plt.title("Train label distribution")
    plt.ylabel("Count")
    save_plot(os.path.join(args.out_dir, "label_distribution.png"))

    # Plot 2: length distribution (train)
    # re-read lengths for train quickly
    lengths = []
    for row in read_jsonl(os.path.join(args.data_dir, splits["train"])):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        lengths.append(len(tokenize(text)))
        if args.max_samples and len(lengths) >= args.max_samples:
            break

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=50)
    plt.title("Train token length distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    save_plot(os.path.join(args.out_dir, "length_distribution.png"))

    # Plot 3-6: top tokens/bigrams (train by label)
    train_top_clean = Counter(dict(stats["train"]["top_tokens_clean"]))
    train_top_toxic = Counter(dict(stats["train"]["top_tokens_toxic"]))
    plot_bar(train_top_clean, "Top tokens (clean) - train", os.path.join(args.out_dir, "top_tokens_clean.png"), topk=args.topk)
    plot_bar(train_top_toxic, "Top tokens (toxic) - train", os.path.join(args.out_dir, "top_tokens_toxic.png"), topk=args.topk)

    train_bigram_clean = Counter(dict(stats["train"]["top_bigrams_clean"]))
    train_bigram_toxic = Counter(dict(stats["train"]["top_bigrams_toxic"]))
    plot_bar(train_bigram_clean, "Top bigrams (clean) - train", os.path.join(args.out_dir, "top_bigrams_clean.png"), topk=args.topk)
    plot_bar(train_bigram_toxic, "Top bigrams (toxic) - train", os.path.join(args.out_dir, "top_bigrams_toxic.png"), topk=args.topk)

    # Write markdown report
    md_path = os.path.join(args.out_dir, "eda_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# EDA Summary (ViCTSD processed)\n\n")
        f.write(f"- Data dir: `{args.data_dir}`\n")
        f.write(f"- Max samples per split: `{args.max_samples or 'ALL'}`\n\n")

        for split in ["train", "validation", "test"]:
            s = stats[split]
            f.write(f"## Split: {split}\n")
            f.write(f"- Rows read: {s['num_rows_read']}\n")
            f.write(f"- Non-empty used: {s['num_non_empty']}\n")
            f.write(f"- Empty skipped: {s['num_empty_skipped']}\n")
            f.write(f"- Label counts: {s['label_counts']}\n")
            f.write(f"- Token length stats: {s['length_tokens']}\n\n")

        f.write("## Train plots\n")
        f.write("- `label_distribution.png`\n")
        f.write("- `length_distribution.png`\n")
        f.write("- `top_tokens_clean.png`, `top_tokens_toxic.png`\n")
        f.write("- `top_bigrams_clean.png`, `top_bigrams_toxic.png`\n\n")

        f.write("## Quick keyword signals (train)\n")
        f.write("### Top tokens (toxic)\n")
        for tok, cnt in stats["train"]["top_tokens_toxic"]:
            f.write(f"- {tok}: {cnt}\n")
        f.write("\n### Top tokens (clean)\n")
        for tok, cnt in stats["train"]["top_tokens_clean"]:
            f.write(f"- {tok}: {cnt}\n")

    print(f"✅ EDA done. Report: {md_path}")

if __name__ == "__main__":
    main()

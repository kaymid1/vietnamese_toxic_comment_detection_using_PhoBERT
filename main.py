from datasets import load_dataset

ds = load_dataset("tarudesu/ViCTSD")

# Export từng split
ds["train"].to_json("train.json", orient="records", force_ascii=False)
ds["validation"].to_json("validation.json", orient="records", force_ascii=False)
ds["test"].to_json("test.json", orient="records", force_ascii=False)

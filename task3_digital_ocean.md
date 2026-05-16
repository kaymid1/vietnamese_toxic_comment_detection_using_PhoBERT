1. Hiện tại sau khi đã đủ data ở phần threshold thì tôi đang tiến hành download thử dataset thì không thấy gì xuất hiện. (Expected output: victsd_gold và sẽ cộng thêm data mới, đảm bảo format của chúng chuẩn nhé, giống với victsd_gold: /Users/mac/git/Thesis/data/processed/victsd_gold)

2. Bắt đầu nghiên cứu để triển khai flow digital ocean (ưu tiên hơn phần làm thủ công trên google colab hiện tại)
- Tạo VM có GPU
- Đẩy dataset mới lên
- Chạy script train phobert /Users/mac/git/Thesis/scripts/06_train_phobert_lora_macro_f1.py
- Export model
- Lưu về (cần suggest solution cho cái này)
- Xoá VM 
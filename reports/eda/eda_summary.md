# EDA Summary (ViCTSD processed)

- Data dir: `data/processed/victsd_v1`
- Max samples per split: `ALL`

## Split: train
- Rows read: 7000
- Non-empty used: 7000
- Empty skipped: 0
- Label counts: {0: 6241, 1: 759}
- Token length stats: {'min': 1, 'p50': 20, 'p90': 61, 'p95': 83, 'max': 337, 'mean': 29.236}

## Split: validation
- Rows read: 2000
- Non-empty used: 2000
- Empty skipped: 0
- Label counts: {0: 1768, 1: 232}
- Token length stats: {'min': 1, 'p50': 21, 'p90': 63, 'p95': 87, 'max': 343, 'mean': 30.316}

## Split: test
- Rows read: 1000
- Non-empty used: 1000
- Empty skipped: 0
- Label counts: {1: 110, 0: 890}
- Token length stats: {'min': 1, 'p50': 20, 'p90': 62, 'p95': 81, 'max': 336, 'mean': 28.925}

## Train plots
- `label_distribution.png`
- `length_distribution.png`
- `top_tokens_clean.png`, `top_tokens_toxic.png`
- `top_bigrams_clean.png`, `top_bigrams_toxic.png`

## Quick keyword signals (train)
### Top tokens (toxic)
- người: 254
- học: 223
- con: 221
- phải: 156
- còn: 132
- lại: 131
- đi: 123
- làm: 117
- rồi: 115
- thật: 111
- sao: 109
- gì: 108
- để: 107
- nhưng: 95
- nhiều: 93
- mình: 92
- ra: 91
- ko: 88
- chỉ: 87
- thấy: 84

### Top tokens (clean)
- người: 1395
- con: 1097
- đi: 961
- lại: 924
- để: 893
- phải: 828
- còn: 785
- nhiều: 775
- mình: 774
- xe: 769
- hơn: 764
- làm: 749
- trong: 741
- học: 715
- rồi: 703
- nhưng: 680
- ra: 665
- năm: 620
- mới: 616
- chỉ: 607

📊 EDA Artifacts & Interpretation
Phần này mô tả ý nghĩa học thuật của các biểu đồ (PNG) được sinh ra trong bước Exploratory Data Analysis (EDA), nhằm làm rõ đặc tính của dataset ViCTSD và định hướng cho các quyết định mô hình hóa tiếp theo.
1️⃣ label_distribution.png — Phân bố nhãn
Mô tả:
Biểu đồ cột thể hiện số lượng mẫu thuộc hai lớp:
clean (0)
toxic (1)
trong tập train.
Ý nghĩa:
Giúp xác định mức độ mất cân bằng nhãn (class imbalance).
Nếu lớp toxic chiếm tỷ lệ nhỏ, các metric như accuracy sẽ không còn phù hợp; thay vào đó cần dùng Macro-F1, F1_toxic.
Là cơ sở để:
dùng class_weight="balanced" trong baseline
ưu tiên F1_toxic khi chọn checkpoint PhoBERT
Liên hệ nghiên cứu:
→ Trực tiếp hỗ trợ cho phần Evaluation Metrics và Model Selection Criteria.
2️⃣ length_distribution.png — Phân bố độ dài câu (token length)
Mô tả:
Histogram thể hiện phân bố số lượng token của các comment trong tập train.
Ý nghĩa:
Cho biết comment trong ViCTSD:
thường ngắn hay dài
có nhiều outlier rất dài hay không
Là cơ sở để lựa chọn:
max_length cho PhoBERT (ví dụ 128 hay 256)
tránh cắt mất ngữ cảnh quan trọng
Liên hệ mô hình:
Nếu phần lớn comment < 128 tokens → max_length=128 là hợp lý, tiết kiệm tài nguyên.
Nếu có đuôi dài (long-tail) → cần phân tích lỗi do truncation.
3️⃣ top_tokens_toxic.png — Từ khóa phổ biến trong lớp toxic
Mô tả:
Biểu đồ cột ngang hiển thị Top-K token xuất hiện nhiều nhất trong các comment được gán nhãn toxic.
Ý nghĩa:
Phát hiện explicit toxic keywords (chửi thề, công kích trực tiếp).
Giúp xác định:
dataset có keyword bias hay không
baseline model có thể “ăn gian” bằng keyword matching
Liên hệ Error Analysis:
Các token này thường gây:
False Positive: clean nhưng chứa từ “xấu”
Là input quan trọng cho phần Error Analysis – Keyword Bias.
4️⃣ top_tokens_clean.png — Từ khóa phổ biến trong lớp clean
Mô tả:
Top-K token xuất hiện nhiều nhất trong các comment clean.
Ý nghĩa:
So sánh với lớp toxic để:
xác định token nào mang tính phân biệt
phát hiện token trung tính nhưng có tần suất cao
Giúp hiểu rõ ngôn ngữ phổ biến trong comment không độc hại.
Giá trị nghiên cứu:
Giải thích vì sao baseline có thể nhầm lẫn khi token overlap giữa hai lớp.
5️⃣ top_bigrams_toxic.png — Bigram phổ biến trong lớp toxic
Mô tả:
Biểu đồ các cặp từ (bigram) xuất hiện nhiều nhất trong toxic comments.
Ý nghĩa:
Cho thấy pattern ngữ cảnh ngắn, ví dụ:
công kích gián tiếp
cụm từ mỉa mai
Bigram giúp vượt qua hạn chế của unigram (chỉ nhìn từng từ đơn lẻ).
Liên hệ mô hình:
Giải thích vì sao:
TF-IDF (1–2 grams) cải thiện so với unigram
PhoBERT có lợi thế hơn nhờ contextual encoding.
6️⃣ top_bigrams_clean.png — Bigram phổ biến trong lớp clean
Mô tả:
Top bigram xuất hiện trong comment clean.
Ý nghĩa:
Làm đối chứng với bigram toxic.
Giúp xác định:
các pattern giao tiếp trung tính / tích cực
bigram có thể gây nhiễu cho baseline nếu xuất hiện ở cả hai lớp.
7️⃣ Tổng kết vai trò của EDA trong nghiên cứu
EDA không nhằm tối ưu mô hình, mà nhằm:
✅ Hiểu đặc tính ngôn ngữ của dataset tiếng Việt
✅ Giải thích hành vi của baseline và PhoBERT
✅ Làm nền cho Error Analysis (false positive / false negative)
✅ Đưa ra quyết định có cơ sở về:
metric đánh giá
tham số mô hình
chiến lược cải thiện
EDA results are directly used to interpret model performance and analyze systematic errors rather than to optimize training.
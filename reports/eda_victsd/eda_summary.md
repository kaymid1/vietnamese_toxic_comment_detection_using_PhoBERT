# EDA Summary (ViCTSD processed)

- Data dir: `/Users/mac/git/Thesis/data/victsd`
- Max samples per split: `ALL`

## Split: train
- Rows read: 6482
- Non-empty used: 6482
- Empty skipped: 0
- Label counts: {0: 4369, 1: 2113}
- Token length stats: {'min': 1, 'p50': 15, 'p90': 53, 'p95': 72, 'max': 337, 'mean': 24.329527923480406}

## Split: validation
- Rows read: 1852
- Non-empty used: 1852
- Empty skipped: 0
- Label counts: {0: 1248, 1: 604}
- Token length stats: {'min': 1, 'p50': 17, 'p90': 52, 'p95': 76, 'max': 261, 'mean': 25.14362850971922}

## Split: test
- Rows read: 926
- Non-empty used: 926
- Empty skipped: 0
- Label counts: {0: 624, 1: 302}
- Token length stats: {'min': 0, 'p50': 16, 'p90': 56, 'p95': 81, 'max': 336, 'mean': 25.982721382289416}

## Train plots
- `label_distribution.png`
- `length_distribution.png`
- `top_tokens_clean.png`, `top_tokens_toxic.png`
- `top_bigrams_clean.png`, `top_bigrams_toxic.png`

## Quick keyword signals (train)
### Top tokens (toxic)
- con: 242
- cái: 240
- người: 230
- gì: 207
- đi: 204
- học: 184
- làm: 169
- rồi: 164
- ra: 159
- phải: 159
- còn: 151
- lại: 144
- nói: 141
- đéo: 134
- mẹ: 132
- ko: 130
- để: 129
- nào: 117
- thấy: 112
- thật: 110

### Top tokens (clean)
- người: 1018
- con: 765
- đi: 662
- lại: 655
- để: 632
- phải: 620
- còn: 556
- mình: 550
- nhiều: 539
- làm: 533
- xe: 525
- học: 510
- trong: 509
- hơn: 507
- rồi: 500
- nhưng: 465
- ra: 453
- mới: 452
- chỉ: 441
- năm: 411

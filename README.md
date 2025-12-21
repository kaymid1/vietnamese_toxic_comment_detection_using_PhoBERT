# vietnamese_toxic_comment_detection_using_PhoBERT
```mermaid
sequenceDiagram
  autonumber
  actor User as User
  participant Page as Web Page (Any site)
  participant Ext as Browser Extension (Content Script / Popup)
  participant API as Inference API (FastAPI/Flask)
  participant Pre as Preprocess
  participant Tok as PhoBERT Tokenizer
  participant Model as PhoBERT Classifier
  participant Log as Logging/Monitoring (async)

  User->>Page: Select (highlight) a text segment
  User->>Ext: Right-click / click extension → "Scan Toxicity"
  Ext->>Ext: Read selected text from page\n(validate non-empty, length limit)
  Ext->>API: POST /predict {text, page_url(optional), client_ts}
  API->>Pre: clean_text(text)\n(strip, normalize whitespace, keep diacritics)
  Pre-->>API: cleaned_text
  API->>Tok: tokenize(cleaned_text)\n(max_length=128/256, truncation, padding)
  Tok-->>API: input_ids, attention_mask
  API->>Model: forward(input_ids, attention_mask)
  Model-->>API: logits / probabilities
  API->>API: thresholding + response formatting\n(label, confidence, request_id, latency_ms)
  API-->>Ext: 200 OK {label, confidence, request_id, latency_ms}
  Ext->>Page: Render inline overlay/tooltip\n(color badge + confidence)
  Page-->>User: Show result next to selected text

  par Async observability
    API-->>Log: log(request_id, latency_ms, text_len,\npred_label, confidence)
```
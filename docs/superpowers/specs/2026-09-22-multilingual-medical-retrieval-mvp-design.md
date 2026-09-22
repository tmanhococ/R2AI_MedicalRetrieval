# DESIGN SPECIFICATION: MULTILINGUAL MEDICAL RETRIEVAL MVP (ROAD TO AI)

- **Date**: 2026-09-22
- **Branch**: `feat/r2ai-multilingual-retrieval-mvp`
- **Status**: Complete Design — Đã hợp nhất đầy đủ nội dung gốc và bổ sung các thành phần mở rộng
- **Mục tiêu**: Xây dựng một hệ thống benchmark-first cho bài toán Medical Retrieval: nhận **Vietnamese Query** (hỗ trợ cả câu hỏi ngắn lẫn ca bệnh lâm sàng dài), áp dụng **Adaptive Query Router**, tạo 3 nhánh truy vấn **VI / EN / ZH**, chạy **BM25 + Dense Retrieval** trên 6 streams song song với backend linh hoạt (BM25s/Rank-BM25, FAISS/NumPy), hợp nhất bằng **Weighted Fusion**, lấy `top 100–300` ứng viên để **Rerank** (với chiến lược **Chunk Aggregation** giải quyết giới hạn 512 tokens), tính điểm liên kết giữa **Document + Chunk Joint Scoring**, kiểm soát tính nhất quán **Structural Consistency**, sau đó **F2 Calibration + Adaptive Cutoff** để tối ưu hóa trực tiếp **Macro F2**, sinh output cuối cùng và export sang format submission của cuộc thi.

---

## 1. Context & Motivation

Bài toán MVP này không nhằm xây chatbot hay hệ thống sinh câu trả lời (Answer Generation). Mục tiêu cốt lõi là xây một **retrieval engine có thể đo, so sánh và tối ưu trực tiếp theo benchmark thi đấu**.

Hệ thống cần giải quyết đồng thời các vấn đề sau:

1. **Cross-lingual retrieval:** Query đầu vào bằng tiếng Việt nhưng tài liệu có thể nằm ở tiếng Việt, tiếng Anh hoặc tiếng Trung.
2. **Hybrid retrieval:** Kết hợp lexical matching (BM25) và semantic matching (Dense Retrieval) để giảm bỏ sót.
3. **Adaptive Query Handling:** Tự động điều chỉnh chiến lược theo độ dài query (câu hỏi ngắn vs ca bệnh lâm sàng dài) để tránh làm loãng từ khóa trên BM25 và giảm tải dịch thuật.
4. **Document + Chunk retrieval:** Không chỉ tìm đúng tài liệu mà còn phải xác định đúng đoạn liên quan bên trong tài liệu.
5. **Context-Length Aware Reranking:** Giải quyết triệt để rào cản 512 tokens của Cross-Encoder đối với tài liệu dài thông qua cơ chế tổng hợp điểm từ chunk (*Chunk Aggregation*).
6. **Metric-aware output:** Không mặc định `top_k` cố định; số lượng kết quả cuối cùng phải được hiệu chỉnh để tối ưu **Macro F2**, ưu tiên Recall nhưng vẫn kiểm soát Precision.

MVP được xem như một **experiment platform**. Mỗi thay đổi model, trọng số, fusion, reranker hoặc cutoff phải có thể được chạy lại và so sánh bằng metric rõ ràng.

---

## 2. YÊU CẦU & NGUYÊN TẮC BẮT BUỘC

1. **Competition-first:**
   - Ưu tiên benchmark, reproducibility và tốc độ thử nghiệm hơn UI.
   - Mọi component phải có thể bật/tắt và thay đổi bằng config.

2. **Không Data Leakage:**
   - Không dùng label/test ground truth trong retrieval, reranking hoặc calibration.
   - Calibration chỉ được fit/tune trên validation set được phép sử dụng.

3. **Reproducible:**
   - Tất cả experiment có `seed`, config snapshot, model version và output riêng.
   - Cùng dataset + config + seed phải cho cùng kết quả logic và tie-breaking.

4. **Tách Offline Indexing khỏi Query-time Inference:**
   - Document/chunk embeddings và BM25 index được build trước.
   - Khi chạy benchmark, không rebuild index cho từng query.

5. **Không hard-code format dữ liệu BTC:**
   - Dataset chính thức đi qua `DatasetAdapter`.
   - Submission chính thức đi qua `SubmissionExporter`.
   - Core retrieval không phụ thuộc CSV/JSON/JSONL cụ thể.

6. **Giữ Original Vietnamese Query làm nhánh bắt buộc:**
   - Translation EN/ZH là nhánh bổ sung.
   - Translation lỗi không được làm hỏng toàn pipeline.

7. **Ưu tiên Recall ở Candidate Generation:**
   - Retrieval stage lấy tập ứng viên rộng (`100–300`).
   - Precision được cải thiện ở reranker + joint scoring + cutoff.

8. **Config-Driven & Strategy-Pluggable:**
   - Tất cả các điểm chưa cố định về dữ liệu (chunking sẵn hay tự chunk, tokenizer, vector backend, translation provider, consistency mode) được trừu tượng hóa và điều khiển qua config YAML.

9. **Không đưa các thành phần không phục vụ benchmark vào MVP:**
   - Không Answer Generation.
   - Không Agent loop.
   - Không MCP server.
   - Không full Adaptive RAG orchestration.
   - Không UI sản phẩm trong scope MVP đầu tiên.

---

## 3. DATA CONTRACT & INTERNAL SCHEMA

Do schema dataset chính thức có thể thay đổi, hệ thống sử dụng internal schema ổn định.

### 3.1. QueryRecord

```python
@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text_vi: str
    metadata: dict | None = None
```

### 3.2. DocumentRecord

```python
@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    language: Literal["vi", "en", "zh"]
    text: str
    title: str | None = None
    metadata: dict | None = None
```

### 3.3. ChunkRecord

```python
@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    language: Literal["vi", "en", "zh"]
    text: str
    chunk_index: int | None = None
    metadata: dict | None = None
```

### 3.4. GroundTruthRecord

Chỉ tồn tại cho train/dev/evaluation set được phép sử dụng.

```python
@dataclass(frozen=True)
class GroundTruthRecord:
    query_id: str
    relevant_doc_ids: set[str]
    relevant_chunk_ids: set[str]
```

### 3.5. SearchHit & Intermediate Hits

```python
@dataclass(frozen=True)
class SearchHit:
    item_id: str
    item_type: Literal["document", "chunk"]
    doc_id: str
    language: str
    source: str
    raw_score: float
    rank: int

@dataclass(frozen=True)
class FusedHit:
    item_id: str
    item_type: Literal["document", "chunk"]
    doc_id: str
    language: str
    fusion_score: float
    stream_ranks: dict[str, int]

@dataclass(frozen=True)
class ScoredHit:
    item_id: str
    item_type: Literal["document", "chunk"]
    doc_id: str
    language: str
    rerank_score: float
    normalized_score: float
    raw_fusion_score: float
```

### 3.6. FinalPrediction

```python
@dataclass(frozen=True)
class FinalPrediction:
    query_id: str
    document_ids: list[str]
    chunk_ids: list[str]
    document_scores: dict[str, float]
    chunk_scores: dict[str, float]
```

---

## 4. CONFIGURATION & PARAMETER PRIORITY

### 4.1. Hierarchy

1. **CLI flags**: ưu tiên cao nhất cho experiment override nhanh.
2. **YAML config**: cấu hình chính của từng experiment.
3. **Safe defaults**: dùng khi không khai báo explicit.

### 4.2. YAML Schema (`configs/r2ai_mvp.yaml`)

```yaml
run:
  experiment_name: "baseline_multilingual_hybrid"
  seed: 42
  output_dir: "artifacts/experiments"

indexing:
  chunking:
    mode: "hybrid" # prechunked | auto | hybrid
    auto_strategy:
      type: "sliding_window" # sliding_window | fixed_tokens
      chunk_size: 512
      chunk_overlap: 64
      min_chunk_length: 50
  validation:
    strict_foreign_keys: true
    drop_duplicates: true

query_understanding:
  normalize: true
  adaptive_routing:
    enabled: true
    length_threshold: 50
    strip_clinical_boilerplate: true
    dense_strategy: "head_tail" # full | head_tail
    max_dense_tokens: 512
  translation:
    enabled: true
    target_languages: ["en", "zh"]
    provider: "precomputed_cache" # precomputed_cache | local_hf | api | passthrough
    cache_path: "artifacts/cache/translations.jsonl"
    fallback_on_error: true
    local_hf_model: "facebook/nllb-200-distilled-600M"

retrieval:
  candidate_top_k: 200

  bm25:
    enabled: true
    backend: "bm25s" # bm25s | rank_bm25
    top_k: 200
    tokenizers:
      vi: "syllable"   # syllable | pyvi
      en: "whitespace" # whitespace | nltk_stem
      zh: "jieba"      # jieba | char

  dense:
    enabled: true
    backend: "faiss" # faiss | numpy
    device: "auto"   # auto | cuda | cpu
    model_name: "intfloat/multilingual-e5-large"
    top_k: 200
    batch_size: 64
    normalize_embeddings: true
    instruction_prefix:
      query: "query: "
      passage: "passage: "

fusion:
  method: "weighted_rrf"
  rrf_k: 60
  weights:
    vi_bm25: 1.0
    vi_dense: 1.0
    en_bm25: 1.0
    en_dense: 1.0
    zh_bm25: 1.0
    zh_dense: 1.0

reranker:
  enabled: true
  model_name: "BAAI/bge-reranker-v2-m3"
  document_strategy: "chunk_aggregation" # chunk_aggregation | truncate_direct
  document_top_k: 100
  chunk_top_k: 200
  batch_size: 32
  max_length: 512

joint_scoring:
  document:
    base_weight: 0.70
    best_chunk_weight: 0.20
    top_chunks_mean_weight: 0.10
    top_m_chunks: 3

  chunk:
    base_weight: 0.85
    parent_document_weight: 0.15

post_processing:
  consistency_mode: "independent" # independent | enforce_parent_doc | filter_orphan_chunks

calibration:
  metric: "macro_f2"
  beta: 2.0
  policies:
    - "fixed_top_k"
    - "global_threshold"
    - "relative_threshold"
    - "score_gap"
  bounds:
    documents: { min: 1, max: 50 }
    chunks: { min: 1, max: 100 }

submission:
  exporter: "internal_jsonl" # internal_jsonl | competition_csv
```

> Các giá trị weight/model trên là **baseline khởi đầu**, không phải giá trị cố định. Mọi giá trị phải được tune trên validation set.

---

## 5. ARCHITECTURE & DATA FLOW

```text
                           [Vietnamese Query]
                                    │
                                    ▼
                          [Query Understanding]
                 normalize + medical symbol protection
                                    │
                                    ▼
                         [Adaptive Query Router]
                  Length > 50 words? (Clinical Case)
                         /                      \
             (Yes)      /                        \   (No - Short Query)
                       ▼                          ▼
          [Condensed VI] + [Head-Tail]         [Raw VI]
                 │                 │               │
                 ▼                 │               ▼
        [Translation Layer]        │      [Translation Layer]
      (Precomputed / HF / API)     │    (Precomputed / HF / API)
          │            │           │        │            │
          ▼            ▼           │        ▼            ▼
      [Query EN]   [Query ZH]      │    [Query EN]   [Query ZH]
          │            │           │        │            │
          └────────────┼───────────┴────────┼────────────┘
                       │                    │
                       ▼                    ▼
     ┌────────────────────────────────────────────────────────┐
     │           6 Parallel Retrieval Streams                 │
     │  VI_BM25     VI_DENSE    EN_BM25   EN_DENSE   ZH_BM25  │
     │ (BM25s/PyVi) (FAISS/E5)  (NLTK)   (FAISS/E5)  (Jieba)  │
     └─────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                       [Weighted Fusion]
                          Weighted RRF
                               │
                               ▼ top 100–300 candidates
                    [Cross-Encoder Reranker]
                Chunk Reranking + Chunk Aggregation
                               │
                               ▼
                   [Min-Max Normalization]
                               │
                               ▼
                 [Document ↔ Chunk Joint Scoring]
                    document ↔ chunk evidence
                               │
                               ▼
             [Structural Consistency Post-Processing]
                 (Independent / Enforce Parent)
                               │
                               ▼
                       [F2 Calibration]
                 (Global / Relative / Score-Gap)
                               │
                               ▼
                       [Adaptive Cutoff]
                               │
                               ▼
                      [Final Prediction]
                 doc_ids + chunk_ids + scores
                               │
                               ▼
                     [Submission Exporter]
```

---

## 6. OFFLINE INDEXING ARCHITECTURE

### 6.1. Language Partitioning

Corpus được chia logic theo ngôn ngữ:

```text
corpus/
├── vi/
├── en/
└── zh/
```

Không bắt buộc tách file vật lý, nhưng index phải hỗ trợ filter theo `language`.

### 6.2. BM25 Index

Build riêng cho:
- Document text.
- Chunk text.
- Mỗi language branch `vi`, `en`, `zh`.

Hỗ trợ 2 backend linh hoạt:
- `bm25s`: Mặc định, tốc độ tính bằng C, hỗ trợ mmap, tiêu tốn ít RAM.
- `rank_bm25`: Chuẩn Python thuần phục vụ đối chứng.

BM25 phục vụ exact/lexical matching, đặc biệt hữu ích với:
- tên bệnh;
- tên thuốc;
- thuật ngữ chuyên môn;
- chữ viết tắt;
- số liệu/mức độ/phân loại.

Tokenizer cho từng ngôn ngữ được cấu hình riêng:
- `vi`: `"syllable"` (regex tách âm tiết `\w+`) hoặc `"pyvi"` (tách từ ghép).
- `en`: `"whitespace"` hoặc `"nltk_stem"`.
- `zh`: `"jieba"` hoặc `"char"`.

### 6.3. Dense Index

Embedding được tạo offline cho toàn bộ chunk.

```text
Chunk Text
   ↓
Instruction Prefix Wrapper (e.g. "passage: ")
   ↓
Dense Encoder (e.g. multilingual-e5-large)
   ↓
Embedding Vector
   ↓
Vector Index (FAISS / NumPy)
```

Vector backend:
- `faiss`: Tối ưu tìm kiếm vector nhanh trên CPU (IndexFlatIP, HNSW) hoặc GPU.
- `numpy`: Zero-dependency matrix dot-product fallback.

Document dense score không bắt buộc embed cả document dài. Mặc định MVP tính document dense evidence từ các chunk con phù hợp nhất:

```text
dense_doc_score(doc) = max(dense_chunk_scores_of_doc)
```

Có thể mở rộng sang `mean(top_m)` trong experiment sau.

### 6.4. Dynamic Chunking Layer (`src/indexing/chunker.py`)

Cung cấp interface `BaseChunker` xử lý 3 kịch bản:
- `mode == "prechunked"`: Đọc trực tiếp danh sách `ChunkRecord` do adapter nạp.
- `mode == "auto"`: Kích hoạt `AutoChunker` chia nhỏ Document thành các chunk theo `sliding_window` (chunk_size: 512, overlap: 64) và sinh `chunk_id = f"{doc_id}_c{chunk_index}"`.
- `mode == "hybrid"`: Nạp chunk có sẵn nếu có; chỉ tự sinh chunk cho những document chưa có chunk con.

### 6.5. Index Artifacts

```text
artifacts/indexes/
├── bm25/
│   ├── vi_documents/
│   ├── vi_chunks/
│   ├── en_documents/
│   ├── en_chunks/
│   ├── zh_documents/
│   └── zh_chunks/
└── dense/
    ├── chunk_embeddings.*
    ├── vector_index.*
    └── metadata.*
```

Index phải chứa metadata mapping ngược:
- `chunk_id -> doc_id`
- `item_id -> language`
- `item_id -> original text`

---

## 7. COMPONENT SPECIFICATION

### 7.1. Dataset Adapter — `src/data/dataset_adapter.py`

#### Interface

```python
class DatasetAdapter(Protocol):
    def load_documents(self) -> list[DocumentRecord]: ...
    def load_chunks(self) -> list[ChunkRecord]: ...
    def load_queries(self, split: str) -> list[QueryRecord]: ...
    def load_ground_truth(self, split: str) -> list[GroundTruthRecord]: ...
```

#### Responsibilities

- Parse format dataset bên ngoài.
- Chuẩn hóa ID thành string.
- Chuẩn hóa language về `vi | en | zh`.
- Validate `chunk.doc_id` luôn tồn tại trong document table.
- Không chứa retrieval/model logic.

---

### 7.2. Dynamic Chunker — `src/indexing/chunker.py`

#### Interface

```python
class Chunker(Protocol):
    def chunk_documents(self, documents: list[DocumentRecord]) -> list[ChunkRecord]: ...
```

#### Responsibilities
- Chia nhỏ document thành các chunk dựa trên cấu hình `chunk_size` và `chunk_overlap`.
- Bảo toàn liên kết `doc_id` và thứ tự `chunk_index`.
- Gắn nhãn metadata nhận diện chunk tự sinh.

---

### 7.3. Query Understanding & Adaptive Router — `src/query/query_processor.py`

#### `normalize_vi_query(text: str) -> str`

- Unicode normalize chuẩn NFC.
- Trim duplicate whitespace.
- **Medical Rule Guard:** Giữ nguyên số, đơn vị, mã bệnh, tên thuốc, abbreviation chuyên ngành:
  `HA`, `ĐTĐ`, `THA`, `COPD`, `mg`, `ml`, `mmol/L`, `mmHg`, `(+)`, `(-)`.
- Không stemming mạnh làm mất nghĩa y khoa.

#### `route_query(query: QueryRecord) -> ProcessedQuery`

- **Nếu Query ngắn (`<= length_threshold`, mặc định 50 từ):**
  - Giữ nguyên query cho tất cả các nhánh retrieval.
- **Nếu Query dài (`> length_threshold`, dạng ca lâm sàng):**
  - Sinh bản rút gọn `condensed_vi` bằng cách loại bỏ các cụm từ đệm bệnh án thông dụng (*"bệnh nhân nam/nữ"*, *"năm nay... tuổi"*, *"vào viện vì lý do"*, *"tiền sử ghi nhận"*...).
  - Bản `condensed_vi` được dùng làm input cho **BM25** và **Translation Provider**.
  - Nhánh **Dense Retrieval** sử dụng chiến lược `head_tail` (lấy 128 tokens đầu + 384 tokens cuối chứa phần kết luận chẩn đoán) để không làm đứt ngữ cảnh.

#### `build_query_variants(query: QueryRecord) -> QueryVariants`

```python
@dataclass
class QueryVariants:
    vi: str
    vi_condensed: str | None
    en: str | None
    zh: str | None
```

Rules:
- Original VI luôn tồn tại.
- EN/ZH translation được cache theo hash của normalized query.
- Translation lỗi ở một language chỉ disable branch đó cho query hiện tại.
- Translation output rỗng hoặc quá ngắn bị xem là invalid.

---

### 7.4. Translation Provider — `src/query/translator.py`

```python
class Translator(Protocol):
    def translate(self, text: str, source_lang: str, target_lang: str) -> str: ...
```

Implementation hỗ trợ:
- `PrecomputedCacheTranslator`: Đọc từ file JSONL cache dịch sẵn (chuẩn bị trước khi submit, 0ms latency, an toàn cho sandbox offline).
- `LocalHfTranslator`: Dùng local model (NLLB-200-distilled / MarianMT) offline.
- `ApiTranslator`: Dùng OpenAI / Gemini API khi cần thử nghiệm chất lượng dịch.
- `PassthroughTranslator`: Tắt dịch, trả về None (ablation study).

MVP không phụ thuộc vào một vendor cụ thể.

---

### 7.5. BM25 Retriever — `src/retrieval/bm25_retriever.py`

```python
class BM25Retriever:
    def search_documents(self, query: str, language: str, top_k: int) -> list[SearchHit]: ...
    def search_chunks(self, query: str, language: str, top_k: int) -> list[SearchHit]: ...
```

Requirements:
- Không search chéo language index.
- Preserve raw BM25 score và rank.
- Return empty list an toàn nếu index không tồn tại hoặc branch không có dữ liệu.

---

### 7.6. Dense Retriever — `src/retrieval/dense_retriever.py`

```python
class DenseRetriever:
    def encode_query(self, query: str) -> np.ndarray: ...
    def search_chunks(self, query: str, language: str, top_k: int) -> list[SearchHit]: ...
    def aggregate_documents(self, chunk_hits: list[SearchHit]) -> list[SearchHit]: ...
```

Default baseline encoder: `intfloat/multilingual-e5-large`.

Requirements:
- Query và corpus embedding phải tự động gắn đúng prefix/template của model (`query: ` và `passage: `).
- Embedding normalization phải nhất quán giữa indexing và query-time.
- `aggregate_documents()` mặc định lấy max dense chunk score theo `doc_id`.

---

### 7.7. Retrieval Orchestrator — `src/retrieval/orchestrator.py`

Mỗi query tạo tối đa 6 retrieval streams:

```text
VI_BM25
VI_DENSE
EN_BM25
EN_DENSE
ZH_BM25
ZH_DENSE
```

Mỗi stream tạo:
- document candidates;
- chunk candidates.

Nếu EN/ZH translation unavailable, stream tương ứng bị bỏ qua và remaining stream vẫn hoạt động bình thường.

---

### 7.8. Weighted Fusion — `src/fusion/weighted_rrf.py`

Default MVP sử dụng **Weighted Reciprocal Rank Fusion (Weighted RRF)** để tránh việc BM25 score và dense cosine score nằm ở hai thang đo khác nhau.

#### Formula

Với candidate `d` xuất hiện trong nhiều ranked list:

$$\text{fusion\_score}(d) = \sum_{i} \frac{\text{weight}_i}{\text{rrf\_k} + \text{rank}_i(d)}$$

Trong đó:
- `weight_i`: trọng số stream.
- `rrf_k`: mặc định `60`.
- Candidate không xuất hiện trong stream có contribution = `0`.
- Deterministic tie-breaking: khi bằng điểm, ưu tiên rank trung bình tốt hơn, sau đó so sánh `item_id`.

#### Functions

```python
def weighted_rrf(
    ranked_lists: dict[str, list[SearchHit]],
    weights: dict[str, float],
    rrf_k: int = 60,
) -> list[FusedHit]: ...
```

Fusion được chạy độc lập cho:
- document candidates;
- chunk candidates.

Output được cắt về `candidate_top_k` trước reranking.

---

### 7.9. Reranker — `src/reranking/reranker.py`

```python
class Reranker(Protocol):
    def rerank_documents(
        self,
        query_variants: QueryVariants,
        candidates: list[FusedHit],
        top_k: int,
    ) -> list[ScoredHit]: ...

    def rerank_chunks(
        self,
        query_variants: QueryVariants,
        candidates: list[FusedHit],
        top_k: int,
    ) -> list[ScoredHit]: ...
```

Rules:
- Candidate language `vi` sử dụng VI query.
- Candidate language `en` ưu tiên EN query; fallback VI nếu EN unavailable.
- Candidate language `zh` ưu tiên ZH query; fallback VI nếu ZH unavailable.
- Reranker chỉ chạy trên candidate set, không scan toàn corpus.
- Preserve `fusion_score` để phục vụ ablation/debugging.
- **Document Reranking Strategy (`document_strategy`):**
  - `"chunk_aggregation"` (Mặc định & Khuyến nghị): Chỉ đưa các Chunk vào Cross-Encoder (vì chunk ngắn < 512 tokens). Điểm của Document được suy ra:
    $$\text{base\_doc\_score}(d) = \max_{c \in d}(\text{rerank\_chunk\_score}(c))$$
    *Tiết kiệm 50% thời gian GPU và giải quyết triệt để vấn đề Document dài vượt 512 tokens.*
  - `"truncate_direct"`: Rerank trực tiếp cặp `(Query, Doc Title + First 512 tokens)`.

---

## 8. DOCUMENT + CHUNK JOINT SCORING & CONSISTENCY

Mục tiêu: document và chunk hỗ trợ nhau thay vì được quyết định hoàn toàn độc lập.

### 8.1. Base Scores & Normalization

Sau reranker:
- `base_doc_score(d)`
- `base_chunk_score(c)`

Scores được normalize về `[0, 1]` bằng Min-Max Normalization trong phạm vi candidate list của từng query trước khi joint scoring.

### 8.2. Final Document Score

$$\text{final\_doc\_score}(d) = \alpha \cdot \text{base\_doc\_score}(d) + \beta \cdot \max_{c \in d}(\text{base\_chunk\_score}(c)) + \gamma \cdot \text{mean}(\text{top\_m\_chunk\_scores}(d))$$

Default:
```text
α = 0.70
β = 0.20
γ = 0.10
m = 3
```

Nếu document không có chunk candidate: `chunk evidence = 0`.

### 8.3. Final Chunk Score

Sau khi có `final_doc_score`:

$$\text{final\_chunk\_score}(c) = \delta \cdot \text{base\_chunk\_score}(c) + \epsilon \cdot \text{final\_doc\_score}(\text{parent\_doc}(c))$$

Default:
```text
δ = 0.85
ε = 0.15
```

### 8.4. Invariants

- $\alpha + \beta + \gamma = 1.0$
- $\delta + \epsilon = 1.0$
- Output cuối cùng được sort descending theo final score.
- Không để một document score thấp kéo chunk mạnh xuống bằng 0; weights phải được tune bằng validation.

### 8.5. Structural Consistency Post-Processing (`src/scoring/consistency.py`)

Cung cấp các chế độ quản lý quan hệ phụ thuộc cha-con ở đầu ra:
- `"independent"` (Mặc định): Cắt ngưỡng độc lập theo điểm F2 của từng danh sách.
- `"enforce_parent_doc"`: Nếu Chunk $c$ vượt qua ngưỡng cutoff để vào bài nộp, Document cha $\text{parent}(c)$ tự động được bổ sung vào danh sách Document.
- `"filter_orphan_chunks"`: Nếu một Document bị loại bởi cutoff, toàn bộ chunk con của nó bị loại bỏ khỏi danh sách nộp.

---

## 9. F2 EVALUATION & CALIBRATION

### 9.1. Metrics

Với mỗi query:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$F_2 = \frac{(1 + 2^2) \cdot \text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$

Macro F2:

$$\text{Macro F}_2 = \frac{1}{|Q|} \sum_{q \in Q} F_2(q)$$

Hệ thống phải tính độc lập:
- `document_precision`, `document_recall`, `document_f2`
- `chunk_precision`, `chunk_recall`, `chunk_f2`

Đồng thời lưu per-query metric để phục vụ error analysis.

### 9.2. Edge Case Guards

- Nếu Ground Truth rỗng và Model không trả về kết quả: $F_2 = 1.0$.
- Nếu Ground Truth rỗng nhưng Model trả về kết quả: $F_2 = 0.0$.
- Nếu Ground Truth có nhãn nhưng Model không trả về gì: $F_2 = 0.0$.

### 9.3. Calibration Objective

Calibration chọn policy/parameter tối đa hóa `Macro F2` trên validation set.

Tuyệt đối không optimize trực tiếp trên test set (chống data leakage).

### 9.4. Supported Cutoff Policies

#### A. Fixed Top-K (Baseline)
- Giữ lại đúng top $K$ items ($K \in [1, K_{\max}]$).

#### B. Global Threshold
- Giữ lại item nếu $\text{score} \ge T$ (Tune $T \in [0.0, 1.0]$ trên validation).

#### C. Relative Threshold
- Giữ lại item nếu $\text{score} \ge \lambda \cdot \text{score}_{\text{top 1}}$ ($\lambda \in [0.1, 1.0]$).
- Phù hợp khi score scale thay đổi theo từng query.

#### D. Score Gap
- Sắp xếp điểm giảm dần: $s_1, s_2, s_3, \dots, s_n$.
- Tìm điểm có độ giảm điểm lớn nhất $s_i - s_{i+1}$ và cắt tại vị trí đó, có ràng buộc $N_{\min} \le i \le N_{\max}$.

### 9.5. Adaptive Cutoff Selection

```python
class CutoffCalibrator:
    def fit(
        self,
        predictions: list[ScoredPrediction],
        ground_truth: list[GroundTruthRecord],
    ) -> CalibrationState: ...

    def apply(
        self,
        scores: list[float],
        state: CalibrationState,
    ) -> int: ...
```

Calibration chạy riêng cho:
- documents;
- chunks.

Policy tốt nhất trên validation được lưu vào `artifacts/experiments/<run_id>/calibration.json`.

---

## 10. PIPELINE ENTRYPOINTS

### 10.1. Build Index

```bash
python -m src.cli.index \
  --config configs/r2ai_mvp.yaml \
  --dataset data/processed
```

Flow:

```text
DatasetAdapter
   ↓
validate corpus
   ↓
Dynamic Chunker (if needed)
   ↓
BM25 indexes (VI / EN / ZH)
   ↓
Dense chunk embeddings (VI / EN / ZH)
   ↓
Vector index + metadata
```

### 10.2. Evaluate Experiment

```bash
python -m src.cli.evaluate \
  --config configs/r2ai_mvp.yaml \
  --split validation
```

Flow:

```text
Queries
  ↓
Query Understanding & Adaptive Routing
  ↓
6 Retrieval Streams
  ↓
Fusion
  ↓
Reranking
  ↓
Joint Scoring
  ↓
Current Cutoff
  ↓
Metrics + Per-query Error Data
```

### 10.3. Calibrate Cutoff

```bash
python -m src.cli.calibrate \
  --config configs/r2ai_mvp.yaml \
  --split validation
```

Output:

```text
artifacts/experiments/<run_id>/calibration.json
```

### 10.4. Predict / Generate Submission

```bash
python -m src.cli.predict \
  --config configs/r2ai_mvp.yaml \
  --split test \
  --calibration artifacts/experiments/<run_id>/calibration.json \
  --export-submission
```

Flow phải load calibration state đã fit từ validation, tuyệt đối không recalibrate trên test.

---

## 11. EXPERIMENT OUTPUT & OBSERVABILITY

Mỗi run tạo một thư mục độc lập:

```text
artifacts/experiments/<run_id>/
├── config.yaml
├── environment.json
├── metrics.json
├── per_query_metrics.csv
├── predictions.jsonl
├── retrieval_debug.jsonl
├── calibration.json
├── error_analysis.csv
└── submission.*
```

### 11.1. `metrics.json`

```json
{
  "documents": {
    "precision": 0.0,
    "recall": 0.0,
    "macro_f2": 0.0
  },
  "chunks": {
    "precision": 0.0,
    "recall": 0.0,
    "macro_f2": 0.0
  }
}
```

### 11.2. `retrieval_debug.jsonl`

Mỗi query phải lưu tối thiểu:
- VI/EN/ZH query variants và variant rút gọn `condensed_vi`.
- Top candidates của từng stream trong 6 streams.
- Fusion score.
- Reranker score.
- Joint score.
- Final cutoff.
- Final selected IDs.

Mục tiêu: một query sai phải truy ngược được nó sai ở translation, retrieval, fusion, reranking hay cutoff.

### 11.3. `error_analysis.csv`

Các cột tối thiểu:

```text
query_id
query_vi
false_negative_docs
false_positive_docs
false_negative_chunks
false_positive_chunks
doc_f2
chunk_f2
selected_doc_count
selected_chunk_count
```

---

## 12. SUBMISSION EXPORTER

### 12.1. Internal Output

Core pipeline luôn trả:

```python
FinalPrediction(
    query_id=...,
    document_ids=[...],
    chunk_ids=[...],
    document_scores={...},
    chunk_scores={...},
)
```

### 12.2. Export Interface

```python
class SubmissionExporter(Protocol):
    def export(
        self,
        predictions: list[FinalPrediction],
        output_path: Path,
    ) -> None: ...
```

MVP có `InternalJsonlExporter` để test end-to-end.

Khi BTC công bố submission schema (CSV, JSON, định dạng cột ngăn cách), chỉ thêm exporter tương ứng; retrieval core hoàn toàn không thay đổi.

---

## 13. VERIFICATION & TESTING

### 13.1. Unit Tests

#### `tests/test_chunker.py`
- `AutoChunker` chia window, overlap và metadata `chunk_index` chính xác.
- Không bỏ sót từ ở cuối document.

#### `tests/test_query_processor.py`
- Unicode/whitespace normalize đúng.
- Không xóa số, ký hiệu y khoa hoặc abbreviation (`HA`, `ĐTĐ`, `mg/dL`, `(+)`).
- Adaptive Router phân biệt đúng query ngắn và query dài.
- Translation failure vẫn giữ VI branch.
- Translation cache trả cùng kết quả cho cùng normalized query.

#### `tests/test_weighted_rrf.py`
- Candidate xuất hiện ở nhiều stream có score cộng dồn đúng.
- Stream weight = 0 không ảnh hưởng kết quả.
- Tie-breaking deterministic.
- Candidate không có trong stream nhận contribution = 0.

#### `tests/test_reranker.py`
- `chunk_aggregation` tổng hợp điểm document từ chunk con chính xác.
- Ghép cặp đúng ngôn ngữ (`vi` $\to$ VI, `en` $\to$ EN, `zh` $\to$ ZH).

#### `tests/test_joint_scoring.py`
- Document có strong chunk evidence được tăng score.
- Chunk kế thừa đúng parent document score.
- Document không có chunk candidate không crash.
- Weight sums được validate ($\alpha+\beta+\gamma=1.0$, $\delta+\epsilon=1.0$).

#### `tests/test_consistency.py`
- `enforce_parent_doc` tự động kéo document cha vào danh sách nộp khi chunk con được chọn.
- `filter_orphan_chunks` loại bỏ chunk mồ côi nếu document cha bị rớt.

#### `tests/test_metrics.py`
- Precision/Recall/F2 khớp hand-calculated examples.
- Query không có prediction.
- Query không có relevant items (negative/distractor query).
- Macro averaging đúng theo query.

#### `tests/test_calibration.py`
- Global threshold chọn parameter có Macro F2 tốt nhất trên validation fixture.
- Relative threshold hoạt động với score scale khác nhau.
- Score-gap luôn tuân thủ `min_results/max_results`.
- Calibration state serialize/load lại không đổi output.

#### `tests/test_dataset_adapter.py`
- Duplicate IDs bị reject.
- Chunk trỏ tới document không tồn tại bị reject.
- Language ngoài `vi/en/zh` bị reject hoặc map theo adapter rule rõ ràng.

---

### 13.2. Integration Test — Multilingual Retrieval

Tạo synthetic corpus nhỏ:

```text
VI: 5 documents
EN: 5 documents
ZH: 5 documents
```

Một Vietnamese query có relevant evidence nằm ở cả VI, EN và ZH.

Expected:
- Có candidate từ đủ 3 language branch.
- Fusion giữ được relevant candidate.
- Reranker không làm mất toàn bộ cross-lingual relevant candidates.
- FinalPrediction trả cả document và chunk IDs.

---

### 13.3. End-to-End Smoke Test

```bash
python -m src.cli.index --config configs/r2ai_mvp.yaml --dataset tests/fixtures/corpus
python -m src.cli.evaluate --config configs/r2ai_mvp.yaml --split validation
python -m src.cli.calibrate --config configs/r2ai_mvp.yaml --split validation
python -m src.cli.predict --config configs/r2ai_mvp.yaml --split test --export-submission
```

Verify:
- Không exception.
- Artifacts được tạo đủ.
- `metrics.json` parse được.
- Calibration không đọc test labels.
- Submission chứa đúng tất cả `query_id` của test set.

---

## 14. PROJECT STRUCTURE DỰ KIẾN

```text
r2ai-retrieval/
├── configs/
│   └── r2ai_mvp.yaml
│
├── src/
│   ├── cli/
│   │   ├── index.py
│   │   ├── evaluate.py
│   │   ├── calibrate.py
│   │   └── predict.py
│   │
│   ├── data/
│   │   ├── schemas.py
│   │   ├── dataset_adapter.py
│   │   └── validators.py
│   │
│   ├── indexing/
│   │   ├── chunker.py             # BaseChunker & AutoChunker
│   │   ├── tokenizers/            # Tokenizer cho VI, EN, ZH
│   │   ├── bm25_index.py
│   │   └── dense_index.py
│   │
│   ├── query/
│   │   ├── query_processor.py     # Normalize & Adaptive Router
│   │   ├── translator.py          # Pluggable translators
│   │   └── translation_cache.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── orchestrator.py
│   │
│   ├── fusion/
│   │   └── weighted_rrf.py
│   │
│   ├── reranking/
│   │   └── reranker.py            # Chunk aggregation & Truncate
│   │
│   ├── scoring/
│   │   ├── joint_scoring.py
│   │   └── consistency.py         # Structural consistency
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # Macro F2 với edge case guard
│   │   ├── calibrator.py
│   │   └── error_analysis.py
│   │
│   └── submission/
│       ├── base.py
│       └── internal_jsonl.py
│
├── tests/
│   ├── fixtures/
│   ├── test_chunker.py
│   ├── test_query_processor.py
│   ├── test_weighted_rrf.py
│   ├── test_reranker.py
│   ├── test_joint_scoring.py
│   ├── test_consistency.py
│   ├── test_metrics.py
│   ├── test_calibration.py
│   ├── test_dataset_adapter.py
│   └── test_e2e_pipeline.py
│
├── artifacts/
│   ├── indexes/
│   ├── cache/
│   └── experiments/
│
└── README.md
```

---

## 15. MVP SCOPE / NON-GOALS

### 15.1. Trong MVP

- Dataset adapter.
- Dynamic Chunking (prechunked, auto, hybrid).
- Query normalization & Medical character protection.
- Adaptive Query Routing (short query vs clinical vignette).
- VI → EN/ZH translation abstraction + cache + offline providers.
- BM25 retrieval với pluggable backend (bm25s / rank_bm25) và multilingual tokenizers.
- Dense retrieval với pluggable backend (faiss / numpy) và instruction prefix awareness.
- VI/EN/ZH parallel retrieval streams.
- Weighted RRF fusion với deterministic tie-breaking.
- Configurable reranker với cơ chế Chunk Aggregation cho document dài.
- Document/chunk joint scoring.
- Structural consistency enforcement (`independent`, `enforce_parent_doc`, `filter_orphan_chunks`).
- Precision / Recall / Macro F2 evaluator với edge case guards.
- F2 calibration (Fixed top-k, Global threshold, Relative threshold, Score gap).
- Adaptive cutoff.
- Experiment artifacts + per-query debug trajectory + error analysis.
- Submission exporter abstraction.

### 15.2. Ngoài MVP hiện tại

Không triển khai trong phase đầu:
- Answer generation / full RAG.
- MCP server.
- AI Agents / Agent loops.
- Agentic/Adaptive RAG multi-hop orchestration.
- HyDE / Query2Doc.
- Fine-tuning retriever.
- Fine-tuning reranker.
- Hard-negative mining.
- ColBERT / late interaction index.
- Medical knowledge graph.
- Web UI.

Các phần trên chỉ được thêm sau khi baseline cho thấy rõ bottleneck qua error analysis.

---

## 16. EXTENSION POINTS SAU MVP

Kiến trúc phải cho phép thêm mà không thay đổi public contract của pipeline:

### 16.1. Query Expansion

```text
QueryVariants
  ├── original
  ├── translated
  ├── terminology expansion
  ├── HyDE
  └── Query2Doc
```

Các query variant mới chỉ tạo thêm retrieval stream rồi đưa vào cùng `WeightedFusion`.

### 16.2. Fine-tuned Retriever

`DenseRetriever` interface giữ nguyên; chỉ đổi encoder/index.

### 16.3. Fine-tuned Reranker

`Reranker` interface giữ nguyên; model mới có thể load checkpoint fine-tuned từ hard negatives.

### 16.4. Specialist Biomedical Models

Có thể thêm stream chuyên biệt theo language/domain:

```text
EN biomedical retriever
ZH medical retriever
VI medical retriever
```

Sau đó merge vào Weighted Fusion như các stream khác.

### 16.5. Learned Cutoff

Sau khi có đủ labeled validation data, `CutoffCalibrator` có thể thay rule-based policy bằng classifier/regressor dự đoán số lượng relevant results theo từng query mà không thay downstream contract.

---

## 17. ACCEPTANCE CRITERIA

MVP được xem là hoàn thành khi đáp ứng toàn bộ các điều kiện sau:

1. **Data**
   - Load được documents, chunks, queries và ground truth qua adapter.
   - Hỗ trợ cả dữ liệu đã chia chunk sẵn hoặc tự động chia chunk qua `AutoChunker`.
   - Validate được quan hệ `chunk_id -> doc_id`.

2. **Multilingual Query**
   - Một Vietnamese query tạo được VI + EN + ZH variants khi translation hoạt động.
   - Pipeline phân loại và xử lý thích ứng được cả query ngắn và ca lâm sàng dài.
   - Pipeline vẫn chạy nếu EN hoặc ZH translation fail.

3. **Retrieval**
   - Chạy được BM25 + Dense cho cả document/chunk candidates trên 6 streams.
   - Hỗ trợ chuyển đổi giữa backend FAISS / NumPy và BM25s / Rank-BM25 qua cấu hình.
   - Fusion được cả 6 stream và trả candidate set deterministic.

4. **Reranking**
   - Rerank được candidate list mà không scan lại toàn corpus.
   - Áp dụng thành công chiến lược `chunk_aggregation` để rerank document mà không bị giới hạn 512 tokens.

5. **Joint Scoring & Consistency**
   - Document score nhận evidence từ chunks.
   - Chunk score nhận context từ parent document.
   - Quản lý được tính nhất quán giữa danh sách document và chunk theo chính sách đã chọn.

6. **Evaluation**
   - Tính đúng Precision, Recall, F2 và Macro F2, xử lý đúng các trường hợp mẫu rỗng.
   - Có per-query error analysis và file debug chi tiết từng bước.

7. **Calibration**
   - So sánh ít nhất fixed top-k, global threshold, relative threshold và score-gap trên validation.
   - Chọn và persist policy có Macro F2 validation cao nhất trong các policy đã thử.

8. **Submission**
   - Sinh được `FinalPrediction` cho 100% test queries.
   - Có exporter độc lập với retrieval core.

9. **Reproducibility**
   - Experiment lưu config, metrics, calibration state và predictions.
   - Cùng config + seed tạo cùng thứ tự kết quả sau tie-breaking.

---

## 18. DEFINITION OF FIRST SUCCESSFUL BASELINE

Baseline đầu tiên không cần đạt điểm cao nhất. Nó được xem là thành công khi toàn bộ vòng lặp nghiên cứu chạy end-to-end:

```text
Dataset
  ↓
Index (Chunker + BM25s + Dense FAISS/NumPy)
  ↓
Vietnamese Query
  ↓
Adaptive Router (Short / Long Case)
  ↓
VI / EN / ZH (Cache / Offline HF / API)
  ↓
BM25 + Dense (6 Streams)
  ↓
Weighted RRF
  ↓
Reranker (Chunk Aggregation)
  ↓
Document + Chunk Joint Scoring
  ↓
Structural Consistency
  ↓
F2 Evaluation
  ↓
Calibration
  ↓
Adaptive Cutoff
  ↓
Submission Exporter
  ↓
Error Analysis
  ↓
Next Experiment
```

Giá trị chính của MVP là biến bài toán từ:

> “Thử một model xem có tốt không?”

thành:

> “Thay chính xác một component, chạy lại cùng benchmark, đo Macro F2, tìm false negative/false positive và biết bước tiếp theo cần tối ưu ở đâu.”

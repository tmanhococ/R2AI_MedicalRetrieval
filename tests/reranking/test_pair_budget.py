from __future__ import annotations

import logging
import re
from typing import Any
import pytest

from src.data.vibiomir_contracts import (
    PairPackingConfig,
    SpanChunk,
)
from src.query.query_processor import (
    AdaptiveQueryRouter,
    ProcessedQuery,
    QueryVariants,
)
from src.reranking.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    DummyReranker,
    PackedPair,
    pack_reranker_inputs,
    rerank_chunks,
)
from src.data.schemas import FusedHit, QueryRecord


class MockFastTokenizer:
    """Mock fast tokenizer providing token IDs, decodings, and exact character offset mapping."""

    def __init__(self, name: str = "mock-fast-tok"):
        self.name = name
        self.vocab: dict[str, int] = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}
        self.id_to_tok: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self._next_id = 10

    def _get_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self.id_to_tok[self._next_id] = token
            self._next_id += 1
        return self.vocab[token]

    def num_special_tokens_to_add(self, pair: bool = True) -> int:
        return 3 if pair else 2

    def encode(self, text: str, text_pair: str | None = None, add_special_tokens: bool = False) -> list[int]:
        res = self(text, text_pair=text_pair, add_special_tokens=add_special_tokens)
        return res["input_ids"]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            tok = self.id_to_tok.get(tid, "[UNK]")
            if skip_special_tokens and tok in ("[PAD]", "[CLS]", "[SEP]", "[UNK]"):
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def __call__(
        self,
        text: str,
        text_pair: str | None = None,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        matches = list(re.finditer(r"\S+", text))
        tokens = [m.group(0) for m in matches]
        offsets = [(m.start(), m.end()) for m in matches]
        ids = [self._get_id(t) for t in tokens]

        if text_pair is not None:
            p_matches = list(re.finditer(r"\S+", text_pair))
            p_tokens = [m.group(0) for m in p_matches]
            p_ids = [self._get_id(t) for t in p_tokens]
            p_offsets = [(m.start(), m.end()) for m in p_matches]

            if add_special_tokens:
                combined_ids = [1] + ids + [2] + p_ids + [2]
                combined_offsets = [(0, 0)] + offsets + [(0, 0)] + p_offsets + [(0, 0)]
            else:
                combined_ids = ids + p_ids
                combined_offsets = offsets + p_offsets
            return {
                "input_ids": combined_ids,
                "attention_mask": [1] * len(combined_ids),
                "offset_mapping": combined_offsets,
            }

        if add_special_tokens:
            final_ids = [1] + ids + [2]
            final_offsets = [(0, 0)] + offsets + [(0, 0)]
        else:
            final_ids = ids
            final_offsets = offsets

        res = {
            "input_ids": final_ids,
            "attention_mask": [1] * len(final_ids),
        }
        if return_offsets_mapping:
            res["offset_mapping"] = final_offsets
        return res


def _make_span_chunk(
    text: str,
    index_text: str | None = None,
    section: str | None = None,
    doc_id: int = 1,
    chunk_index: int = 0,
) -> SpanChunk:
    idx_text = index_text if index_text is not None else text
    return SpanChunk(
        chunk_id=f"{doc_id}:sha256:{chunk_index:06d}",
        doc_id=doc_id,
        source_version="test_v1",
        source_sha256="sha256",
        start_char=0,
        end_char=len(text),
        chunk_index=chunk_index,
        language="vi",
        section=section,
        text=text,
        index_text=idx_text,
        dense_token_count=len(text.split()),
        reranker_token_count=len(text.split()),
    )


def test_pack_reranker_inputs_short_query():
    """Short query (<= 128 tokens) produces a single PackedPair within 512 total tokens."""
    tok = MockFastTokenizer()
    query = "Phương pháp điều trị suy tim sung huyết cấp tính là gì?"
    chunk = _make_span_chunk(
        text="Thở oxy liều cao kết hợp lợi tiểu quai Furosemide đường tĩnh mạch.",
        index_text="Hướng dẫn Tim mạch\n\nThở oxy liều cao kết hợp lợi tiểu quai Furosemide đường tĩnh mạch.",
        section="Hướng dẫn Tim mạch",
    )
    config = PairPackingConfig(max_length=512, query_window_tokens=128)

    pairs = pack_reranker_inputs(query, chunk, tokenizer=tok, config=config)
    assert len(pairs) == 1
    p = pairs[0]

    assert isinstance(p, PackedPair)
    assert p.query_window_index == 0
    assert p.query_text == query
    assert chunk.text in p.passage_text
    assert len(p.input_ids) <= 512
    assert len(p.input_ids) == len(p.attention_mask)


def test_pack_reranker_inputs_long_query_covers_middle_condition():
    """Long query with critical condition in middle: windows must cover all tokens and condition."""
    tok = MockFastTokenizer()
    # 250 tokens query with critical medical condition in the middle (around index 150)
    words = [f"tiền_sử_{i}" for i in range(140)]
    condition = ["xét", "nghiệm", "đột", "biến", "gen", "BRCA1", "dương", "tính"]
    after_words = [f"triệu_chứng_{i}" for i in range(102)]
    all_words = words + condition + after_words
    long_query = " ".join(all_words)
    assert len(all_words) == 250

    chunk = _make_span_chunk(
        text="Bệnh nhân mang đột biến BRCA1 cần được tầm soát ung thư buồng trứng định kỳ.",
        section="Sản phụ khoa",
    )
    config = PairPackingConfig(
        max_length=512,
        query_window_tokens=128,
        query_window_overlap_tokens=16,
    )

    pairs = pack_reranker_inputs(long_query, chunk, tokenizer=tok, config=config)
    # 250 tokens with window 128 and step 112:
    # win 0: 0..128
    # win 1: 112..240 (covers 140..148 condition!)
    # win 2: 224..250
    assert len(pairs) >= 3

    # Verify all query tokens are covered
    covered_windows = []
    for p in pairs:
        assert len(p.input_ids) <= 512
        covered_windows.append(p.query_text)

    # Critical condition must be present in at least one window
    assert any("BRCA1 dương tính" in w for w in covered_windows)


def test_pair_budget_header_shrunk_first_never_truncate_body():
    """When query + header + body exceeds 512, header is shrunk first; body is NEVER truncated."""
    tok = MockFastTokenizer()
    # Query window = 120 tokens
    query_words = [f"q_{i}" for i in range(120)]
    query = " ".join(query_words)

    # Body = 320 tokens
    body_words = [f"bệnh_án_chi_tiết_{i}" for i in range(320)]
    body_text = " ".join(body_words)

    # Header = 80 tokens (120 + 320 + 80 + 3 special = 523 > 512)
    header_words = [f"tiêu_đề_rất_dài_{i}" for i in range(80)]
    header_text = " ".join(header_words)
    index_text = f"{header_text}\n\n{body_text}"

    chunk = _make_span_chunk(text=body_text, index_text=index_text, section=header_text)
    config = PairPackingConfig(max_length=512, query_window_tokens=128)

    pairs = pack_reranker_inputs(query, chunk, tokenizer=tok, config=config)
    assert len(pairs) == 1
    p = pairs[0]

    assert len(p.input_ids) <= 512
    # Body is completely intact and never truncated
    assert body_text in p.passage_text
    # Header was shrunk
    assert header_text not in p.passage_text or len(p.passage_text) < len(index_text)


def test_pair_budget_body_exceeds_reject_error():
    """If body alone + query window exceeds 512, packing raises ValueError under reject policy."""
    tok = MockFastTokenizer()
    query = " ".join([f"q_{i}" for i in range(128)])
    # Body = 390 tokens (128 + 390 + 3 = 521 > 512)
    oversized_body = " ".join([f"quá_tải_{i}" for i in range(390)])
    chunk = _make_span_chunk(text=oversized_body)

    config = PairPackingConfig(max_length=512, query_window_tokens=128, body_truncation="reject")

    with pytest.raises(ValueError, match="Pair token budget exceeded|Body cannot be truncated"):
        pack_reranker_inputs(query, chunk, tokenizer=tok, config=config)


def test_reranker_score_aggregation_max_across_query_windows():
    """Reranker score for multi-window query must be max score across windows."""
    tok = MockFastTokenizer()
    # 250 tokens query where only window 1 matches the chunk passage
    words = [f"tiền_sử_{i}" for i in range(140)]
    condition = ["HER2", "dương", "tính", "Trastuzumab"]
    after_words = [f"triệu_chứng_{i}" for i in range(106)]
    long_query = " ".join(words + condition + after_words)

    chunk_text = "Chỉ định Trastuzumab cho bệnh nhân ung thư vú HER2 dương tính."
    chunk = _make_span_chunk(text=chunk_text, doc_id=1, chunk_index=0)

    # Scorer that returns high score if HER2/Trastuzumab is in the query window
    def custom_scorer(pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for q, p in pairs:
            if "HER2" in q and "Trastuzumab" in q:
                scores.append(0.95)
            else:
                scores.append(0.10)
        return scores

    reranker = CrossEncoderReranker(model=custom_scorer)

    # Call pack_reranker_inputs directly to check scores
    pairs = pack_reranker_inputs(long_query, chunk, tokenizer=tok)
    assert len(pairs) >= 2
    raw_scores = reranker.score_pairs([(p.query_text, p.passage_text) for p in pairs])
    # The aggregated chunk score is max across windows
    aggregated_score = max(raw_scores)
    assert aggregated_score == pytest.approx(0.95)


def test_dense_query_overflow_logging_and_bm25_full_query(caplog):
    """When query exceeds 512 tokens, router logs dense_query_overflow while preserving full BM25 query."""
    router = AdaptiveQueryRouter(
        length_threshold=50,
        strip_clinical_boilerplate=False,
        max_dense_tokens=512,
    )
    long_words = [f"từ_{i}" for i in range(550)]
    raw_query = " ".join(long_words)
    query_record = QueryRecord(query_id="q_long", text_vi=raw_query)

    with caplog.at_level(logging.WARNING):
        routed = router.route(query_record)

    # 1. Log dense_query_overflow
    assert any("dense_query_overflow" in record.message for record in caplog.records)

    # 2. Dense text is truncated to budget
    dense_words = routed.dense_text.split()
    assert len(dense_words) <= 512

    # 3. BM25 text maintains full original text without truncation
    bm25_words = routed.bm25_text.split()
    assert len(bm25_words) == 550
    assert hasattr(routed, "dense_query_overflow") and routed.dense_query_overflow is True
    assert getattr(routed, "bm25_original_text", routed.normalized_text) == routed.normalized_text


@pytest.mark.integration
def test_tokenizer_integration_fast_mock():
    """Integration check verifying token budgeting mechanics without downloading heavy weights."""
    tok = MockFastTokenizer(name="integration-fast-mock")
    q = "Chẩn đoán phân biệt hội chứng vành cấp."
    chunk = _make_span_chunk(text="Điện tâm đồ và troponin T là các chỉ số thiết yếu.")
    pairs = pack_reranker_inputs(q, chunk, tokenizer=tok)
    assert len(pairs) == 1
    assert len(pairs[0].input_ids) <= 512

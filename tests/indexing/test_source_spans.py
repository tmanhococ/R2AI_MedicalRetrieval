from __future__ import annotations

import hashlib
import re
from typing import Any
import pytest

from src.data.vibiomir_contracts import (
    ExtractedBlock,
    ExtractedDocument,
    SpanChunk,
    ChunkConfig,
)
from src.indexing.chunker import (
    chunk_source_document,
    migrate_legacy_chunk,
    migrate_legacy_chunks,
)


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


def _make_document(
    doc_id: int,
    raw_blocks: list[tuple[str, str | None, str]],
    title: str | None = None,
    language: str = "vi",
) -> ExtractedDocument:
    blocks: list[ExtractedBlock] = []
    current_offset = 0
    for i, (kind, section, text) in enumerate(raw_blocks):
        if i > 0:
            current_offset += 2  # "\n\n" separator
        start_char = current_offset
        end_char = start_char + len(text)
        blocks.append(
            ExtractedBlock(
                block_index=i,
                kind=kind,
                start_char=start_char,
                end_char=end_char,
                section=section,
            )
        )
        current_offset = end_char

    source_text = "\n\n".join(t for _, _, t in raw_blocks)
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    raw_sha256 = hashlib.sha256(b"raw-bytes").hexdigest()

    return ExtractedDocument(
        doc_id=doc_id,
        source_version="test_v1",
        source_text=source_text,
        source_sha256=source_sha256,
        raw_sha256=raw_sha256,
        title=title,
        language=language,
        blocks=tuple(blocks),
        quality_flags=(),
    )


def test_exact_span_identity_basic():
    """Verify core span invariants: exact slice matching, sha256 matching, and chunk_id format."""
    doc = _make_document(
        doc_id=101,
        raw_blocks=[
            ("paragraph", "Tong quan", "Bệnh nhân cần được chẩn đoán sớm và điều trị kịp thời."),
            ("paragraph", "Tong quan", "Các phương pháp cận lâm sàng bao gồm X-quang và siêu âm."),
            ("paragraph", "Dieu tri", "Kháng sinh đường tĩnh mạch được khuyến cáo trong 48 giờ đầu."),
        ],
        title="Hướng dẫn chẩn đoán và điều trị",
    )
    tok = MockFastTokenizer()
    config = ChunkConfig(body_max_tokens=320, contextual_header_max_tokens=32)

    chunks = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok, config=config))
    assert len(chunks) >= 1

    for chunk in chunks:
        # Invariant 1: exact slice equality
        assert chunk.text == doc.source_text[chunk.start_char : chunk.end_char]
        # Invariant 2: source sha256 and version
        assert chunk.source_sha256 == doc.source_sha256
        assert chunk.source_version == doc.source_version
        assert chunk.doc_id == doc.doc_id
        # Invariant 3: deterministic chunk_id format
        expected_chunk_id = f"{doc.doc_id}:{doc.source_sha256}:{chunk.chunk_index:06d}"
        assert chunk.chunk_id == expected_chunk_id
        assert chunk.dense_token_count <= config.body_max_tokens
        assert chunk.reranker_token_count <= config.body_max_tokens


def test_exact_span_combining_unicode():
    """Combining Unicode and multi-byte Vietnamese characters must preserve exact slicing without offset skew."""
    # Decomposed characters: e + combining circumflex (\u0302) + combining acute (\u0301) = ế
    decomposed_e = "e\u0302\u0301"
    text1 = f"B{decomposed_e}nh vi{decomposed_e}n K trung uong tuyen tren."
    text2 = "Triệu chứng khó thở dữ dội kèm theo ho ra máu tươi."
    doc = _make_document(
        doc_id=202,
        raw_blocks=[
            ("paragraph", "Lam sang", text1),
            ("paragraph", "Lam sang", text2),
        ],
    )
    tok = MockFastTokenizer()
    chunks = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok))

    for chunk in chunks:
        assert chunk.text == doc.source_text[chunk.start_char : chunk.end_char]
        assert decomposed_e in chunk.text or text2 in chunk.text


def test_repeated_paragraphs_different_offsets():
    """Identical paragraphs appearing in multiple blocks must resolve to distinct, accurate offsets."""
    identical_text = "Phác đồ điều trị tiêu chuẩn gồm Amlodipine 5mg mỗi ngày."
    other_text = "Bệnh nhân có tiền sử đái tháo đường type 2 kéo dài mười năm."
    doc = _make_document(
        doc_id=303,
        raw_blocks=[
            ("paragraph", "Khoa A", identical_text),
            ("paragraph", "Khoa B", other_text),
            ("paragraph", "Khoa C", identical_text),
        ],
    )
    tok = MockFastTokenizer()
    # Force 1 block per chunk by limiting body tokens to 15
    config = ChunkConfig(body_max_tokens=15)
    chunks = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok, config=config))

    assert len(chunks) == 3
    c0 = chunks[0]
    c2 = chunks[2]
    assert c0.text == identical_text
    assert c2.text == identical_text
    # Distinct character offsets
    assert c0.start_char != c2.start_char
    assert c0.end_char != c2.end_char
    assert c0.start_char == doc.blocks[0].start_char
    assert c2.start_char == doc.blocks[2].start_char
    assert doc.source_text[c0.start_char : c0.end_char] == identical_text
    assert doc.source_text[c2.start_char : c2.end_char] == identical_text


def test_medical_symbols_preservation():
    """Specific medical symbols (HER2+, TNF-α, CD4+/CD8+, 10^9/L) must not be corrupted or offset shifted."""
    med_text = "Xét nghiệm HER2(+) và định lượng TNF-α tăng. Tỷ lệ CD4+/CD8+ là 1.2, bạch cầu 10^9/L."
    doc = _make_document(
        doc_id=404,
        raw_blocks=[("paragraph", "Xet nghiem", med_text)],
    )
    tok = MockFastTokenizer()
    chunks = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok))

    assert len(chunks) == 1
    assert chunks[0].text == med_text
    assert chunks[0].text == doc.source_text[chunks[0].start_char : chunks[0].end_char]
    for sym in ["HER2(+)", "TNF-α", "CD4+/CD8+", "10^9/L"]:
        assert sym in chunks[0].text


def test_long_paragraph_fallback_splitting_with_overlap():
    """Paragraph exceeding body limit (320 tokens) must be split via offset mapping with overlap (64 tokens)."""
    # Create 400 words
    words = [f"từ_{i}" for i in range(400)]
    long_text = " ".join(words)
    doc = _make_document(
        doc_id=505,
        raw_blocks=[("paragraph", "Noi dung dai", long_text)],
    )
    tok = MockFastTokenizer()
    config = ChunkConfig(body_max_tokens=320, long_block_overlap_tokens=64)

    chunks = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok, config=config))
    assert len(chunks) >= 2

    # Every subchunk must strictly satisfy exact slice invariant
    for c in chunks:
        assert c.text == doc.source_text[c.start_char : c.end_char]
        assert c.dense_token_count <= 320
        assert c.reranker_token_count <= 320

    # Verify overlap exists between chunk 0 and chunk 1
    c0_words = set(c0 := chunks[0].text.split())
    c1_words = set(c1 := chunks[1].text.split())
    overlap = c0_words & c1_words
    assert len(overlap) > 0, "Split chunks must overlap"

    # Verify start of first chunk is block start and end of last chunk is block end
    assert chunks[0].start_char == doc.blocks[0].start_char
    assert chunks[-1].end_char == doc.blocks[0].end_char


def test_span_union_covers_all_content_blocks():
    """Span union must cover all blocks without gaps or dropping content blocks."""
    blocks_data = [
        ("heading", "Muc 1", "I. Giới thiệu"),
        ("paragraph", "Muc 1", "Ung thư biểu mô tế bào gan là bệnh lý ác tính phổ biến."),
        ("heading", "Muc 2", "II. Triệu chứng"),
        ("paragraph", "Muc 2", "Vàng da, sụt cân nhanh chóng, đau tức hạ sườn phải."),
    ]
    doc = _make_document(doc_id=606, raw_blocks=blocks_data)
    tok = MockFastTokenizer()
    config = ChunkConfig(body_max_tokens=50, drop_short_blocks=False)

    chunks = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok, config=config))
    assert len(chunks) >= 2

    for b in doc.blocks:
        covered = any(
            c.start_char <= b.start_char and c.end_char >= b.end_char
            for c in chunks
        )
        assert covered, f"Block {b.block_index} ({b.kind}) was not covered by any chunk span"


def test_title_heading_only_in_index_text_never_in_text():
    """Title and section headings must appear ONLY in index_text, NEVER in text (source slice)."""
    doc = _make_document(
        doc_id=707,
        raw_blocks=[
            ("paragraph", "Phần Hồi sức", "Bệnh nhân suy hô hấp cấp cần thở máy xâm lấn ngay."),
        ],
        title="Cấp cứu nội khoa toàn thư",
    )
    tok = MockFastTokenizer()
    config = ChunkConfig(body_max_tokens=320, contextual_header_max_tokens=32)

    chunks = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok, config=config))
    assert len(chunks) == 1
    c = chunks[0]

    # chunk.text is pure source text
    assert c.text == "Bệnh nhân suy hô hấp cấp cần thở máy xâm lấn ngay."
    assert c.text == doc.source_text[c.start_char : c.end_char]
    assert "Cấp cứu nội khoa toàn thư" not in c.text

    # index_text contains the contextual header prefix
    assert "Cấp cứu nội khoa toàn thư" in c.index_text
    assert c.index_text.endswith(c.text)
    assert c.index_text != c.text


def test_deterministic_chunking():
    """Consecutive runs on identical input produce identical chunk IDs, offsets, and content."""
    doc = _make_document(
        doc_id=808,
        raw_blocks=[
            ("paragraph", "Sec A", "Đoạn văn thứ nhất để kiểm tra tính tất định."),
            ("paragraph", "Sec A", "Đoạn văn thứ hai cùng phần kiểm tra tính tất định."),
        ],
        title="Tiêu đề thử nghiệm",
    )
    tok = MockFastTokenizer()

    run1 = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok))
    run2 = list(chunk_source_document(doc, dense_tokenizer=tok, reranker_tokenizer=tok))

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.start_char == c2.start_char
        assert c1.end_char == c2.end_char
        assert c1.text == c2.text
        assert c1.index_text == c2.index_text


def test_legacy_chunk_migration_unique_exact_occurrence():
    """Legacy chunk without provenance metadata matching uniquely in source is preserved."""
    doc = _make_document(
        doc_id=909,
        raw_blocks=[
            ("paragraph", "A", "Mở đầu tài liệu nghiên cứu lâm sàng."),
            ("paragraph", "B", "Đoạn văn cần di trú sang mô hình span chunk."),
            ("paragraph", "C", "Kết luận cuối cùng của nghiên cứu."),
        ],
    )
    legacy = {
        "doc_id": "909",
        "text": "Đoạn văn cần di trú sang mô hình span chunk.",
        "chunk_index": 0,
        "metadata": {},
    }
    tok = MockFastTokenizer()
    migrated = migrate_legacy_chunk(legacy, doc, dense_tokenizer=tok, reranker_tokenizer=tok)

    assert migrated is not None
    assert isinstance(migrated, SpanChunk)
    assert migrated.text == legacy["text"]
    assert migrated.text == doc.source_text[migrated.start_char : migrated.end_char]
    assert migrated.source_sha256 == doc.source_sha256
    assert migrated.doc_id == 909


def test_legacy_chunk_migration_ambiguous_quarantine_or_regenerate():
    """Legacy chunk matching multiple times (or zero times) is quarantined or regenerated."""
    repeated_text = "Đoạn văn bị lặp lại nhiều lần trong văn bản."
    doc = _make_document(
        doc_id=1010,
        raw_blocks=[
            ("paragraph", "A", repeated_text),
            ("paragraph", "B", "Đoạn giữa."),
            ("paragraph", "C", repeated_text),
        ],
    )
    legacy_repeated = {
        "doc_id": "1010",
        "text": repeated_text,
        "chunk_index": 0,
        "metadata": {},
    }
    tok = MockFastTokenizer()

    # 1. on_ambiguous="quarantine" -> returns None / quarantined
    quarantined = migrate_legacy_chunk(
        legacy_repeated, doc, dense_tokenizer=tok, reranker_tokenizer=tok, on_ambiguous="quarantine"
    )
    assert quarantined is None

    # 2. on_ambiguous="regenerate" -> returns fresh chunks covering document
    result = migrate_legacy_chunks(
        [legacy_repeated], doc, dense_tokenizer=tok, reranker_tokenizer=tok, on_ambiguous="regenerate"
    )
    assert len(result.regenerated_chunks) >= 2
    for rc in result.regenerated_chunks:
        assert rc.text == doc.source_text[rc.start_char : rc.end_char]

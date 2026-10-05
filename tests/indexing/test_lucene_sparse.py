from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from src.data.phase2_contracts import IndexManifest, StageManifest
from src.indexing.lexical_tokens import (
    encode_token,
    encode_tokens,
    tokenize_and_encode,
    tokenize_lexical,
)
from src.indexing.lucene import LexicalRecord, LuceneSparseBackend


def test_clinical_symbols_and_negation_survive_tokenization_and_hash_encoding():
    """Preserve clinical symbols, +/- signs, units, drug patterns, and negation tokens."""
    # Vietnamese with clinical symbols, units, and negation
    vi_text = "Bệnh nhân HER2+ không sốt, điều trị 50mg/ml, COVID-19 (-)"
    vi_tokens = tokenize_lexical(vi_text, language="vi")
    assert "her2+" in vi_tokens
    assert "không" in vi_tokens
    assert "50mg/ml" in vi_tokens
    assert "covid-19" in vi_tokens
    assert "-" in vi_tokens

    # English with negation, units, and clinical signs
    en_text = "No fever, non-small cell lung carcinoma with Ca2+ 2.5mmol/l, ± 10mg"
    en_tokens = tokenize_lexical(en_text, language="en")
    assert "no" in en_tokens
    assert "non-small" in en_tokens
    assert "ca2+" in en_tokens
    assert "2.5mmol/l" in en_tokens
    assert "10mg" in en_tokens
    assert "±" in en_tokens

    # Chinese with negation and clinical markers
    zh_text = "患者无发热，非小细胞肺癌，HER2+，给予 100mg"
    zh_tokens = tokenize_lexical(zh_text, language="zh")
    assert "无" in zh_tokens
    assert "非" in zh_tokens
    assert "her2+" in zh_tokens
    assert "100mg" in zh_tokens

    # Safe term encoding verification: t + sha256(utf8)
    for tok in ["her2+", "không", "no", "无", "50mg/ml"]:
        enc = encode_token(tok)
        expected = "t" + hashlib.sha256(tok.lower().encode("utf-8")).hexdigest()
        assert enc == expected
        # Must be safe alphanumeric (t + 64 hex chars = 65 chars total)
        assert len(enc) == 65
        assert enc.startswith("t")
        assert enc[1:].isalnum()

    # Text encoding produces whitespace-delimited safe tokens
    encoded_str = tokenize_and_encode(vi_text, language="vi")
    parts = encoded_str.split()
    assert len(parts) == len(vi_tokens)
    for p in parts:
        assert p.startswith("t")
        assert len(p) == 65


def test_multilingual_query_document_alignment_vi_en_zh_und(tmp_path: Path):
    """Ensure query and documents align across VI, EN, ZH, and und partitions."""
    backend = LuceneSparseBackend()

    # 1. Vietnamese test
    vi_records = [
        LexicalRecord(
            external_id="doc_vi_1",
            official_doc_id=101,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Bệnh nhân không sốt, điều trị 50mg", "vi"),
        ),
        LexicalRecord(
            external_id="doc_vi_2",
            official_doc_id=102,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Bệnh nhân sốt cao, điều trị kháng sinh", "vi"),
        ),
    ]
    p_vi = backend.build_partition(vi_records, tmp_path / "part_vi")
    idx_vi = backend.assemble([tmp_path / "part_vi"], tmp_path / "idx_vi")

    hits_vi = backend.search("không sốt", language="vi", item_type="document", top_k=5, index_path=tmp_path / "idx_vi")
    assert len(hits_vi) >= 1
    assert hits_vi[0].id == "doc_vi_1"
    assert hits_vi[0].score > 0

    # 2. English test
    en_records = [
        LexicalRecord(
            external_id="doc_en_1",
            official_doc_id=201,
            language="en",
            item_type="document",
            encoded_contents=tokenize_and_encode("Patient with HER2+ non-small cell lung cancer", "en"),
        ),
        LexicalRecord(
            external_id="doc_en_2",
            official_doc_id=202,
            language="en",
            item_type="document",
            encoded_contents=tokenize_and_encode("Hypertension management with ACE inhibitors", "en"),
        ),
    ]
    p_en = backend.build_partition(en_records, tmp_path / "part_en")
    idx_en = backend.assemble([tmp_path / "part_en"], tmp_path / "idx_en")

    hits_en = backend.search("HER2+ non-small cancer", language="en", item_type="document", top_k=5, index_path=tmp_path / "idx_en")
    assert len(hits_en) >= 1
    assert hits_en[0].id == "doc_en_1"
    assert hits_en[0].score > 0

    # 3. Chinese test
    zh_records = [
        LexicalRecord(
            external_id="doc_zh_1",
            official_doc_id=301,
            language="zh",
            item_type="document",
            encoded_contents=tokenize_and_encode("患者无发热，给予 100mg 阿司匹林", "zh"),
        ),
        LexicalRecord(
            external_id="doc_zh_2",
            official_doc_id=302,
            language="zh",
            item_type="document",
            encoded_contents=tokenize_and_encode("患者高热，给予抗生素治疗", "zh"),
        ),
    ]
    p_zh = backend.build_partition(zh_records, tmp_path / "part_zh")
    idx_zh = backend.assemble([tmp_path / "part_zh"], tmp_path / "idx_zh")

    hits_zh = backend.search("无发热 100mg", language="zh", item_type="document", top_k=5, index_path=tmp_path / "idx_zh")
    assert len(hits_zh) >= 1
    assert hits_zh[0].id == "doc_zh_1"
    assert hits_zh[0].score > 0

    # 4. Undetermined (und) test
    und_records = [
        LexicalRecord(
            external_id="doc_und_1",
            official_doc_id=401,
            language="und",
            item_type="document",
            encoded_contents=tokenize_and_encode("Clinical trial COVID-19 5.5mmol/L result", "und"),
        ),
    ]
    p_und = backend.build_partition(und_records, tmp_path / "part_und")
    idx_und = backend.assemble([tmp_path / "part_und"], tmp_path / "idx_und")

    hits_und = backend.search("COVID-19 5.5mmol/L", language="und", item_type="document", top_k=5, index_path=tmp_path / "idx_und")
    assert len(hits_und) == 1
    assert hits_und[0].id == "doc_und_1"


def test_real_sparse_hits_outside_pilot_and_ids_above_int32(tmp_path: Path):
    """Test official doc IDs above int32 max (2^31 - 1 = 2147483647) and outside pilot range."""
    backend = LuceneSparseBackend()
    large_id = 5_000_000_000  # Well beyond int32
    records = [
        LexicalRecord(
            external_id=str(large_id),
            official_doc_id=large_id,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Bệnh hiếm gặp biểu hiện lâm sàng đặc biệt", "vi"),
        ),
    ]
    part = backend.build_partition(records, tmp_path / "part_large")
    manifest = backend.assemble([tmp_path / "part_large"], tmp_path / "idx_large")

    hits = backend.search("lâm sàng", language="vi", item_type="document", top_k=5, index_path=tmp_path / "idx_large")
    assert len(hits) == 1
    assert hits[0].doc_id == str(large_id)
    assert int(hits[0].doc_id) == large_id
    assert hits[0].id == str(large_id)
    assert hits[0].score > 0


def test_partitions_with_different_local_df_assemble_into_global_stats(tmp_path: Path):
    """Partitions with different local DF assemble into global stats without per-shard score discrepancy."""
    backend = LuceneSparseBackend()

    # Common query term
    target_term = "oncology"
    other_term = "generic"

    # Shard 1: 10 documents, 9 contain target_term (very high local DF = 9/10, so low local IDF)
    shard1_records: list[LexicalRecord] = []
    for i in range(9):
        shard1_records.append(
            LexicalRecord(
                external_id=f"s1_doc_{i}",
                official_doc_id=1000 + i,
                language="en",
                item_type="document",
                encoded_contents=tokenize_and_encode(f"{target_term} " + " ".join([other_term] * 49), "en"),
            )
        )
    shard1_records.append(
        LexicalRecord(
            external_id="s1_doc_no_target",
            official_doc_id=1099,
            language="en",
            item_type="document",
            encoded_contents=tokenize_and_encode(" ".join([other_term] * 50), "en"),
        )
    )

    # Shard 2: 100 documents, only 1 contains target_term (low local DF = 1/100, so high local IDF)
    shard2_records: list[LexicalRecord] = []
    shard2_records.append(
        LexicalRecord(
            external_id="s2_doc_target",
            official_doc_id=2000,
            language="en",
            item_type="document",
            encoded_contents=tokenize_and_encode(f"{target_term} " + " ".join([other_term] * 49), "en"),
        )
    )
    for j in range(1, 100):
        shard2_records.append(
            LexicalRecord(
                external_id=f"s2_doc_{j}",
                official_doc_id=2000 + j,
                language="en",
                item_type="document",
                encoded_contents=tokenize_and_encode(" ".join([other_term] * 50), "en"),
            )
        )

    # Build both partitions
    p1 = backend.build_partition(shard1_records, tmp_path / "part_s1")
    p2 = backend.build_partition(shard2_records, tmp_path / "part_s2")

    # Assemble into single logical index
    manifest = backend.assemble([tmp_path / "part_s1", tmp_path / "part_s2"], tmp_path / "idx_assembled")

    # Search against assembled logical index
    hits = backend.search(target_term, language="en", item_type="document", top_k=20, index_path=tmp_path / "idx_assembled")

    hit_dict = {h.id: h.score for h in hits}
    assert "s1_doc_0" in hit_dict
    assert "s2_doc_target" in hit_dict

    # Documents s1_doc_0 and s2_doc_target have the EXACT same length (50 tokens) and same TF (1 for target_term).
    # Under global IDF and avgdl, their BM25 scores MUST be identical (no per-shard raw score discrepancy).
    score_s1 = hit_dict["s1_doc_0"]
    score_s2 = hit_dict["s2_doc_target"]
    assert pytest.approx(score_s1, abs=1e-5) == score_s2


def test_crash_replay_does_not_duplicate_final_ids(tmp_path: Path):
    """Crash/replay replaces partitions idempotently and never duplicates IDs."""
    backend = LuceneSparseBackend()

    records = [
        LexicalRecord(
            external_id="doc_dup_1",
            official_doc_id=501,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Viêm đường hô hấp trên", "vi"),
        ),
        LexicalRecord(
            external_id="doc_dup_2",
            official_doc_id=502,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Sốt phát ban trẻ em", "vi"),
        ),
    ]

    p1 = backend.build_partition(records, tmp_path / "part_replay_1")
    # Assemble once
    idx1 = backend.assemble([tmp_path / "part_replay_1"], tmp_path / "idx_replay")
    assert idx1.member_count == 2

    # Replay: Re-build partition (e.g. after crash / rerun) and re-assemble with same or overlapping partition
    p2 = backend.build_partition(records, tmp_path / "part_replay_2")
    idx2 = backend.assemble([tmp_path / "part_replay_1", tmp_path / "part_replay_2"], tmp_path / "idx_replay")

    # Even though two partitions have the same IDs, assembled index MUST have exactly 2 distinct members
    assert idx2.member_count == 2

    hits = backend.search("hô hấp", language="vi", item_type="document", top_k=10, index_path=tmp_path / "idx_replay")
    hit_ids = [h.id for h in hits]
    # No duplicate IDs in search hits
    assert len(hit_ids) == len(set(hit_ids))
    assert hit_ids == ["doc_dup_1"]

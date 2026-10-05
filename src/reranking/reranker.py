from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from src.data.schemas import FusedHit, ScoredHit
from src.data.vibiomir_contracts import (
    PackedPair,
    PairPackingConfig,
    SpanChunk,
)
from src.query.query_processor import QueryVariants

logger = logging.getLogger(__name__)


# =====================================================================
# Language-Aware Routing and Normalization Helpers
# =====================================================================

def resolve_query_for_language(query_variants: QueryVariants, language: str | None) -> str:
    """Select the query text corresponding to candidate language.

    Routing rules:
    - vi -> query_variants.vi
    - en -> query_variants.en or query_variants.vi
    - zh -> query_variants.zh or query_variants.vi
    - Any missing or unknown language safely falls back to query_variants.vi.
    """
    lang = (language or "").strip().lower()
    if lang == "en":
        q_en = query_variants.en.strip() if query_variants.en else ""
        return q_en if q_en else query_variants.vi
    elif lang == "zh":
        q_zh = query_variants.zh.strip() if query_variants.zh else ""
        return q_zh if q_zh else query_variants.vi
    return query_variants.vi


def min_max_normalize(
    scores: Sequence[float],
    eps: float = 1e-9,
    default_val: float = 1.0,
) -> list[float]:
    """Normalize a sequence of float scores to [0.0, 1.0] via min-max normalization.

    Formula:
        norm_score = (score - min) / (max - min + eps)

    If max - min < eps (e.g. single item or identical scores), returns [default_val] * len(scores).
    """
    if not scores:
        return []

    s_list = [float(s) for s in scores]
    min_val = min(s_list)
    max_val = max(s_list)
    diff = max_val - min_val

    if diff < eps:
        return [default_val] * len(s_list)

    return [(s - min_val) / (diff + eps) for s in s_list]


# =====================================================================
# Reranker Pair Packing & Token Budgeting
# =====================================================================

def pack_reranker_inputs(
    query: str,
    chunk: SpanChunk,
    tokenizer: Any,
    config: PairPackingConfig | None = None,
) -> list[PackedPair]:
    """Pack query and SpanChunk into token-budgeted PackedPair inputs.

    Rules:
    - Query window size: config.query_window_tokens (default 128).
    - Query window overlap: config.query_window_overlap_tokens (default 16).
    - Overall budget: config.max_length (default 512).
    - Special tokens (separators, CLS, SEP) are accounted for.
    - If total tokens exceed 512, contextual header is shrunk first.
    - Chunk body is NEVER truncated. If query window + body + special exceeds 512:
      raise ValueError if body_truncation == 'reject'.
    """
    if config is None:
        config = PairPackingConfig()

    max_length = config.max_length
    q_window_tokens = config.query_window_tokens
    q_overlap_tokens = config.query_window_overlap_tokens
    body_truncation = config.body_truncation

    # 1. Query tokenization & offset mapping
    q_tokens: list[int] = []
    q_offsets: list[tuple[int, int]] = []

    if hasattr(tokenizer, "encode"):
        try:
            q_tokens = tokenizer.encode(query, add_special_tokens=False)
        except Exception:
            pass

    if callable(tokenizer):
        try:
            res = tokenizer(query, add_special_tokens=False, return_offsets_mapping=True)
            if isinstance(res, dict):
                if not q_tokens:
                    q_tokens = res.get("input_ids", [])
                raw_offsets = res.get("offset_mapping", [])
                q_offsets = [(int(s), int(e)) for s, e in raw_offsets if e > s]
        except Exception:
            pass

    if not q_tokens:
        q_tokens = list(range(len(query.split())))

    total_q_tokens = len(q_tokens)
    step = max(1, q_window_tokens - q_overlap_tokens)
    windows: list[tuple[int, int, str]] = []

    if total_q_tokens <= q_window_tokens:
        windows.append((0, total_q_tokens, query))
    else:
        start_idx = 0
        while start_idx < total_q_tokens:
            end_idx = min(start_idx + q_window_tokens, total_q_tokens)
            win_tokens = q_tokens[start_idx:end_idx]
            if q_offsets and end_idx <= len(q_offsets):
                c_start = q_offsets[start_idx][0]
                c_end = q_offsets[end_idx - 1][1]
                w_text = query[c_start:c_end].strip()
            elif hasattr(tokenizer, "decode"):
                w_text = tokenizer.decode(win_tokens, skip_special_tokens=True).strip()
            else:
                words = query.split()
                w_text = " ".join(words[start_idx:end_idx])
            windows.append((start_idx, end_idx, w_text))
            if end_idx >= total_q_tokens:
                break
            start_idx += step

    # 2. Extract passage body and header from SpanChunk
    body_text = chunk.text
    if chunk.index_text != chunk.text and chunk.index_text.endswith(chunk.text):
        raw_header = chunk.index_text[: -len(chunk.text)].strip()
    elif chunk.section:
        raw_header = chunk.section.strip()
    else:
        raw_header = ""

    b_tokens: list[int] = []
    if hasattr(tokenizer, "encode"):
        try:
            b_tokens = tokenizer.encode(body_text, add_special_tokens=False)
        except Exception:
            pass
    if not b_tokens and callable(tokenizer):
        try:
            res = tokenizer(body_text, add_special_tokens=False)
            if isinstance(res, dict):
                b_tokens = res.get("input_ids", [])
        except Exception:
            pass
    if not b_tokens:
        b_tokens = list(range(len(body_text.split())))
    num_body = len(b_tokens)

    if hasattr(tokenizer, "num_special_tokens_to_add"):
        try:
            num_special = tokenizer.num_special_tokens_to_add(pair=True)
        except Exception:
            num_special = 3
    else:
        num_special = 3

    packed_pairs: list[PackedPair] = []

    for win_idx, (w_start, w_end, w_text) in enumerate(windows):
        num_q = w_end - w_start
        required = num_q + num_body + num_special
        if required > max_length:
            if body_truncation == "reject":
                raise ValueError(
                    f"Pair token budget exceeded: query window ({num_q}) + body ({num_body}) + "
                    f"special ({num_special}) = {required} > {max_length}. Body cannot be truncated."
                )

        available_for_header = max_length - required
        final_header = ""
        if raw_header and available_for_header > 0:
            h_tokens: list[int] = []
            if hasattr(tokenizer, "encode"):
                try:
                    h_tokens = tokenizer.encode(raw_header, add_special_tokens=False)
                except Exception:
                    pass
            if not h_tokens and callable(tokenizer):
                try:
                    res = tokenizer(raw_header, add_special_tokens=False)
                    if isinstance(res, dict):
                        h_tokens = res.get("input_ids", [])
                except Exception:
                    pass
            if not h_tokens:
                h_tokens = list(range(len(raw_header.split())))

            if len(h_tokens) <= available_for_header:
                final_header = raw_header
            else:
                shrunk_tokens = h_tokens[:available_for_header]
                if hasattr(tokenizer, "decode") and shrunk_tokens:
                    final_header = tokenizer.decode(shrunk_tokens, skip_special_tokens=True).strip()
                else:
                    h_words = raw_header.split()
                    final_header = " ".join(h_words[:available_for_header])

        passage_text = f"{final_header}\n\n{body_text}" if final_header else body_text

        input_ids: list[int] = []
        attention_mask: list[int] = []
        if callable(tokenizer):
            try:
                enc = tokenizer(w_text, passage_text, add_special_tokens=True)
                if isinstance(enc, dict):
                    input_ids = enc.get("input_ids", [])
                    attention_mask = enc.get("attention_mask", [1] * len(input_ids))
            except Exception:
                pass
        if not input_ids:
            toks = [1] + q_tokens[w_start:w_end] + [2] + b_tokens + [2]
            input_ids = toks
            attention_mask = [1] * len(toks)

        # Strict clamp guard: guarantee input_ids and attention_mask never exceed max_length
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        packed_pairs.append(
            PackedPair(
                query_text=w_text,
                passage_text=passage_text,
                input_ids=input_ids,
                attention_mask=attention_mask,
                query_window_index=win_idx,
            )
        )

    return packed_pairs


# =====================================================================
# Standalone Document Reranking via Chunk Aggregation
# =====================================================================

def rerank_documents_via_chunk_aggregation(
    doc_candidates: list[FusedHit] | None = None,
    scored_chunks: list[ScoredHit] | None = None,
    top_k: int = 100,
    fallback_floor: float | None = None,
) -> list[ScoredHit]:
    """Rerank documents by aggregating chunk scores using max pooling.

    Formula:
        base_doc_score(d) = max_{c in d}(rerank_chunk_score(c))

    If document d has no matching chunks in scored_chunks, falls back to fallback_floor
    or (min_chunk_score - 1.0) so it is not completely dropped.
    Normalized score in [0, 1] is computed across all scored documents.
    Preserves raw_fusion_score = doc_candidate.fusion_score.
    """
    chunks = scored_chunks or []

    if doc_candidates is None:
        # Synthesize doc candidates from scored_chunks
        chunks_by_doc_id: dict[str, list[ScoredHit]] = {}
        for c in chunks:
            chunks_by_doc_id.setdefault(c.doc_id, []).append(c)
        doc_candidates = [
            FusedHit(
                item_id=doc_id,
                item_type="document",
                doc_id=doc_id,
                language=doc_chunks[0].language,
                fusion_score=max(c.raw_fusion_score for c in doc_chunks),
                stream_ranks={},
            )
            for doc_id, doc_chunks in chunks_by_doc_id.items()
        ]

    if not doc_candidates:
        return []

    # Group scored chunks by doc_id
    chunks_by_doc: dict[str, list[ScoredHit]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)

    if chunks:
        min_chunk_score = min(c.rerank_score for c in chunks)
        default_floor = min_chunk_score - 1.0
    else:
        min_chunk_score = 0.0
        default_floor = 0.0

    floor = fallback_floor if fallback_floor is not None else default_floor

    raw_doc_scores: list[float] = []
    for d in doc_candidates:
        matching = chunks_by_doc.get(d.doc_id) or chunks_by_doc.get(d.item_id)
        if matching:
            score = max(c.rerank_score for c in matching)
        else:
            if not chunks:
                score = float(d.fusion_score)
            else:
                score = float(floor)
        raw_doc_scores.append(float(score))

    norm_scores = min_max_normalize(raw_doc_scores)

    scored_docs = [
        ScoredHit(
            item_id=d.item_id,
            item_type="document",
            doc_id=d.doc_id,
            language=d.language,
            rerank_score=raw_score,
            normalized_score=norm_score,
            raw_fusion_score=float(d.fusion_score),
        )
        for d, raw_score, norm_score in zip(doc_candidates, raw_doc_scores, norm_scores)
    ]

    # Deterministic sort: rerank_score desc, raw_fusion_score desc, item_id asc
    scored_docs.sort(key=lambda h: (-h.rerank_score, -h.raw_fusion_score, h.item_id))
    return scored_docs[:top_k]


# =====================================================================
# Reranker Protocol and Base Class
# =====================================================================

@runtime_checkable
class Reranker(Protocol):
    """Protocol for Cross-Encoder and deterministic rerankers."""

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-encoder scores for (query, text) pairs."""
        ...

    def rerank_chunks(
        self,
        query_variants: QueryVariants,
        candidates: list[FusedHit],
        chunk_texts: dict[str, str],
        top_k: int = 100,
    ) -> list[ScoredHit]:
        """Score candidate chunks and return normalized ScoredHit instances."""
        ...

    def rerank_documents_via_chunk_aggregation(
        self,
        doc_candidates: list[FusedHit],
        scored_chunks: list[ScoredHit],
        top_k: int = 100,
    ) -> list[ScoredHit]:
        """Aggregate chunk scores to document level using max pooling."""
        ...

    def rerank_documents_direct(
        self,
        query_variants: QueryVariants,
        candidates: list[FusedHit],
        doc_texts: dict[str, str],
        max_length: int = 512,
        top_k: int = 100,
    ) -> list[ScoredHit]:
        """Truncate document text and directly rerank documents."""
        ...


class BaseReranker(ABC):
    """Abstract base class providing shared reranking workflows."""

    @abstractmethod
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a list of (query, passage) pairs."""
        raise NotImplementedError

    def rerank_chunks(
        self,
        query_variants: QueryVariants,
        candidates: list[FusedHit],
        chunk_texts: dict[str, Any],
        top_k: int = 100,
        tokenizer: Any = None,
        config: PairPackingConfig | None = None,
    ) -> list[ScoredHit]:
        """Rerank candidate chunks with language-aware pairing and min-max normalization.

        When a tokenizer is available (or passed in) and query exceeds window limit (or chunk is SpanChunk),
        pairs are packed via pack_reranker_inputs and chunk scores are aggregated using max across query windows.
        """
        if not candidates:
            return []

        tok = tokenizer or getattr(self, "tokenizer", None) or getattr(getattr(self, "_model", None), "tokenizer", None)
        cfg = config or getattr(self, "pair_config", None) or PairPackingConfig()

        valid_candidates: list[FusedHit] = []
        candidate_pair_slices: list[tuple[int, int]] = []
        flat_pairs: list[tuple[str, str]] = []

        for candidate in candidates:
            c_val = chunk_texts.get(candidate.item_id)
            if c_val is None and candidate.doc_id in chunk_texts:
                c_val = chunk_texts.get(candidate.doc_id)
            if c_val is None:
                continue

            query_text = resolve_query_for_language(query_variants, candidate.language)

            start_pair_idx = len(flat_pairs)
            if tok is not None:
                if isinstance(c_val, SpanChunk):
                    sc = c_val
                else:
                    text_str = str(c_val)
                    sc = SpanChunk(
                        chunk_id=candidate.item_id,
                        doc_id=int(candidate.doc_id) if str(candidate.doc_id).isdigit() else 0,
                        source_version="v1",
                        source_sha256="",
                        start_char=0,
                        end_char=len(text_str),
                        chunk_index=0,
                        language=candidate.language,
                        section=None,
                        text=text_str,
                        index_text=text_str,
                        dense_token_count=len(text_str.split()),
                        reranker_token_count=len(text_str.split()),
                    )
                packed = pack_reranker_inputs(query_text, sc, tokenizer=tok, config=cfg)
                for p in packed:
                    flat_pairs.append((p.query_text, p.passage_text))
            else:
                text_str = c_val.text if isinstance(c_val, SpanChunk) else str(c_val)
                flat_pairs.append((query_text, text_str))

            end_pair_idx = len(flat_pairs)
            candidate_pair_slices.append((start_pair_idx, end_pair_idx))
            valid_candidates.append(candidate)

        if not valid_candidates:
            return []

        all_scores = self.score_pairs(flat_pairs)

        raw_scores: list[float] = []
        for s_idx, e_idx in candidate_pair_slices:
            cand_scores = all_scores[s_idx:e_idx]
            if cand_scores:
                raw_scores.append(float(max(cand_scores)))
            else:
                raw_scores.append(0.0)

        norm_scores = min_max_normalize(raw_scores)

        scored_hits = [
            ScoredHit(
                item_id=c.item_id,
                item_type="chunk",
                doc_id=c.doc_id,
                language=c.language,
                rerank_score=float(score),
                normalized_score=float(norm),
                raw_fusion_score=float(c.fusion_score),
            )
            for c, score, norm in zip(valid_candidates, raw_scores, norm_scores)
        ]

        scored_hits.sort(key=lambda h: (-h.rerank_score, -h.raw_fusion_score, h.item_id))
        return scored_hits[:top_k]

    def rerank_documents_via_chunk_aggregation(
        self,
        doc_candidates: list[FusedHit],
        scored_chunks: list[ScoredHit],
        top_k: int = 100,
        fallback_floor: float | None = None,
    ) -> list[ScoredHit]:
        """Aggregate chunk scores to doc-level scores via max-pooling."""
        return rerank_documents_via_chunk_aggregation(
            doc_candidates=doc_candidates,
            scored_chunks=scored_chunks,
            top_k=top_k,
            fallback_floor=fallback_floor,
        )

    def rerank_documents_direct(
        self,
        query_variants: QueryVariants,
        candidates: list[FusedHit],
        doc_texts: dict[str, str],
        max_length: int = 512,
        top_k: int = 100,
    ) -> list[ScoredHit]:
        """Truncate doc texts to max_length and directly score (query, truncated_text)."""
        if not candidates:
            return []

        valid_candidates: list[FusedHit] = []
        pairs: list[tuple[str, str]] = []

        for candidate in candidates:
            text = doc_texts.get(candidate.item_id)
            if text is None and candidate.doc_id in doc_texts:
                text = doc_texts.get(candidate.doc_id)
            if text is None:
                continue

            truncated_text = text[:max_length]
            query_text = resolve_query_for_language(query_variants, candidate.language)
            pairs.append((query_text, truncated_text))
            valid_candidates.append(candidate)

        if not valid_candidates:
            return []

        raw_scores = self.score_pairs(pairs)
        norm_scores = min_max_normalize(raw_scores)

        scored_hits = [
            ScoredHit(
                item_id=c.item_id,
                item_type="document",
                doc_id=c.doc_id,
                language=c.language,
                rerank_score=float(score),
                normalized_score=float(norm),
                raw_fusion_score=float(c.fusion_score),
            )
            for c, score, norm in zip(valid_candidates, raw_scores, norm_scores)
        ]

        scored_hits.sort(key=lambda h: (-h.rerank_score, -h.raw_fusion_score, h.item_id))
        return scored_hits[:top_k]


# =====================================================================
# CrossEncoderReranker Implementation
# =====================================================================

class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder Reranker supporting models, callables, and mock objects."""

    def __init__(
        self,
        model_name_or_model: str | Any | Callable[[list[tuple[str, str]]], list[float]] = "BAAI/bge-reranker-v2-m3",
        *,
        model: str | Any | Callable[[list[tuple[str, str]]], list[float]] | None = None,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        tokenizer: Any = None,
        pair_config: PairPackingConfig | None = None,
    ) -> None:
        target = model if model is not None else (model_name if model_name is not None else model_name_or_model)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.pair_config = pair_config

        self._callable: Callable[[list[tuple[str, str]]], list[float]] | None = None
        self._model: Any = None

        if callable(target) and not hasattr(target, "predict"):
            self._callable = target
            self.model_name = getattr(target, "__name__", "custom_callable")
        elif isinstance(target, str):
            self.model_name = target
            self._model = None  # Lazy loaded on first inference
        else:
            self._model = target
            self.model_name = getattr(target, "__class__", type(target)).__name__

    def _get_model(self) -> Any:
        """Lazily instantiate sentence_transformers.CrossEncoder if needed."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = self.device
            if device is None or device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            logger.info("Initializing CrossEncoder '%s' on %s", self.model_name, device)
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=device,
            )
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-encoder scores for (query, passage) pairs."""
        if not pairs:
            return []

        if self._callable is not None:
            raw_scores = self._callable(pairs)
            return [float(s) for s in raw_scores]

        model = self._get_model()
        try:
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except TypeError:
            # Simple mocks or alternative interfaces that only take pairs
            raw_scores = model.predict(pairs)

        if isinstance(raw_scores, np.ndarray):
            return [float(s) for s in raw_scores.flatten().tolist()]
        elif isinstance(raw_scores, (list, tuple)):
            return [float(s) for s in raw_scores]
        else:
            return [float(raw_scores)]


# =====================================================================
# DummyReranker Implementation
# =====================================================================

class DummyReranker(BaseReranker):
    """Deterministic lexical overlap reranker for unit testing and CPU environments."""

    def __init__(
        self,
        scoring_fn: Callable[[str, str], float] | None = None,
        default_score: float = 0.5,
    ) -> None:
        self.scoring_fn = scoring_fn
        self.default_score = default_score

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs deterministically based on word overlap or custom scoring_fn."""
        if not pairs:
            return []

        scores: list[float] = []
        for query, text in pairs:
            if self.scoring_fn is not None:
                scores.append(float(self.scoring_fn(query, text)))
            else:
                q_words = set(re.findall(r"\w+", query.lower()))
                t_words = set(re.findall(r"\w+", text.lower()))
                if not q_words:
                    scores.append(self.default_score)
                else:
                    overlap = len(q_words & t_words) / len(q_words)
                    scores.append(float(overlap))
        return scores


# =====================================================================
# Standalone Helper Functions
# =====================================================================

def rerank_chunks(
    query_variants_or_reranker: Any = None,
    candidates_or_query_variants: Any = None,
    chunk_texts_or_candidates: Any = None,
    *args: Any,
    query_variants: Any = None,
    candidates: Any = None,
    chunk_texts: Any = None,
    top_k: int = 100,
    reranker: Any = None,
    **kwargs: Any,
) -> list[ScoredHit]:
    """Rerank chunks either via (reranker, query_variants, candidates, chunk_texts)
    or (query_variants, candidates, chunk_texts, top_k, reranker=...).
    """
    if hasattr(query_variants_or_reranker, "rerank_chunks"):
        # Called as rerank_chunks(reranker, query_variants, candidates, chunk_texts, ...)
        return query_variants_or_reranker.rerank_chunks(
            query_variants=candidates_or_query_variants or query_variants,
            candidates=chunk_texts_or_candidates or candidates,
            chunk_texts=args[0] if args else (chunk_texts or kwargs.get("chunk_texts", {})),
            top_k=args[1] if len(args) > 1 else kwargs.get("top_k", top_k),
        )
    else:
        # Called as rerank_chunks(query_variants, candidates, chunk_texts, top_k=..., reranker=...)
        qv = query_variants_or_reranker or query_variants
        cands = candidates_or_query_variants or candidates
        texts = chunk_texts_or_candidates or chunk_texts or {}
        r_inst = reranker or kwargs.get("reranker")
        if r_inst is None:
            r_inst = DummyReranker()
        return r_inst.rerank_chunks(
            query_variants=qv,
            candidates=cands,
            chunk_texts=texts,
            top_k=kwargs.get("top_k", args[0] if args else top_k),
        )


def rerank_documents_direct(
    query_variants_or_reranker: Any = None,
    candidates_or_query_variants: Any = None,
    doc_texts_or_candidates: Any = None,
    *args: Any,
    query_variants: Any = None,
    candidates: Any = None,
    doc_texts: Any = None,
    max_length: int = 512,
    top_k: int = 100,
    reranker: Any = None,
    **kwargs: Any,
) -> list[ScoredHit]:
    """Rerank documents directly either via (reranker, query_variants, candidates, doc_texts)
    or (query_variants, candidates, doc_texts, max_length, top_k, reranker=...).
    """
    if hasattr(query_variants_or_reranker, "rerank_documents_direct"):
        return query_variants_or_reranker.rerank_documents_direct(
            query_variants=candidates_or_query_variants or query_variants,
            candidates=doc_texts_or_candidates or candidates,
            doc_texts=args[0] if args else (doc_texts or kwargs.get("doc_texts", {})),
            max_length=args[1] if len(args) > 1 else kwargs.get("max_length", max_length),
            top_k=args[2] if len(args) > 2 else kwargs.get("top_k", top_k),
        )
    else:
        qv = query_variants_or_reranker or query_variants
        cands = candidates_or_query_variants or candidates
        texts = doc_texts_or_candidates or doc_texts or {}
        r_inst = reranker or kwargs.get("reranker")
        if r_inst is None:
            r_inst = DummyReranker()
        return r_inst.rerank_documents_direct(
            query_variants=qv,
            candidates=cands,
            doc_texts=texts,
            max_length=kwargs.get("max_length", args[0] if len(args) > 0 else max_length),
            top_k=kwargs.get("top_k", args[1] if len(args) > 1 else top_k),
        )

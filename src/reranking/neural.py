from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.phase2_contracts import EvidenceSpan, RerankConfig
from src.query.query_processor import QueryVariants
from src.data.vibiomir_contracts import SourceStore, SpanChunk
from src.reranking.reranker import resolve_query_for_language, pack_reranker_inputs

logger = logging.getLogger(__name__)


class NeuralReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        device: str | None = None,
        batch_size: int = 4,
        fp16: bool = True,
        require_real_weights: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.fp16 = fp16
        self.require_real_weights = require_real_weights

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self._is_mock = False

        try:
            logger.info("Loading tokenizer for %s", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Loading sequence classification model for %s", model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            if self.fp16 and self.device == "cuda":
                self.model.half()
                logger.info("Converted model to FP16 on CUDA")
        except Exception as e:
            logger.warning("Failed to load real weights for %s: %s", model_name, e)
            if self.require_real_weights:
                raise RuntimeError(
                    f"Failed to load real weights for {model_name} with require_real_weights=True: {e}"
                ) from e
            else:
                logger.warning("Falling back to mock reranker behavior due to require_real_weights=False")
                self._is_mock = True

    @property
    def model_fingerprint(self) -> str:
        h = hashlib.sha256(f"{self.model_name}:{self._is_mock}".encode("utf-8")).hexdigest()[:16]
        return f"bge_reranker:{self.model_name.replace('/', '_')}:{h}"

    def score_pairs(self, pairs: Sequence[tuple[str, str]], batch_size: int | None = None) -> list[float]:
        if not pairs:
            return []

        if self._is_mock or self.model is None or self.tokenizer is None:
            # Mock scoring: return pseudo-scores based on length / hash
            scores = []
            for q, p in pairs:
                h = int(hashlib.md5(f"{q}:{p}".encode("utf-8")).hexdigest(), 16)
                score = float((h % 1000) / 100.0 - 5.0)  # range roughly -5 to 5
                scores.append(score)
            return scores

        bs = batch_size if batch_size is not None else self.batch_size
        all_scores: list[float] = []

        i = 0
        while i < len(pairs):
            batch_pairs = list(pairs[i : i + bs])
            try:
                encoded = self.tokenizer(
                    [p[0] for p in batch_pairs],
                    [p[1] for p in batch_pairs],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = self.model(**encoded)
                    # BGE-reranker outputs logits of shape (batch_size, 1) or (batch_size,)
                    logits = outputs.logits
                    if logits.ndim > 1:
                        logits = logits.squeeze(-1)
                    batch_scores = logits.detach().cpu().float().tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    all_scores.extend(batch_scores)
                i += bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    logger.warning("CUDA OOM encountered with batch_size=%d. Halving batch size and retrying...", bs)
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        return all_scores


class MockNeuralReranker(NeuralReranker):
    def __init__(self) -> None:
        self.model_name = "mock-bge-reranker"
        self.max_length = 512
        self.batch_size = 4
        self.fp16 = False
        self.require_real_weights = False
        self.tokenizer = None
        self.model = None
        self._is_mock = True


def get_mock_reranker() -> NeuralReranker:
    return MockNeuralReranker()


def score_evidence(
    query: QueryVariants,
    spans: Sequence[EvidenceSpan],
    reranker: NeuralReranker,
    store: SourceStore | Any | None = None,
    config: RerankConfig | None = None,
    cache_dir: Path | str | None = None,
) -> list[EvidenceSpan]:
    """Score source evidence spans using NeuralReranker with query-passage windowing,
    window aggregation (max raw logit), OOM resilience, and deterministic caching.
    """
    if config is None:
        config = RerankConfig()

    if not config.enabled or not spans:
        return list(spans)

    cache_path: Path | None = None
    cache_dict: dict[str, float] = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_fp = getattr(reranker, "model_fingerprint", "bge_reranker_default")
        cache_filename = f"rerank_cache_{model_fp}.json"
        cache_path = cache_dir / cache_filename
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_dict = json.load(f)
            except Exception as e:
                logger.warning("Failed to load rerank cache from %s: %s", cache_path, e)

    # Prepare windowing config for packing
    from src.data.vibiomir_contracts import PairPackingConfig
    packing_config = PairPackingConfig(
        max_length=config.max_length,
        query_window_tokens=config.query_window_tokens,
        query_window_overlap_tokens=config.query_window_overlap,
        query_window_aggregation=config.window_aggregation,
        body_truncation="reject",
    )

    updated_spans: list[EvidenceSpan] = []
    pairs_to_score: list[tuple[str, str]] = []
    pair_map: list[tuple[int, int]] = []  # maps pair index to (span_index, window_index)

    # First pass: resolve texts and generate windows for each span
    span_windows: list[list[str]] = []
    for span_idx, span in enumerate(spans):
        # Resolve passage text
        passage_text = ""
        doc_lang = "vi"
        if store is not None:
            try:
                # Try getting chunk or document
                if hasattr(store, "get_chunk"):
                    try:
                        chunk = store.get_chunk(span.span_id)
                        passage_text = chunk.text
                        doc_lang = getattr(chunk, "language", "vi")
                    except Exception:
                        pass
                if not passage_text and hasattr(store, "get_document"):
                    doc = store.get_document(span.doc_id)
                    doc_lang = getattr(doc, "language", "vi")
                    source_text = getattr(doc, "source_text", "")
                    if source_text and span.start_char >= 0 and span.end_char > span.start_char:
                        passage_text = source_text[span.start_char : span.end_char]
            except Exception as e:
                logger.warning("Failed to resolve text for span %s: %s", span.span_id, e)

        if not passage_text:
            passage_text = f"Span {span.span_id} doc {span.doc_id}"

        # Resolve query text for language
        query_text = resolve_query_for_language(query, doc_lang)

        # Windowing passage if tokens exceed passage_window_tokens (approximate via words or tokenizer if available)
        # If reranker has tokenizer, use tokenizer windowing, otherwise simple word/char windowing
        passage_windows = [passage_text]
        if hasattr(reranker, "tokenizer") and reranker.tokenizer is not None:
            try:
                p_tokens = reranker.tokenizer.encode(passage_text, add_special_tokens=False)
                if len(p_tokens) > config.passage_window_tokens:
                    passage_windows = []
                    step = config.passage_window_tokens - 16
                    if step < 1:
                        step = config.passage_window_tokens
                    for start_idx in range(0, len(p_tokens), step):
                        window_tokens = p_tokens[start_idx : start_idx + config.passage_window_tokens]
                        w_text = reranker.tokenizer.decode(window_tokens, skip_special_tokens=True)
                        if w_text.strip():
                            passage_windows.append(w_text)
            except Exception as e:
                logger.debug("Tokenizer passage windowing failed, using fallback: %s", e)

        # Query windowing if query tokens > query_window_tokens
        query_windows = [query_text]
        if hasattr(reranker, "tokenizer") and reranker.tokenizer is not None:
            try:
                q_tokens = reranker.tokenizer.encode(query_text, add_special_tokens=False)
                if len(q_tokens) > config.query_window_tokens:
                    query_windows = []
                    q_step = config.query_window_tokens - config.query_window_overlap
                    if q_step < 1:
                        q_step = config.query_window_tokens
                    for start_idx in range(0, len(q_tokens), q_step):
                        w_tokens = q_tokens[start_idx : start_idx + config.query_window_tokens]
                        w_text = reranker.tokenizer.decode(w_tokens, skip_special_tokens=True)
                        if w_text.strip():
                            query_windows.append(w_text)
            except Exception as e:
                logger.debug("Tokenizer query windowing failed, using fallback: %s", e)

        # Build all combinations of (query_window, passage_window) for this span
        span_pair_indices = []
        for q_w in query_windows:
            for p_w in passage_windows:
                # Check cache first
                q_hash = hashlib.sha256(q_w.encode("utf-8")).hexdigest()[:16]
                p_hash = hashlib.sha256(p_w.encode("utf-8")).hexdigest()[:16]
                model_fp = getattr(reranker, "model_fingerprint", "default")
                cache_key = f"{q_hash}:{p_hash}:{model_fp}:{config.max_length}"

                if cache_key in cache_dict:
                    score = cache_dict[cache_key]
                    span_pair_indices.append(score)
                else:
                    pair_idx = len(pairs_to_score)
                    pairs_to_score.append((q_w, p_w))
                    pair_map.append((span_idx, pair_idx))
                    span_pair_indices.append(cache_key)  # placeholder or deferred

        span_windows.append(span_pair_indices)

    # Score pending pairs
    scored_pair_values: list[float] = []
    if pairs_to_score:
        raw_scores = reranker.score_pairs(pairs_to_score)
        scored_pair_values = raw_scores

    # Map back and cache scores
    span_scores: list[list[float]] = [[] for _ in spans]
    pair_val_idx = 0

    for span_idx, indices in enumerate(span_windows):
        for item in indices:
            if isinstance(item, (int, float)):
                span_scores[span_idx].append(float(item))
            else:
                # it's a cache_key
                score = scored_pair_values[pair_val_idx]
                pair_val_idx += 1
                cache_dict[item] = score
                span_scores[span_idx].append(float(score))

    # Save cache if cache_path provided
    if cache_path is not None:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to write rerank cache to %s: %s", cache_path, e)

    # Aggregate scores per span using window_aggregation (default max)
    for span_idx, span in enumerate(spans):
        scores = span_scores[span_idx]
        if not scores:
            agg_score = 0.0
        else:
            if config.window_aggregation.lower() == "mean":
                agg_score = float(sum(scores) / len(scores))
            else:
                agg_score = float(max(scores))

        updated_span = EvidenceSpan(
            span_id=span.span_id,
            doc_id=span.doc_id,
            source_version=span.source_version,
            source_sha256=span.source_sha256,
            start_char=span.start_char,
            end_char=span.end_char,
            seed_chunk_ids=span.seed_chunk_ids,
            variant_kind=span.variant_kind,
            base_span_id=span.base_span_id,
            added_block_span_ids=span.added_block_span_ids,
            score=agg_score,
        )
        updated_spans.append(updated_span)

    return updated_spans

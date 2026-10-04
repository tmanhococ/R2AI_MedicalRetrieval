import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.schemas import ChunkRecord, DocumentRecord
from src.indexing.tokenizers import get_tokenizer

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _RANK_BM25_AVAILABLE = True
except ImportError:
    _RANK_BM25_AVAILABLE = False

try:
    import bm25s  # type: ignore
    _BM25S_AVAILABLE = True
except ImportError:
    _BM25S_AVAILABLE = False


class BM25Indexer:
    """Multilingual BM25 Indexer supporting rank_bm25 and bm25s backends.

    Partitions documents and chunks by language ('vi', 'en', 'zh', etc.) and builds
    separate indexes for documents and chunks.
    """

    DEFAULT_TOKENIZERS: dict[str, str] = {
        "vi": "syllable",
        "en": "whitespace",
        "zh": "jieba",
    }

    def __init__(
        self,
        backend: str = "bm25s",
        tokenizers_config: dict[str, str] | None = None,
    ) -> None:
        backend_lower = backend.lower()
        if backend_lower not in ("bm25s", "rank_bm25"):
            raise ValueError(
                f"Unsupported BM25 backend '{backend}'. Supported backends: 'bm25s', 'rank_bm25'"
            )

        if backend_lower == "rank_bm25" and not _RANK_BM25_AVAILABLE:
            raise ImportError(
                "Backend 'rank_bm25' is selected but package 'rank_bm25' is not installed."
            )
        if backend_lower == "bm25s" and not _BM25S_AVAILABLE:
            raise ImportError(
                "Backend 'bm25s' is selected but package 'bm25s' is not installed."
            )

        self.backend = backend_lower
        self.tokenizers_config = (
            tokenizers_config if tokenizers_config is not None else dict(self.DEFAULT_TOKENIZERS)
        )

    def build_indexes(
        self,
        documents: Sequence[DocumentRecord] | Iterable[DocumentRecord],
        chunks: Sequence[ChunkRecord] | Iterable[ChunkRecord],
        output_dir: Path | str,
        snapshot_id: str | None = None,
        source_membership_sha256: str | None = None,
        chunker_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Build partitioned BM25 indexes for documents and chunks.

        Args:
            documents: List/Sequence or Iterable of DocumentRecord items.
            chunks: List/Sequence or Iterable of ChunkRecord items.
            output_dir: Directory where indexes and metadata will be saved.
            snapshot_id: Optional snapshot identifier for provenance tracking.
            source_membership_sha256: Optional checksum of source documents.
            chunker_fingerprint: Optional chunker configuration fingerprint.

        Returns:
            Dictionary containing the index manifest.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        docs_by_lang: dict[str, list[DocumentRecord]] = defaultdict(list)
        total_docs = 0
        for doc in documents:
            docs_by_lang[doc.language].append(doc)
            total_docs += 1

        chunks_by_lang: dict[str, list[ChunkRecord]] = defaultdict(list)
        total_chunks = 0
        for chunk in chunks:
            chunks_by_lang[chunk.language].append(chunk)
            total_chunks += 1

        # Standard languages plus any custom languages present in corpus
        languages = sorted(set(self.DEFAULT_TOKENIZERS.keys()) | set(docs_by_lang.keys()) | set(chunks_by_lang.keys()))

        manifest: dict[str, Any] = {
            "backend": self.backend,
            "tokenizers_config": self.tokenizers_config,
            "partitions": {},
        }
        if snapshot_id is not None:
            manifest["snapshot_id"] = snapshot_id
        if source_membership_sha256 is not None:
            manifest["source_membership_sha256"] = source_membership_sha256
        if chunker_fingerprint is not None:
            manifest["chunker_fingerprint"] = chunker_fingerprint

        manifest["resource_estimate"] = {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
        }

        for lang in languages:
            # 1. Documents Partition
            doc_partition_name = f"{lang}_documents"
            lang_docs = docs_by_lang.get(lang, [])
            if lang_docs:
                tokenizer_mode = self.tokenizers_config.get(lang)
                tokenizer = get_tokenizer(lang, tokenizer_mode)

                texts = [
                    f"{d.title} {d.text}".strip() if d.title else d.text
                    for d in lang_docs
                ]
                tokenized_corpus = [tokenizer(t) for t in texts]
                item_ids = [d.doc_id for d in lang_docs]
                doc_ids = [d.doc_id for d in lang_docs]

                self._save_partition(
                    partition_name=doc_partition_name,
                    tokenized_corpus=tokenized_corpus,
                    item_ids=item_ids,
                    doc_ids=doc_ids,
                    language=lang,
                    item_type="document",
                    output_path=output_path,
                )
                manifest["partitions"][doc_partition_name] = {
                    "count": len(lang_docs),
                    "item_type": "document",
                    "language": lang,
                }
            else:
                manifest["partitions"][doc_partition_name] = {
                    "count": 0,
                    "item_type": "document",
                    "language": lang,
                }

            # 2. Chunks Partition
            chunk_partition_name = f"{lang}_chunks"
            lang_chunks = chunks_by_lang.get(lang, [])
            if lang_chunks:
                tokenizer_mode = self.tokenizers_config.get(lang)
                tokenizer = get_tokenizer(lang, tokenizer_mode)

                texts = [c.text for c in lang_chunks]
                tokenized_corpus = [tokenizer(t) for t in texts]
                item_ids = [c.chunk_id for c in lang_chunks]
                doc_ids = [c.doc_id for c in lang_chunks]

                self._save_partition(
                    partition_name=chunk_partition_name,
                    tokenized_corpus=tokenized_corpus,
                    item_ids=item_ids,
                    doc_ids=doc_ids,
                    language=lang,
                    item_type="chunk",
                    output_path=output_path,
                )
                manifest["partitions"][chunk_partition_name] = {
                    "count": len(lang_chunks),
                    "item_type": "chunk",
                    "language": lang,
                }
            else:
                manifest["partitions"][chunk_partition_name] = {
                    "count": 0,
                    "item_type": "chunk",
                    "language": lang,
                }

        # Write manifest.json
        manifest_file = output_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

    def _save_partition(
        self,
        partition_name: str,
        tokenized_corpus: list[list[str]],
        item_ids: list[str],
        doc_ids: list[str],
        language: str,
        item_type: str,
        output_path: Path,
    ) -> None:
        """Helper to build and persist a single partition model and metadata."""
        if self.backend == "rank_bm25":
            bm25_model = BM25Okapi(tokenized_corpus)
            # Ensure terms have a positive idf floor to avoid 0 scores on small corpora (N<=2) or half-corpus terms
            for word, idf_val in bm25_model.idf.items():
                if idf_val <= 0:
                    bm25_model.idf[word] = (
                        0.25
                        if bm25_model.average_idf <= 0
                        else max(bm25_model.epsilon * bm25_model.average_idf, 0.25)
                    )
            pkl_path = output_path / f"{partition_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {
                        "bm25": bm25_model,
                        "item_ids": item_ids,
                        "doc_ids": doc_ids,
                        "language": language,
                        "item_type": item_type,
                    },
                    f,
                )

        elif self.backend == "bm25s":
            model = bm25s.BM25()
            model.index(tokenized_corpus, show_progress=False)
            model_dir = output_path / partition_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir))

            meta_pkl = output_path / f"{partition_name}_meta.pkl"
            with open(meta_pkl, "wb") as f:
                pickle.dump(
                    {
                        "item_ids": item_ids,
                        "doc_ids": doc_ids,
                        "language": language,
                        "item_type": item_type,
                    },
                    f,
                )

        # Write human-readable metadata jsonl
        meta_jsonl = output_path / f"{partition_name}_metadata.jsonl"
        with open(meta_jsonl, "w", encoding="utf-8") as f:
            for item_id, doc_id in zip(item_ids, doc_ids):
                record = {
                    "item_id": item_id,
                    "doc_id": doc_id,
                    "language": language,
                    "item_type": item_type,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

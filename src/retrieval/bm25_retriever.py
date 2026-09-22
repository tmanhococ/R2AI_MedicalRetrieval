import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.data.schemas import SearchHit
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


class BM25Retriever:
    """Multilingual BM25 Retriever for partitioned documents and chunks."""

    def __init__(
        self,
        index_dir: Path | str,
        backend: str | None = None,
        tokenizers_config: dict[str, str] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self._manifest = self._load_manifest()

        # Determine backend: explicit argument -> manifest -> default "bm25s" (or "rank_bm25" if bm25s not available)
        if backend is not None:
            self.backend = backend.lower()
        elif "backend" in self._manifest:
            self.backend = str(self._manifest["backend"]).lower()
        else:
            self.backend = "bm25s" if _BM25S_AVAILABLE else "rank_bm25"

        if self.backend == "rank_bm25" and not _RANK_BM25_AVAILABLE:
            raise ImportError("Backend 'rank_bm25' is selected but 'rank_bm25' is not installed.")
        if self.backend == "bm25s" and not _BM25S_AVAILABLE:
            raise ImportError("Backend 'bm25s' is selected but 'bm25s' is not installed.")

        # Determine tokenizers config
        if tokenizers_config is not None:
            self.tokenizers_config = tokenizers_config
        else:
            self.tokenizers_config = self._manifest.get("tokenizers_config", {})

        # Cache for loaded partition models and metadata
        self._loaded_partitions: dict[str, dict[str, Any]] = {}

    def _load_manifest(self) -> dict[str, Any]:
        manifest_file = self.index_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def search_documents(
        self,
        query: str,
        language: str,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Search across document index for a specific language."""
        return self._search_partition(
            partition_name=f"{language}_documents",
            query=query,
            language=language,
            item_type="document",
            top_k=top_k,
        )

    def search_chunks(
        self,
        query: str,
        language: str,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Search across chunk index for a specific language."""
        return self._search_partition(
            partition_name=f"{language}_chunks",
            query=query,
            language=language,
            item_type="chunk",
            top_k=top_k,
        )

    def _get_partition(self, partition_name: str) -> dict[str, Any] | None:
        """Load and cache partition data if available."""
        if partition_name in self._loaded_partitions:
            return self._loaded_partitions[partition_name]

        # Check manifest counts if present
        partitions_info = self._manifest.get("partitions", {})
        if partition_name in partitions_info:
            if partitions_info[partition_name].get("count", 0) <= 0:
                return None

        if self.backend == "rank_bm25":
            pkl_path = self.index_dir / f"{partition_name}.pkl"
            if not pkl_path.exists():
                return None
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            # Ensure loaded bm25 has positive idf values if needed
            bm25_model = data.get("bm25")
            if bm25_model is not None and hasattr(bm25_model, "idf"):
                for word, idf_val in bm25_model.idf.items():
                    if idf_val <= 0:
                        bm25_model.idf[word] = (
                            0.25
                            if getattr(bm25_model, "average_idf", 0) <= 0
                            else max(bm25_model.epsilon * bm25_model.average_idf, 0.25)
                        )
            self._loaded_partitions[partition_name] = data
            return data

        elif self.backend == "bm25s":
            model_dir = self.index_dir / partition_name
            meta_path = self.index_dir / f"{partition_name}_meta.pkl"
            if not model_dir.exists() or not meta_path.exists():
                return None

            model = bm25s.BM25.load(str(model_dir), load_corpus=False)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            data = {
                "bm25": model,
                "item_ids": meta["item_ids"],
                "doc_ids": meta["doc_ids"],
                "language": meta.get("language"),
                "item_type": meta.get("item_type"),
            }
            self._loaded_partitions[partition_name] = data
            return data

        return None

    def _search_partition(
        self,
        partition_name: str,
        query: str,
        language: str,
        item_type: Literal["document", "chunk"],
        top_k: int,
    ) -> list[SearchHit]:
        if not query or not query.strip() or top_k <= 0:
            return []

        partition = self._get_partition(partition_name)
        if partition is None:
            return []

        tokenizer_mode = self.tokenizers_config.get(language)
        tokenizer = get_tokenizer(language, tokenizer_mode)
        query_tokens = tokenizer(query)
        if not query_tokens:
            return []

        item_ids: list[str] = partition["item_ids"]
        doc_ids: list[str] = partition["doc_ids"]

        if not item_ids:
            return []

        if self.backend == "rank_bm25":
            bm25_model = partition["bm25"]
            raw_scores = bm25_model.get_scores(query_tokens)
            scores = np.asarray(raw_scores, dtype=float)

            pos_mask = scores > 0.0
            if not np.any(pos_mask):
                return []

            pos_indices = np.where(pos_mask)[0]
            sorted_order = np.argsort(-scores[pos_indices])
            top_indices = pos_indices[sorted_order][:top_k]

            hits: list[SearchHit] = []
            for rank_idx, idx in enumerate(top_indices, start=1):
                hits.append(
                    SearchHit(
                        item_id=str(item_ids[idx]),
                        item_type=item_type,
                        doc_id=str(doc_ids[idx]),
                        language=language,
                        source="bm25",
                        raw_score=float(scores[idx]),
                        rank=rank_idx,
                    )
                )
            return hits

        elif self.backend == "bm25s":
            model = partition["bm25"]
            num_docs = len(item_ids)
            k = min(top_k, num_docs)
            if k <= 0:
                return []

            retrieved_docs, retrieved_scores = model.retrieve(
                [query_tokens], k=k, show_progress=False
            )
            top_doc_indices = retrieved_docs[0]
            top_scores = retrieved_scores[0]

            hits = []
            rank_idx = 1
            for doc_idx, score in zip(top_doc_indices, top_scores):
                score_val = float(score)
                if score_val <= 0.0:
                    continue
                hits.append(
                    SearchHit(
                        item_id=str(item_ids[int(doc_idx)]),
                        item_type=item_type,
                        doc_id=str(doc_ids[int(doc_idx)]),
                        language=language,
                        source="bm25",
                        raw_score=score_val,
                        rank=rank_idx,
                    )
                )
                rank_idx += 1
            return hits

        return []

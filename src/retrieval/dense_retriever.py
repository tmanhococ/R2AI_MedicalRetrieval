import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.schemas import SearchHit
from src.indexing.dense_index import _encode_texts

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class DenseRetriever:
    """Multilingual Dense Vector Retriever supporting FAISS and NumPy backends."""

    def __init__(
        self,
        encoder: Any = None,
        index_dir: Path | str = "",
        backend: str | None = None,
        instruction_prefix: dict[str, str] | None = None,
        device: str = "auto",
        normalize_embeddings: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._raw_encoder = encoder
        self._encoder: Any = None

        self._manifest = self._load_manifest()

        # Determine backend: explicit argument -> manifest -> default "faiss" (or "numpy")
        if backend is not None:
            self.backend = backend.lower()
        elif "backend" in self._manifest:
            self.backend = str(self._manifest["backend"]).lower()
        else:
            self.backend = "faiss" if _FAISS_AVAILABLE else "numpy"

        if self.backend not in ("faiss", "numpy"):
            raise ValueError(
                f"Unsupported dense backend '{self.backend}'. Supported backends: 'faiss', 'numpy'"
            )

        if self.backend == "faiss" and not _FAISS_AVAILABLE:
            raise ImportError(
                "Backend 'faiss' is selected but package 'faiss' is not installed."
            )

        # Determine instruction prefix
        if instruction_prefix is not None:
            self.instruction_prefix = instruction_prefix
        else:
            self.instruction_prefix = self._manifest.get("instruction_prefix", {})

        # Cache for loaded partitions
        self._loaded_partitions: dict[str, dict[str, Any]] = {}

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        if isinstance(self._raw_encoder, str):
            from sentence_transformers import SentenceTransformer  # type: ignore

            dev = self.device
            if dev == "auto":
                try:
                    import torch  # type: ignore
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    dev = "cpu"
            self._encoder = SentenceTransformer(self._raw_encoder, device=dev)
        else:
            self._encoder = self._raw_encoder

        return self._encoder

    def _load_manifest(self) -> dict[str, Any]:
        manifest_file = self.index_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _get_partition(self, language: str) -> dict[str, Any] | None:
        """Load and cache partition data for a given language."""
        if language in self._loaded_partitions:
            return self._loaded_partitions[language]

        # Check manifest counts if present
        partitions_info = self._manifest.get("partitions", {})
        if language in partitions_info:
            if partitions_info[language].get("count", 0) <= 0:
                return None

        # Load metadata
        meta_path = self.index_dir / f"metadata_{language}.jsonl"
        if not meta_path.exists():
            return None

        metadata: list[dict[str, Any]] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    metadata.append(json.loads(line_str))

        if not metadata:
            return None

        partition: dict[str, Any] = {"metadata": metadata}

        if self.backend == "faiss":
            faiss_path = self.index_dir / f"faiss_{language}.index"
            npy_path = self.index_dir / f"chunk_embeddings_{language}.npy"

            if faiss_path.exists():
                index = faiss.read_index(str(faiss_path))
                partition["faiss_index"] = index
            elif npy_path.exists():
                # Fallback: build FAISS index in-memory from .npy
                embeddings = np.load(str(npy_path))
                index = faiss.IndexFlatIP(embeddings.shape[1])
                if len(embeddings) > 0:
                    index.add(embeddings)
                partition["faiss_index"] = index
            else:
                return None

        elif self.backend == "numpy":
            npy_path = self.index_dir / f"chunk_embeddings_{language}.npy"
            faiss_path = self.index_dir / f"faiss_{language}.index"

            if npy_path.exists():
                partition["embeddings"] = np.load(str(npy_path))
            elif faiss_path.exists() and _FAISS_AVAILABLE:
                faiss_idx = faiss.read_index(str(faiss_path))
                num_vecs = faiss_idx.ntotal
                partition["embeddings"] = faiss_idx.reconstruct_n(0, num_vecs)
            else:
                return None

        self._loaded_partitions[language] = partition
        return partition

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query string into normalized 1D vector."""
        prefix = self.instruction_prefix.get("query", "")
        formatted_query = f"{prefix}{query}" if prefix else query

        encoder = self._get_encoder()
        vec = _encode_texts(
            encoder=encoder,
            texts=[formatted_query],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
        )

        query_vec = vec[0]
        if self.normalize_embeddings:
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

        return query_vec.astype(np.float32)

    def search_chunks(
        self,
        query: str,
        language: str,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Search chunk index for a given language partition."""
        if not query or not query.strip() or top_k <= 0:
            return []

        partition = self._get_partition(language)
        if partition is None:
            return []

        metadata = partition["metadata"]
        num_items = len(metadata)
        if num_items == 0:
            return []

        query_vec = self.encode_query(query)
        k = min(top_k, num_items)

        if self.backend == "faiss":
            faiss_index = partition["faiss_index"]
            query_mat = query_vec.reshape(1, -1)
            scores_mat, indices_mat = faiss_index.search(query_mat, k)
            scores = scores_mat[0]
            indices = indices_mat[0]
        else:  # numpy backend
            embeddings = partition["embeddings"]
            all_scores = np.dot(embeddings, query_vec)
            # Sort indices descending
            sorted_indices = np.argsort(-all_scores)[:k]
            scores = all_scores[sorted_indices]
            indices = sorted_indices

        hits: list[SearchHit] = []
        for rank_idx, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= num_items:
                continue
            item_meta = metadata[int(idx)]
            hits.append(
                SearchHit(
                    item_id=str(item_meta["chunk_id"]),
                    item_type="chunk",
                    doc_id=str(item_meta["doc_id"]),
                    language=language,
                    source="dense",
                    raw_score=float(score),
                    rank=rank_idx,
                )
            )

        return hits

    def aggregate_documents(self, chunk_hits: list[SearchHit]) -> list[SearchHit]:
        """Aggregate chunk hits into document hits by maximum chunk score."""
        if not chunk_hits:
            return []

        # Group by doc_id, track max score and best hit metadata
        best_doc_scores: dict[str, float] = {}
        doc_language: dict[str, str] = {}

        for hit in chunk_hits:
            doc_id = hit.doc_id
            if doc_id not in best_doc_scores or hit.raw_score > best_doc_scores[doc_id]:
                best_doc_scores[doc_id] = hit.raw_score
                doc_language[doc_id] = hit.language

        # Sort documents by score descending
        sorted_docs = sorted(
            best_doc_scores.items(), key=lambda item: item[1], reverse=True
        )

        doc_hits: list[SearchHit] = []
        for rank_idx, (doc_id, score) in enumerate(sorted_docs, start=1):
            doc_hits.append(
                SearchHit(
                    item_id=doc_id,
                    item_type="document",
                    doc_id=doc_id,
                    language=doc_language[doc_id],
                    source="dense",
                    raw_score=score,
                    rank=rank_idx,
                )
            )

        return doc_hits

    def search_documents(
        self,
        query: str,
        language: str,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Search and return document hits by aggregating chunk hits."""
        if not query or not query.strip() or top_k <= 0:
            return []
        # Retrieve chunks with a multiplier to ensure sufficient unique documents
        chunk_hits = self.search_chunks(query=query, language=language, top_k=top_k * 4)
        doc_hits = self.aggregate_documents(chunk_hits)
        return doc_hits[:top_k]

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.schemas import ChunkRecord

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


def _encode_texts(
    encoder: Any,
    texts: list[str],
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Helper to encode texts into float32 embeddings array."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if hasattr(encoder, "encode"):
        try:
            embeddings = encoder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )
        except TypeError:
            # Encoder encode() does not accept batch_size or normalize_embeddings
            embeddings = encoder.encode(texts)
    elif callable(encoder):
        embeddings = encoder(texts)
    else:
        raise ValueError(
            f"Encoder object {encoder} is neither callable nor provides an 'encode' method."
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if normalize_embeddings and embeddings.size > 0:
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

    return embeddings


class DenseIndexer:
    """Multilingual Dense Vector Indexer supporting FAISS and NumPy backends.

    Partitions chunks by language ('vi', 'en', 'zh', etc.) and builds
    vector indexes with metadata mapping.
    """

    DEFAULT_LANGUAGES: list[str] = ["vi", "en", "zh"]

    def __init__(
        self,
        encoder: Any = None,
        backend: str = "faiss",
        instruction_prefix: dict[str, str] | None = None,
        device: str = "auto",
        normalize_embeddings: bool = True,
        batch_size: int = 64,
    ) -> None:
        backend_lower = backend.lower()
        if backend_lower not in ("faiss", "numpy"):
            raise ValueError(
                f"Unsupported dense backend '{backend}'. Supported backends: 'faiss', 'numpy'"
            )

        if backend_lower == "faiss" and not _FAISS_AVAILABLE:
            raise ImportError(
                "Backend 'faiss' is selected but package 'faiss' is not installed."
            )

        self.backend = backend_lower
        self.instruction_prefix = instruction_prefix or {}
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._raw_encoder = encoder
        self._encoder: Any = None

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

    def build_index(
        self,
        chunks: Sequence[ChunkRecord],
        output_dir: Path | str,
    ) -> dict[str, Any]:
        """Build partitioned dense vector indexes for chunks.

        Args:
            chunks: Sequence of ChunkRecord items.
            output_dir: Directory where indexes and metadata will be saved.

        Returns:
            Dictionary containing the index manifest.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        chunks_by_lang: dict[str, list[ChunkRecord]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_lang[chunk.language].append(chunk)

        languages = sorted(set(self.DEFAULT_LANGUAGES) | set(chunks_by_lang.keys()))
        encoder = self._get_encoder()
        passage_prefix = self.instruction_prefix.get("passage", "")

        manifest: dict[str, Any] = {
            "backend": self.backend,
            "instruction_prefix": self.instruction_prefix,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
            "dim": None,
            "partitions": {},
        }

        embedding_dim: int | None = None

        for lang in languages:
            lang_chunks = chunks_by_lang.get(lang, [])
            if not lang_chunks:
                manifest["partitions"][lang] = {
                    "count": 0,
                    "language": lang,
                }
                continue

            texts_to_encode = [
                f"{passage_prefix}{c.text}" if passage_prefix else c.text
                for c in lang_chunks
            ]

            embeddings = _encode_texts(
                encoder=encoder,
                texts=texts_to_encode,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
            )

            if embeddings.ndim == 2 and embeddings.shape[0] > 0:
                embedding_dim = int(embeddings.shape[1])
                manifest["dim"] = embedding_dim

            # Always save embeddings as .npy for portability and fallback
            npy_path = output_path / f"chunk_embeddings_{lang}.npy"
            np.save(str(npy_path), embeddings)

            # If FAISS backend, build and save FAISS index
            if self.backend == "faiss":
                if not _FAISS_AVAILABLE:
                    raise ImportError("faiss is required for backend='faiss'")
                dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
                index = faiss.IndexFlatIP(dim)
                if len(embeddings) > 0:
                    index.add(embeddings)
                faiss_path = output_path / f"faiss_{lang}.index"
                faiss.write_index(index, str(faiss_path))

            # Save metadata jsonl: chunk_id, doc_id, language, chunk_index
            metadata_file = output_path / f"metadata_{lang}.jsonl"
            with open(metadata_file, "w", encoding="utf-8") as f:
                for c in lang_chunks:
                    rec = {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "language": c.language,
                        "chunk_index": c.chunk_index,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            manifest["partitions"][lang] = {
                "count": len(lang_chunks),
                "language": lang,
            }

        # Write manifest.json
        manifest_file = output_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

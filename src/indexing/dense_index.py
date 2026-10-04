import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

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
        chunks: Sequence[ChunkRecord] | Iterable[ChunkRecord],
        output_dir: Path | str,
        snapshot_id: str | None = None,
        source_membership_sha256: str | None = None,
        chunker_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Build partitioned dense vector indexes for chunks.

        Supports streaming Iterable[ChunkRecord] consuming chunks in a single pass
        and encoding in bounded batches without materializing all texts in memory.

        Args:
            chunks: Sequence or Iterable of ChunkRecord items.
            output_dir: Directory where indexes and metadata will be saved.
            snapshot_id: Optional snapshot identifier for provenance tracking.
            source_membership_sha256: Optional checksum of source documents.
            chunker_fingerprint: Optional chunker configuration fingerprint.

        Returns:
            Dictionary containing the index manifest.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        encoder = self._get_encoder()
        passage_prefix = self.instruction_prefix.get("passage", "")

        # Stream and accumulate in per-language buffers of batch_size
        buffers: dict[str, list[ChunkRecord]] = defaultdict(list)
        embeddings_by_lang: dict[str, list[np.ndarray]] = defaultdict(list)
        metadata_by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_chunk_ids_ordered: list[str] = []

        def flush_buffer(lang: str) -> None:
            buf = buffers[lang]
            if not buf:
                return
            texts_to_encode = [
                f"{passage_prefix}{c.text}" if passage_prefix else c.text
                for c in buf
            ]
            batch_emb = _encode_texts(
                encoder=encoder,
                texts=texts_to_encode,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
            )
            if batch_emb.size > 0:
                embeddings_by_lang[lang].append(batch_emb)
            for c in buf:
                metadata_by_lang[lang].append({
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "language": c.language,
                    "chunk_index": c.chunk_index,
                })
            buf.clear()

        for chunk in chunks:
            all_chunk_ids_ordered.append(chunk.chunk_id)
            lang = chunk.language
            buffers[lang].append(chunk)
            if len(buffers[lang]) >= self.batch_size:
                flush_buffer(lang)

        # Flush any remaining chunks in buffers
        for lang in list(buffers.keys()):
            flush_buffer(lang)

        # Union of default languages and any seen languages
        seen_langs = set(embeddings_by_lang.keys()) | set(metadata_by_lang.keys())
        languages = sorted(set(self.DEFAULT_LANGUAGES) | seen_langs)

        manifest: dict[str, Any] = {
            "backend": self.backend,
            "instruction_prefix": self.instruction_prefix,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
            "dim": None,
            "partitions": {},
        }
        if snapshot_id is not None:
            manifest["snapshot_id"] = snapshot_id
        if source_membership_sha256 is not None:
            manifest["source_membership_sha256"] = source_membership_sha256
        if chunker_fingerprint is not None:
            manifest["chunker_fingerprint"] = chunker_fingerprint

        mapping_repr = "".join(f"{i}:{cid}\n" for i, cid in enumerate(all_chunk_ids_ordered)).encode("utf-8")
        manifest["vector_chunk_mapping_hash"] = hashlib.sha256(mapping_repr).hexdigest()

        embedding_dim: int | None = None
        total_vectors = 0

        for lang in languages:
            meta_records = metadata_by_lang.get(lang, [])
            emb_batches = embeddings_by_lang.get(lang, [])

            if not meta_records:
                manifest["partitions"][lang] = {
                    "count": 0,
                    "language": lang,
                }
                continue

            if emb_batches:
                embeddings = np.concatenate(emb_batches, axis=0)
            else:
                embeddings = np.empty((0, 0), dtype=np.float32)

            count = len(meta_records)
            total_vectors += count

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
                for rec in meta_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            manifest["partitions"][lang] = {
                "count": count,
                "language": lang,
            }

        # Resource estimation
        dim_val = embedding_dim or 0
        manifest["resource_estimate"] = {
            "vector_count": total_vectors,
            "dim": dim_val,
            "bytes_per_vector": dim_val * 4,
            "estimated_vector_ram_bytes": total_vectors * dim_val * 4,
        }

        # Write manifest.json
        manifest_file = output_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

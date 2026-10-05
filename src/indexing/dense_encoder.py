from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class DenseEncoder:
    """Real multilingual E5-base dense encoder with prefix handling, masked pooling, L2 normalization, and OOM resilience."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        dimension: int = 768,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        normalize: bool = True,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading tokenizer and model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_mock_encoder(cls, dimension: int = 768) -> DenseEncoderMock:
        return DenseEncoderMock(dimension=dimension)

    @staticmethod
    def validate_vectors(vecs: np.ndarray) -> None:
        if not np.isfinite(vecs).all():
            raise ValueError("Vectors contain NaN or infinite values!")

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        prefixed = [prefix + t for t in texts]
        all_embeddings = []

        curr_batch_size = self.batch_size
        i = 0
        while i < len(prefixed):
            batch_texts = prefixed[i : i + curr_batch_size]
            try:
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    # Average pooling with attention mask for E5
                    token_embeddings = model_output.last_hidden_state
                    input_mask_expanded = (
                        encoded_input["attention_mask"]
                        .unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                    if self.normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                vecs = embeddings.cpu().to(torch.float32).numpy()
                self.validate_vectors(vecs)
                all_embeddings.append(vecs)
                i += curr_batch_size
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and curr_batch_size > 1:
                    curr_batch_size = max(1, curr_batch_size // 2)
                    logger.warning(f"CUDA OOM encountered. Reducing batch size to {curr_batch_size} and retrying.")
                    torch.cuda.empty_cache()
                else:
                    raise e

        if not all_embeddings:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack(all_embeddings)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), self.passage_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), self.query_prefix)


class DenseEncoderMock:
    """Mock encoder for testing environments without real model weights."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    @staticmethod
    def validate_vectors(vecs: np.ndarray) -> None:
        if not np.isfinite(vecs).all():
            raise ValueError("Vectors contain NaN or infinite values!")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(n, self.dimension)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, a_min=1e-9, a_max=None)
        self.validate_vectors(vecs)
        return vecs

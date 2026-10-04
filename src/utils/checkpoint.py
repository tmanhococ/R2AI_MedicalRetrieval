from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from src.data.schemas import FinalPrediction

logger = logging.getLogger(__name__)


def compute_run_fingerprint(
    query_manifest_or_ids: Any = None,
    snapshot_manifest_or_membership: Any = None,
    index_manifests: Sequence[dict[str, Any]] | None = None,
    model_revisions: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Computes a deterministic run_fingerprint incorporating query, corpus, index, model, and config parameters."""
    h = hashlib.sha256()

    # 1. Queries
    if query_manifest_or_ids is not None:
        if isinstance(query_manifest_or_ids, (list, tuple, set)):
            q_repr = ",".join(str(qid) for qid in sorted(query_manifest_or_ids))
            h.update(q_repr.encode("utf-8"))
        else:
            h.update(str(query_manifest_or_ids).encode("utf-8"))

    # 2. Snapshot / Corpus
    if snapshot_manifest_or_membership is not None:
        if hasattr(snapshot_manifest_or_membership, "source_membership_sha256"):
            h.update(str(snapshot_manifest_or_membership.source_membership_sha256).encode("utf-8"))
        elif hasattr(snapshot_manifest_or_membership, "snapshot_id"):
            h.update(str(snapshot_manifest_or_membership.snapshot_id).encode("utf-8"))
        else:
            h.update(str(snapshot_manifest_or_membership).encode("utf-8"))

    # 3. Index Manifests
    if index_manifests:
        for idx in index_manifests:
            serialized_idx = json.dumps(idx, sort_keys=True)
            h.update(serialized_idx.encode("utf-8"))

    # 4. Models
    if model_revisions:
        serialized_models = json.dumps(model_revisions, sort_keys=True)
        h.update(serialized_models.encode("utf-8"))

    # 5. Pipeline config settings (retrieval, fusion, scoring, post_processing, calibration)
    if config:
        critical_cfg = {
            "retrieval": config.get("retrieval", {}),
            "fusion": config.get("fusion", {}),
            "reranker": config.get("reranker", {}),
            "joint_scoring": config.get("joint_scoring", {}),
            "post_processing": config.get("post_processing", {}),
            "calibration": config.get("calibration", {}),
        }
        h.update(json.dumps(critical_cfg, sort_keys=True).encode("utf-8"))

    return h.hexdigest()


def _norm_qid(qid: Any) -> Any:
    if isinstance(qid, int) and not isinstance(qid, bool):
        return qid
    if isinstance(qid, str) and qid.isdigit():
        return int(qid)
    return str(qid)


class InferenceCheckpointManager:
    """Manages transactional per-query prediction checkpoints, resume semantics, and error blocking."""

    def __init__(
        self,
        output_dir: Path | str,
        run_fingerprint: str,
        resume: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_fingerprint = run_fingerprint
        self.checkpoint_path = self.output_dir / "predictions_checkpoint.jsonl"

        self.completed_predictions: dict[Any, FinalPrediction | dict[str, Any]] = {}
        self.failed_queries: dict[Any, str] = {}

        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupted checkpoint line {line_idx} in {self.checkpoint_path}")
                        continue

                    fp = record.get("run_fingerprint")
                    if fp is not None and fp != self.run_fingerprint:
                        if resume:
                            raise ValueError(
                                f"Run fingerprint mismatch in checkpoint: expected {self.run_fingerprint}, "
                                f"found {fp}. Cannot resume."
                            )

                    qid = _norm_qid(record["query_id"])
                    status = record.get("status")
                    if status == "completed":
                        self.completed_predictions[qid] = record.get("prediction")
                        self.failed_queries.pop(qid, None)
                    elif status == "failed":
                        self.failed_queries[qid] = str(record.get("error", "Unknown error"))
                        self.completed_predictions.pop(qid, None)

    def save_checkpoint(
        self,
        query_id: Any,
        status: str,
        prediction: FinalPrediction | dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Atomically appends a per-query checkpoint record."""
        now_iso = datetime.now(timezone.utc).isoformat()
        norm_id = _norm_qid(query_id)
        pred_dict = None
        if prediction is not None:
            if isinstance(prediction, FinalPrediction):
                pred_dict = {
                    "query_id": prediction.query_id,
                    "document_ids": list(prediction.document_ids),
                    "chunk_ids": list(prediction.chunk_ids),
                    "document_scores": dict(prediction.document_scores),
                    "chunk_scores": dict(prediction.chunk_scores),
                }
            elif isinstance(prediction, dict):
                pred_dict = prediction

        record = {
            "run_fingerprint": self.run_fingerprint,
            "query_id": query_id,
            "status": status,
            "prediction": pred_dict,
            "error": error,
            "timestamp": now_iso,
        }

        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        if status == "completed":
            self.completed_predictions[norm_id] = pred_dict
            self.failed_queries.pop(norm_id, None)
        elif status == "failed":
            self.failed_queries[norm_id] = str(error)
            self.completed_predictions.pop(norm_id, None)

    def get_completed_query_ids(self) -> set[Any]:
        return set(self.completed_predictions.keys())

    def get_failed_query_ids(self) -> set[Any]:
        return set(self.failed_queries.keys())

    def has_failed_queries(self) -> bool:
        return len(self.failed_queries) > 0

    def validate_and_get_all_predictions(
        self,
        required_query_ids: Sequence[Any],
    ) -> list[FinalPrediction]:
        """Validates that all required queries completed without failures and returns FinalPrediction objects."""
        if self.has_failed_queries():
            failed_sample = list(self.failed_queries.keys())[:5]
            raise RuntimeError(
                f"Query inference failed for {len(self.failed_queries)} queries (sample: {failed_sample}). "
                f"Blocking final export."
            )

        norm_required = [_norm_qid(q) for q in required_query_ids]
        missing = set(norm_required) - set(self.completed_predictions.keys())
        if missing:
            missing_sample = sorted(list(missing))[:5]
            raise RuntimeError(
                f"Missing predictions for {len(missing)} required queries (sample: {missing_sample}). "
                f"Blocking final export."
            )

        results: list[FinalPrediction] = []
        for qid in norm_required:
            raw = self.completed_predictions[qid]
            if isinstance(raw, FinalPrediction):
                results.append(raw)
            else:
                results.append(
                    FinalPrediction(
                        query_id=str(raw["query_id"]),
                        document_ids=list(raw.get("document_ids", [])),
                        chunk_ids=list(raw.get("chunk_ids", [])),
                        document_scores=dict(raw.get("document_scores", {})),
                        chunk_scores=dict(raw.get("chunk_scores", {})),
                    )
                )

        return results

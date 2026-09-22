"""Cutoff Calibration Engine for Retrieval MVP.

ANTI-LEAKAGE PRINCIPLE:
Calibration of cutoff policies (fixed_top_k, global_threshold, relative_threshold, score_gap)
MUST be performed strictly on validation data. Calibration parameters must NEVER be tuned
or fitted on test evaluation queries to prevent optimistic bias and test set data leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from src.data.schemas import GroundTruthRecord, ScoredHit
from src.evaluation.metrics import calculate_precision_recall_fbeta


@dataclass
class CalibrationState:
    """State resulting from cutoff parameter calibration.

    Attributes:
        item_type: Target item type ('document' or 'chunk').
        best_policy: Name of the optimal cutoff policy.
        best_params: Parameter dictionary for the optimal policy.
        best_f2: Best macro F2 score achieved on validation data.
        all_policy_results: Evaluation summary for all tested policy configurations.
    """

    item_type: str
    best_policy: str
    best_params: dict[str, Any]
    best_f2: float
    all_policy_results: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration state to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationState:
        """Construct CalibrationState from dictionary."""
        return cls(
            item_type=str(data["item_type"]),
            best_policy=str(data["best_policy"]),
            best_params=dict(data["best_params"]),
            best_f2=float(data["best_f2"]),
            all_policy_results=dict(data.get("all_policy_results", {})),
        )

    def save(self, filepath: str | Path) -> None:
        """Serialize state to a JSON file."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> CalibrationState:
        """Load state from a JSON file."""
        source_path = Path(filepath)
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _extract_id_and_score(item: Any) -> tuple[str, float]:
    """Extract item ID and score from various item formats (ScoredHit, tuple, dict, float)."""
    if isinstance(item, ScoredHit):
        score = item.normalized_score if item.normalized_score is not None else item.rerank_score
        return item.item_id, float(score)
    if isinstance(item, (tuple, list)):
        return str(item[0]), float(item[1])
    if isinstance(item, (int, float)):
        return "", float(item)
    if isinstance(item, dict):
        item_id = str(item.get("item_id", item.get("doc_id", item.get("id", ""))))
        score = float(item.get("normalized_score", item.get("score", item.get("rerank_score", 0.0))))
        return item_id, score
    item_id = str(getattr(item, "item_id", getattr(item, "doc_id", getattr(item, "id", ""))))
    score = float(getattr(item, "normalized_score", getattr(item, "score", getattr(item, "rerank_score", 0.0))))
    return item_id, score


class CutoffCalibrator:
    """Anti-leakage Cutoff Calibrator supporting multiple cutoff policies.

    Supported policies:
    1. fixed_top_k: Selects fixed number of top items K in [K_min, K_max].
    2. global_threshold: Keeps items with score >= T (T in [0.0, 1.0]).
    3. relative_threshold: Keeps items with score >= lambda * score_top_1 (lambda in [0.1, 1.0]).
    4. score_gap: Identifies largest score drop s_i - s_{i+1} within bounds [N_min, N_max].
    """

    SUPPORTED_POLICIES = ("fixed_top_k", "global_threshold", "relative_threshold", "score_gap")

    def __init__(
        self,
        beta: float = 2.0,
        policies: Sequence[str] | None = None,
        bounds: dict[str, Any] | None = None,
    ) -> None:
        """Initialize CutoffCalibrator.

        Args:
            beta: Beta weight for F_beta evaluation (default 2.0 for Macro F2).
            policies: List of policy names to tune. Defaults to all 4 supported policies.
            bounds: Parameter search bounds for candidate policies.
        """
        self.beta = beta
        self.policies = list(policies) if policies is not None else list(self.SUPPORTED_POLICIES)
        self.bounds = bounds or {}

    def _generate_candidate_params(self, policy: str) -> list[dict[str, Any]]:
        """Generate grid of parameter configurations for a given policy."""
        if policy == "fixed_top_k":
            b = self.bounds.get("fixed_top_k")
            if isinstance(b, (tuple, list)) and len(b) == 2 and isinstance(b[0], int) and isinstance(b[1], int):
                k_vals = list(range(b[0], b[1] + 1))
            elif isinstance(b, list):
                k_vals = b
            elif isinstance(b, dict):
                k_vals = list(range(b.get("min_k", 1), b.get("max_k", 10) + 1))
            else:
                k_vals = list(range(1, 11))
            return [{"k": k} for k in k_vals]

        elif policy == "global_threshold":
            b = self.bounds.get("global_threshold")
            if isinstance(b, (tuple, list)) and len(b) == 2 and isinstance(b[0], (int, float)) and isinstance(b[1], (int, float)):
                min_t, max_t = float(b[0]), float(b[1])
                t_vals = [round(min_t + i * 0.05, 3) for i in range(int((max_t - min_t) / 0.05) + 1)]
            elif isinstance(b, list):
                t_vals = b
            elif isinstance(b, dict):
                min_t = b.get("min_t", 0.0)
                max_t = b.get("max_t", 1.0)
                step = b.get("step", 0.05)
                t_vals = [round(min_t + i * step, 3) for i in range(int((max_t - min_t) / step) + 1)]
            else:
                t_vals = [round(x * 0.05, 2) for x in range(1, 20)]
            return [{"threshold": t} for t in t_vals]

        elif policy == "relative_threshold":
            b = self.bounds.get("relative_threshold")
            if isinstance(b, (tuple, list)) and len(b) == 2 and isinstance(b[0], (int, float)) and isinstance(b[1], (int, float)):
                min_r, max_r = float(b[0]), float(b[1])
                r_vals = [round(min_r + i * 0.05, 3) for i in range(int((max_r - min_r) / 0.05) + 1)]
            elif isinstance(b, list):
                r_vals = b
            elif isinstance(b, dict):
                min_r = b.get("min_ratio", 0.1)
                max_r = b.get("max_ratio", 1.0)
                step = b.get("step", 0.05)
                r_vals = [round(min_r + i * step, 3) for i in range(int((max_r - min_r) / step) + 1)]
            else:
                r_vals = [round(x * 0.05, 2) for x in range(2, 21)]
            return [{"lambda_ratio": r} for r in r_vals]

        elif policy == "score_gap":
            b = self.bounds.get("score_gap")
            if isinstance(b, dict):
                min_n = b.get("min_n", 1)
                max_n = b.get("max_n", 10)
                gap_pairs = [(min_n, max_n)]
            elif isinstance(b, (tuple, list)) and len(b) == 2 and isinstance(b[0], int) and isinstance(b[1], int):
                gap_pairs = [(b[0], b[1])]
            elif isinstance(b, list):
                gap_pairs = b
            else:
                gap_pairs = [
                    (1, 3),
                    (1, 5),
                    (1, 10),
                    (2, 5),
                    (2, 10),
                    (3, 10),
                ]
            return [{"n_min": pair[0], "n_max": pair[1]} for pair in gap_pairs]

        else:
            raise ValueError(f"Unsupported policy: {policy}")

    def _apply_policy(self, ranked_items: Sequence[Any], policy: str, params: dict[str, Any]) -> int:
        """Apply a cutoff policy to ranked items and return cutoff index K."""
        if not ranked_items:
            return 0

        n = len(ranked_items)

        if policy == "fixed_top_k":
            k = params.get("k", n)
            return max(0, min(k, n))

        elif policy == "global_threshold":
            threshold = float(params.get("threshold", 0.0))
            count = 0
            for it in ranked_items:
                _, score = _extract_id_and_score(it)
                if score >= threshold - 1e-9:
                    count += 1
                else:
                    break
            return count

        elif policy == "relative_threshold":
            ratio = float(params.get("lambda_ratio", params.get("lambda", 0.5)))
            _, top_score = _extract_id_and_score(ranked_items[0])
            threshold = ratio * top_score
            count = 0
            for it in ranked_items:
                _, score = _extract_id_and_score(it)
                if score >= threshold - 1e-9:
                    count += 1
                else:
                    break
            return count

        elif policy == "score_gap":
            if n <= 1:
                return n
            n_min = params.get("n_min", 1)
            n_max = params.get("n_max", n)
            effective_min = max(1, min(n_min, n))
            effective_max = min(n_max, n - 1)
            if effective_min > effective_max:
                return min(n, effective_min)

            scores = [_extract_id_and_score(it)[1] for it in ranked_items]
            best_i = effective_min
            max_drop = -float("inf")
            for i in range(effective_min, effective_max + 1):
                # 1-based index i: drop between i-th item and (i+1)-th item
                drop = scores[i - 1] - scores[i]
                if drop > max_drop:
                    max_drop = drop
                    best_i = i
            return best_i

        else:
            raise ValueError(f"Unknown cutoff policy: {policy}")

    def fit(
        self,
        query_scores: dict[str, Sequence[Any]],
        ground_truths: Sequence[GroundTruthRecord] | dict[str, GroundTruthRecord],
        item_type: str = "document",
    ) -> CalibrationState:
        """Calibrate cutoff policy using validation set queries to maximize Macro F2.

        ANTI-LEAKAGE NOTE:
        This fitting procedure must ONLY be executed on validation queries. Fitting on test
        data constitutes leakage and will invalidate evaluation integrity.

        Args:
            query_scores: Mapping from query_id to sequence of ranked items (ScoredHit or (id, score)).
            ground_truths: GroundTruthRecord instances for validation queries.
            item_type: 'document' or 'chunk'.

        Returns:
            CalibrationState with the best policy and parameter configuration.
        """
        # Normalize ground truths
        if isinstance(ground_truths, dict):
            gt_dict = ground_truths
        else:
            gt_dict = {gt.query_id: gt for gt in ground_truths}

        num_queries = len(gt_dict)
        if num_queries == 0:
            return CalibrationState(
                item_type=item_type,
                best_policy=self.policies[0] if self.policies else "fixed_top_k",
                best_params={"k": 5},
                best_f2=0.0,
                all_policy_results={},
            )

        # Pre-extract ground truth target sets
        gt_target_ids: dict[str, set[str]] = {}
        for qid, gt_rec in gt_dict.items():
            if item_type == "chunk":
                gt_target_ids[qid] = gt_rec.relevant_chunk_ids
            else:
                gt_target_ids[qid] = gt_rec.relevant_doc_ids

        # Pre-extract item IDs and scores per query
        preprocessed_candidates: dict[str, list[tuple[str, float]]] = {}
        for qid in gt_dict.keys():
            items = query_scores.get(qid, [])
            preprocessed_candidates[qid] = [_extract_id_and_score(it) for it in items]

        all_policy_results: dict[str, dict[str, Any]] = {}
        candidates_evaluated: list[tuple[float, float, int, Any, str, dict[str, Any]]] = []

        for policy_idx, policy in enumerate(self.policies):
            all_policy_results[policy] = {}
            params_list = self._generate_candidate_params(policy)

            for params in params_list:
                sum_f2 = 0.0
                sum_recall = 0.0
                sum_precision = 0.0

                for qid, gt_set in gt_target_ids.items():
                    ranked_items = preprocessed_candidates.get(qid, [])
                    cutoff = self._apply_policy(ranked_items, policy, params)
                    pred_ids = [ranked_items[i][0] for i in range(cutoff)]
                    metrics = calculate_precision_recall_fbeta(pred_ids, gt_set, beta=self.beta)
                    sum_f2 += metrics["f2"]
                    sum_recall += metrics["recall"]
                    sum_precision += metrics["precision"]

                macro_f2 = sum_f2 / num_queries
                macro_recall = sum_recall / num_queries
                macro_precision = sum_precision / num_queries

                param_key = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
                all_policy_results[policy][param_key] = {
                    "macro_f2": round(macro_f2, 5),
                    "macro_recall": round(macro_recall, 5),
                    "macro_precision": round(macro_precision, 5),
                    "params": params,
                }

                # Tie-breaking key:
                # 1. Higher Macro F2 (-macro_f2)
                # 2. Higher Macro Recall (-macro_recall)
                # 3. Policy index in self.policies (preserves policy order)
                # 4. Smaller parameter value
                if policy == "fixed_top_k":
                    param_val = params.get("k", 0)
                elif policy == "global_threshold":
                    param_val = params.get("threshold", 0.0)
                elif policy == "relative_threshold":
                    param_val = params.get("lambda_ratio", 0.0)
                elif policy == "score_gap":
                    param_val = (params.get("n_min", 0), params.get("n_max", 0))
                else:
                    param_val = 0

                candidates_evaluated.append((
                    -macro_f2,
                    -macro_recall,
                    policy_idx,
                    param_val,
                    policy,
                    params,
                ))

        # Sort candidates deterministically according to tie-breaking specification
        candidates_evaluated.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        best_candidate = candidates_evaluated[0]
        best_neg_f2, _, _, _, best_policy, best_params = best_candidate

        return CalibrationState(
            item_type=item_type,
            best_policy=best_policy,
            best_params=best_params,
            best_f2=-best_neg_f2,
            all_policy_results=all_policy_results,
        )

    def apply(self, ranked_items: Sequence[Any], state: CalibrationState | dict[str, Any]) -> int:
        """Apply calibrated state to ranked items and return cutoff index K.

        Args:
            ranked_items: Ranked candidate items.
            state: CalibrationState instance or serialized dict.

        Returns:
            Integer cutoff index K indicating how many top items to retain.
        """
        if isinstance(state, dict):
            state = CalibrationState.from_dict(state)
        return self._apply_policy(ranked_items, state.best_policy, state.best_params)

    def filter_items(self, ranked_items: Sequence[Any], state: CalibrationState | dict[str, Any]) -> list[Any]:
        """Filter ranked items using calibrated state cutoff.

        Args:
            ranked_items: Ranked candidate items.
            state: CalibrationState instance or serialized dict.

        Returns:
            Sliced list containing top items according to calibrated cutoff.
        """
        cutoff = self.apply(ranked_items, state)
        return list(ranked_items[:cutoff])

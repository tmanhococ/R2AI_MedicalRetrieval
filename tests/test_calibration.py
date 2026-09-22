from __future__ import annotations

from pathlib import Path
import pytest

from src.data.schemas import GroundTruthRecord, ScoredHit
from src.evaluation.calibrator import CalibrationState, CutoffCalibrator


def test_calibrator_selects_best_policy():
    # Synthetic query candidates and ground truth
    query_scores = {
        "q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.3)],
        "q2": [("d4", 0.85), ("d5", 0.2)],
    }
    gt = [
        GroundTruthRecord(query_id="q1", relevant_doc_ids={"d1", "d2"}, relevant_chunk_ids=set()),
        GroundTruthRecord(query_id="q2", relevant_doc_ids={"d4"}, relevant_chunk_ids=set()),
    ]
    calibrator = CutoffCalibrator(beta=2.0)
    state = calibrator.fit(query_scores, gt, item_type="document")
    assert state.best_policy in {"fixed_top_k", "global_threshold", "relative_threshold", "score_gap"}
    assert state.best_f2 > 0.8
    assert state.item_type == "document"


def test_cutoff_policies_apply_logic():
    calibrator = CutoffCalibrator()
    scores = [("d1", 0.95), ("d2", 0.80), ("d3", 0.40), ("d4", 0.10)]

    # 1. Fixed Top-K
    state_top_k = CalibrationState(
        item_type="document",
        best_policy="fixed_top_k",
        best_params={"k": 2},
        best_f2=1.0,
        all_policy_results={},
    )
    assert calibrator.apply(scores, state_top_k) == 2
    assert calibrator.filter_items(scores, state_top_k) == [("d1", 0.95), ("d2", 0.80)]

    # 2. Global Threshold
    state_global = CalibrationState(
        item_type="document",
        best_policy="global_threshold",
        best_params={"threshold": 0.50},
        best_f2=1.0,
        all_policy_results={},
    )
    assert calibrator.apply(scores, state_global) == 2
    assert calibrator.filter_items(scores, state_global) == [("d1", 0.95), ("d2", 0.80)]

    # 3. Relative Threshold: 0.95 * 0.80 = 0.76 -> d1 (0.95) and d2 (0.80) >= 0.76
    state_relative = CalibrationState(
        item_type="document",
        best_policy="relative_threshold",
        best_params={"lambda_ratio": 0.80},
        best_f2=1.0,
        all_policy_results={},
    )
    assert calibrator.apply(scores, state_relative) == 2

    # 4. Score Gap: drops:
    # i=1: 0.95 - 0.80 = 0.15
    # i=2: 0.80 - 0.40 = 0.40 (largest)
    # i=3: 0.40 - 0.10 = 0.30
    state_gap = CalibrationState(
        item_type="document",
        best_policy="score_gap",
        best_params={"n_min": 1, "n_max": 3},
        best_f2=1.0,
        all_policy_results={},
    )
    assert calibrator.apply(scores, state_gap) == 2


def test_calibrator_with_scored_hit_objects():
    calibrator = CutoffCalibrator()
    hits = [
        ScoredHit("c1", "chunk", "d1", "vi", 0.9, 0.9, 0.5),
        ScoredHit("c2", "chunk", "d1", "vi", 0.8, 0.8, 0.4),
        ScoredHit("c3", "chunk", "d2", "vi", 0.3, 0.3, 0.1),
    ]
    state = CalibrationState(
        item_type="chunk",
        best_policy="global_threshold",
        best_params={"threshold": 0.75},
        best_f2=1.0,
        all_policy_results={},
    )
    cutoff = calibrator.apply(hits, state)
    assert cutoff == 2
    filtered = calibrator.filter_items(hits, state)
    assert len(filtered) == 2
    assert [h.item_id for h in filtered] == ["c1", "c2"]


def test_calibration_state_serialization_roundtrip(tmp_path: Path):
    state = CalibrationState(
        item_type="document",
        best_policy="global_threshold",
        best_params={"threshold": 0.65},
        best_f2=0.925,
        all_policy_results={
            "global_threshold": {"threshold=0.65": {"macro_f2": 0.925, "macro_recall": 0.90}},
            "fixed_top_k": {"k=2": {"macro_f2": 0.85, "macro_recall": 0.80}},
        },
    )
    file_path = tmp_path / "calibration.json"
    state.save(file_path)
    assert file_path.exists()

    loaded = CalibrationState.load(file_path)
    assert loaded.item_type == state.item_type
    assert loaded.best_policy == state.best_policy
    assert loaded.best_params == state.best_params
    assert loaded.best_f2 == pytest.approx(state.best_f2)
    assert loaded.all_policy_results == state.all_policy_results

    # Test to_dict / from_dict directly
    d = state.to_dict()
    reconstructed = CalibrationState.from_dict(d)
    assert reconstructed == loaded


def test_calibrator_chunk_item_type():
    query_scores = {
        "q1": [("c1_1", 0.9), ("c1_2", 0.7), ("c1_3", 0.2)],
    }
    gt = [
        GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids={"d1"},
            relevant_chunk_ids={"c1_1", "c1_2"},
        )
    ]
    calibrator = CutoffCalibrator(beta=2.0)
    state = calibrator.fit(query_scores, gt, item_type="chunk")
    assert state.item_type == "chunk"
    assert state.best_f2 == pytest.approx(1.0)


def test_anti_leakage_documentation_present():
    import src.evaluation.calibrator as cal_module

    # Must mention validation requirement to prevent test set data leakage
    assert "validation" in cal_module.__doc__.lower()
    assert "leakage" in cal_module.__doc__.lower() or "leakage" in cal_module.CutoffCalibrator.fit.__doc__.lower()


def test_calibrator_custom_bounds_support():
    custom_bounds = {
        "fixed_top_k": {"min_k": 2, "max_k": 4},
        "global_threshold": {"min_t": 0.4, "max_t": 0.8, "step": 0.2},
        "relative_threshold": {"min_ratio": 0.5, "max_ratio": 0.9, "step": 0.2},
        "score_gap": {"min_n": 1, "max_n": 3},
    }
    calibrator = CutoffCalibrator(bounds=custom_bounds)
    k_params = calibrator._generate_candidate_params("fixed_top_k")
    assert [p["k"] for p in k_params] == [2, 3, 4]

    gt_params = calibrator._generate_candidate_params("global_threshold")
    assert [p["threshold"] for p in gt_params] == [0.4, 0.6, 0.8]

    rt_params = calibrator._generate_candidate_params("relative_threshold")
    assert [p["lambda_ratio"] for p in rt_params] == [0.5, 0.7, 0.9]

    sg_params = calibrator._generate_candidate_params("score_gap")
    assert sg_params == [{"n_min": 1, "n_max": 3}]


def test_calibrator_empty_and_single_edge_cases():
    calibrator = CutoffCalibrator()
    state = CalibrationState(
        item_type="document",
        best_policy="score_gap",
        best_params={"n_min": 1, "n_max": 5},
        best_f2=1.0,
        all_policy_results={},
    )
    # Empty items
    assert calibrator.apply([], state) == 0
    assert calibrator.filter_items([], state) == []

    # Single item
    assert calibrator.apply([("d1", 0.9)], state) == 1
    assert calibrator.filter_items([("d1", 0.9)], state) == [("d1", 0.9)]


def test_calibrator_deterministic_tie_breaking():
    # Only test fixed_top_k
    calibrator = CutoffCalibrator(
        policies=["fixed_top_k"],
        bounds={"fixed_top_k": [1, 2, 3]},
    )
    # Suppose GT is {"d1"}.
    # k=1: predicts ["d1"] -> P=1.0, R=1.0, F2=1.0
    # k=2: predicts ["d1", "d2"] -> P=0.5, R=1.0, F2=5*0.5*1 / (4*0.5+1) = 2.5/3 = 0.833
    # Both have R=1.0, but k=1 has higher F2.
    query_scores = {"q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]}
    gt = [GroundTruthRecord(query_id="q1", relevant_doc_ids={"d1"}, relevant_chunk_ids=set())]
    state = calibrator.fit(query_scores, gt)
    assert state.best_params["k"] == 1
    assert state.best_f2 == pytest.approx(1.0)


def test_calibrator_with_pure_floats_and_discrete_bounds():
    calibrator = CutoffCalibrator(
        policies=["fixed_top_k", "relative_threshold"],
        bounds={
            "fixed_top_k": (1, 3, 5),  # discrete tuple
            "relative_threshold": (0.2, 0.8),
        },
    )
    state = CalibrationState(
        item_type="document",
        best_policy="relative_threshold",
        best_params={"lambda_ratio": 0.5},
        best_f2=0.9,
    )
    # Apply on pure float list
    cutoff = calibrator.apply([0.9, 0.8, 0.3, 0.1], state)
    assert cutoff == 2
    filtered = calibrator.filter_items([0.9, 0.8, 0.3, 0.1], state)
    assert filtered == [0.9, 0.8]


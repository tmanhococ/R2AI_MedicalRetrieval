from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Union
import yaml

ALLOWED_CONFIG_SECTIONS = {
    "run": {
        "experiment_name", "seed", "output_dir", "require_real_models"
    },
    "dataset": {
        "adapter", "query_path", "source_path", "expected_query_count", "registry_path", "dir",
        "expected_official_count", "verify_actual_counts", "scope", "accepted_languages"
    },
    "campaign": {
        "scope", "domain_whitelist", "source_limit", "reuse_existing_content",
        "require_no_unresolved_sources", "priority_changes_order_only"
    },
    "resources": {
        "embedding_start_batch", "reranker_start_batch", "oom_min_batch",
        "concurrent_gpu_models", "java_heap", "row_group_cache_bytes", "disk_reserve_fraction"
    },
    "crawl": {
        "concurrency", "per_domain_concurrency", "min_domain_interval_seconds",
        "connect_timeout_seconds", "read_timeout_seconds", "total_deadline_seconds",
        "max_attempts", "base_backoff_seconds", "lease_seconds", "max_redirects",
        "max_html_bytes", "max_pdf_bytes", "respect_robots", "raw_root", "timeout"
    },
    "extraction": {
        "html_backend", "pdf_backend", "language_model_path",
        "language_confidence_min", "accepted_languages", "quarantine_unknown_language"
    },
    "indexing": {
        "snapshot_root", "parquet_row_group_size", "target_shard_bytes",
        "chunking", "validation"
    },
    "query_understanding": {
        "normalize", "adaptive_routing", "translation"
    },
    "retrieval": {
        "candidate_top_k", "bm25", "dense", "sparse", "candidate_docs"
    },
    "fusion": {
        "method", "rrf_k", "weights", "family_weights", "group_correlated_dense_variants"
    },
    "evidence": {
        "document_top_k", "base_windows_per_doc", "base_spans_per_query_max",
        "scan_all_chunks_in_candidate_doc", "neighbor_blocks", "expanded_soft_target_tokens",
        "expansion_logit_tolerance", "require_positive_neighbor", "dedup_span_overlap",
        "preserve_source_offsets"
    },
    "reranker": {
        "enabled", "model_name", "document_strategy", "document_top_k",
        "chunk_top_k", "batch_size", "max_length", "query_window_tokens",
        "query_window_overlap_tokens", "query_window_aggregation", "body_truncation",
        "passage_window_tokens", "query_window_overlap", "window_aggregation",
        "use_raw_logits", "require_real_weights"
    },
    "joint_scoring": {
        "document", "chunk"
    },
    "post_processing": {
        "consistency_mode"
    },
    "evaluation": {
        "profile", "normalization", "tokenizer", "overlap_threshold",
        "undefined_pr", "empty_both_f2", "scorer_parity_verified", "beta"
    },
    "calibration": {
        "metric", "beta", "policies", "bounds", "fit_enabled",
        "fallback_policy", "fixed_top_k"
    },
    "output": {
        "fit_only_allowed_dev", "fallback_policy", "fallback_raw_logit_min",
        "min_results", "document_cap", "evidence_cap", "evidence_cap_per_doc",
        "fixed_count_validation"
    },
    "ann_audit": {
        "sample_queries", "full_audit_queries", "top_k", "candidate_retention_min"
    },
    "submission": {
        "exporter", "doc_id_type", "emit_chunk_order", "json_filename",
        "zip_filename", "provenance_check", "require_scope",
        "require_complete_sparse_membership", "require_complete_dense_membership",
        "require_completed_queries", "report_terminal_gaps"
    }
}


def merge_configs(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge overlay configuration on top of base configuration."""
    merged = copy.deepcopy(base)
    for key, val in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Strictly validates configuration keys against allowed schema.

    Raises:
        ValueError: If unknown top-level sections or unrecognized configuration keys are found.
    """
    for section_name, section_val in config.items():
        if section_name not in ALLOWED_CONFIG_SECTIONS:
            raise ValueError(f"Unknown top-level configuration section: {section_name!r}")

        if isinstance(section_val, dict):
            allowed_subkeys = ALLOWED_CONFIG_SECTIONS[section_name]
            for subkey in section_val.keys():
                if subkey not in allowed_subkeys:
                    raise ValueError(
                        f"Unknown configuration key {subkey!r} in section {section_name!r}"
                    )


def load_config(
    config_path: Union[str, Path],
    base_config_path: Union[str, Path, None] = None,
    validate: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and parse a YAML configuration file with support for base overlay and deep overrides.

    Args:
        config_path: Path to the primary YAML configuration file.
        base_config_path: Optional path to a base configuration file to merge with.
        validate: Whether to validate the merged configuration against schema.
        overrides: Optional runtime dictionary overrides (highest precedence).

    Returns:
        dict: Parsed and merged configuration dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValueError: If the config contains unknown sections or keys when validate=True.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
        cfg = loaded if isinstance(loaded, dict) else {}

    # If base_config_path is provided, or if config_path is vibiomir_phase1.yaml overlay,
    # load and merge base config (configs/r2ai_mvp.yaml)
    base_path = Path(base_config_path) if base_config_path is not None else None
    if base_path is None and path.name == "vibiomir_phase1.yaml":
        candidate_base = path.parent / "r2ai_mvp.yaml"
        if candidate_base.exists():
            base_path = candidate_base

    if base_path is not None and base_path.exists() and base_path.resolve() != path.resolve():
        with open(base_path, "r", encoding="utf-8") as bf:
            base_loaded = yaml.safe_load(bf)
            base_dict = base_loaded if isinstance(base_loaded, dict) else {}
            cfg = merge_configs(base_dict, cfg)

    if overrides:
        cfg = merge_configs(cfg, overrides)

    if validate:
        validate_config(cfg)

    return cfg

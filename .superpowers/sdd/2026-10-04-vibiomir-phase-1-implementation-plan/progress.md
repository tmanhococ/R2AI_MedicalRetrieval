# SDD ledger — plan: docs/superpowers/plans/2026-10-04-vibiomir-phase-1-implementation-plan.md
Task 0: complete (commits 7e7ed6b..0963a6a, review clean)
Task 1: minor (deferred): OfficialSourceIterator implements Iterable instead of Iterator (__next__)
Task 1: complete (commits ca62aa0..61e6719, review clean)
Task 2: complete (commit 08c1fd5, review clean)
Task 2: minor (deferred): Pass config.max_redirects in crawl_runner managed client
Task 2: minor (deferred): Add User-Agent header in RobotsManager.is_allowed
Task 2: complete (commits 68751ee..08c1fd5, review clean)
Task 3: fix round 1/5 (2 addressed, 0 open — semantic article headers preserved, soft-404 regex bounded; commits 66ca27b..b48af26)
Task 3: complete (commits f7c6f22..b48af26, review clean)
Task 4: complete (commit 9d4950e, review clean)

Task 4: minor (deferred): Add clamp on len(input_ids) <= max_length in pack_reranker_inputs for separator margin
Task 4: complete (commits b48af26..9d4950e, review clean)
Task 5: fix round 1/5 (4 addressed, 0 open — strict provenance hashes, CLI streaming, cursor batching, tests added; commits 6bf23a4..f816295)
Task 5: minor (deferred): iter_documents generator re-instantiation if multi-language partitions iterate sequentially
Task 5: complete (commits 9d4950e..f816295, review clean)
Task 6: minor (deferred): Handle float conversion via Fraction(str(threshold)) in overlap_matches
Task 6: complete (commits f816295..e3474f3, review clean)
Task 7: minor (deferred): Calibrator constructor default fit_enabled align with config fit_enabled: false once legacy fixtures parameterized
Task 7: minor (deferred): PerQueryMetrics separate by_id lookup from positional indexing
Task 7: complete (commits e3474f3..86abaec, review clean)
Task 8: complete (commit 4a59f22, review clean)

Task 8: minor (deferred): Add explicit bool check before int() in exporter resolution
Task 8: minor (deferred): Add strict dictionary allowlist key check in validate_predictions
Task 8: complete (commits 86abaec..4a59f22, review clean)
Task 9: fix round 1/5 (2 addressed, 0 open — non-resume checkpoint handling, completed integration assertions; commits 0624030..6726f1f)
Task 9: complete (commits 4a59f22..6726f1f, review clean)
Task 10: complete (commits 6726f1f..dfb4a8f, review clean)
Task 11: fix round 1/5 (5 addressed, 0 open — stratified sampling 500 URLs seed 42, 30 sources / 20 chunks audit, coverage.json, scorer parity and RAM/GPU reporting, ZIP size; commits 592d637..7c18ab3)
Task 11: fix round 2/5 (2 addressed, 0 open — test isolation for experiment coverage.json, tracemalloc try-finally guard; commits 7c18ab3..190a7f2)
Task 11: complete (commits dfb4a8f..190a7f2, review clean)

# SDD ledger — plan: docs/superpowers/plans/2026-10-05-vibiomir-phase-2-full-corpus-implementation-plan.md

- [x] Task 0: Audit baseline và benchmark hardware thật (Commit: a523e85)
Task 0: minor (deferred): JAVA_HOME check in detect_java_version for portability
Task 0: minor (deferred): auto_out.parent.mkdir(parents=True, exist_ok=True) in doctor.py for doctor_report.json
Task 0: minor (deferred): direct unit test for measure_process_tree_rss
Task 0: complete (commits 94cbc51..a523e85, review clean)
- [x] Task 1: Full coverage ledger và strict scope guards (Commits: ca217a5, f3e77e2)
Task 1: complete (commits ca217a5..f3e77e2, review clean, 383/383 tests passed)


Task 1: fix round 1/5 (2 addressed, 0 open — non-empty hash, explicit allowed scopes; commits ca217a5..f3e77e2)
Task 1: complete (commits a523e85..f3e77e2, review clean)
Task 2: fix round 1/5 (5 addressed, 0 open — retry loop, inflight leases, resume flag, priority boost, batch extraction tests; commits 67d17fa..ccf271c)
Task 2: complete (commits f3e77e2..ccf271c, review clean)
Task 3: fix round 1/5 (3 addressed, 0 open — token budget re-chunking, thread-safe LRU cache, span bounds validation; commits b4e1b03..788c276)
Task 3: complete (commits ccf271c..788c276, review clean)
Task 4: complete (commits 788c276..9e990c1, review clean)
Task 5: complete (commits 77dcae9..54b8629, review clean)
Task 6: complete (commits f57b7fb..83c916b, review clean)
Task 7: minor (deferred): In src/cli/generate_candidates.py, remove redundant translator instantiation
Task 7: complete (commits 66b9b69..9fa13d2, review clean)

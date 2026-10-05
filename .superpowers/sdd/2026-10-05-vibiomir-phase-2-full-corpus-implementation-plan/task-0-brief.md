### Task 0: Audit baseline và benchmark hardware thật

**Files:** Read Phase 1 source/config/manifest/tests; create operations/resources.py, cli/doctor.py, tests/operations/test_resource_report.py.

**Interfaces:** inspect_resources → ResourceReport; reuse existing package manager; source types/method signatures chốt theo mục11.

- [ ] Verify repo branch/commit thật; re-run Phase 1 regression. Inspect baseline index membership, actual tokenizer, predictor backend, model loading và exporter validator.
- [ ] Test ResourceReport CPU/GPU/native memory sampling và missing CUDA handling; run pytest tests/operations/test_resource_report.py -q, expected FAIL trước implement.
- [ ] Implement doctor đo process RSS gồm JVM worker riêng, disk free, CUDA allocated/reserved, detected VRAM. Không dùng tracemalloc như total RAM.
- [ ] Smoke-load E5-base và BGE model lần lượt; 32 passages/32 pairs, log outputs không constant/random, batch throughput/OOM. Pin revisions/package versions sau smoke.
- [ ] Verify Pyserini worker Python3.12/Java21 route; nếu core khác versions, subprocess worker + JSONL protocol, không phá core environment.
- [ ] Run tests và doctor; expected valid report và model checkpoints load thật, hoặc blocker cụ thể. Commit: feat: add Phase 2 hardware and runtime diagnostics.


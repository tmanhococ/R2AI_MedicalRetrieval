### Task 6: Exact normalization, LCS và text match profile

**Files:** Create evaluation/vibiomir_text.py; tests/evaluation/test_vibiomir_text.py.

**Interfaces:** Bốn functions tại mục 7.1; evaluator profile sử dụng Fraction, tokenizer_name và normalization_version.

- [x] Viết test_nfkc_then_unescape_order: normalize_eval_text("HER2＋ &amp;  A\n B") == "her2+ & a b"; normalize_eval_text("&#xFF21;") == "ａ" trong thứ tự hiện hành, không thành "a".
- [x] Viết test_lcs_not_substring_and_order: LCS(("a","b","c"),("a","x","b","y","c")) == 3; reversed token order không được tính như bag overlap.
- [x] Viết repeated-token tests và threshold boundary 4/10 pass, 3/10 fail; wrong denominator test dự đoán dài không được dùng làm denominator.
- [x] Viết Chinese whitespace profile test: "肺癌治疗" thành một token; không implicit Jieba. Empty reference tokens phải reject ở matching contract.
- [x] Run: pytest tests/evaluation/test_vibiomir_text.py -q. Expected: FAIL thiếu normalization/LCS.
- [x] Implement exact DP memory hai rows, Fraction boundary; cache normalized/tokenized text theo profile+text hash trong một evaluation run.
- [x] Run tests. Expected: PASS, output đúng hand-calculated cases.
- [x] Commit: feat: implement ViBioMIR text overlap evaluator.


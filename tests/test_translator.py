from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.data.schemas import QueryRecord
from src.query.query_processor import AdaptiveQueryRouter, ProcessedQuery, QueryVariants
from src.query.translation_cache import TranslationCache
from src.query.translator import (
    ApiTranslator,
    LocalHfTranslator,
    MultilingualQueryBuilder,
    PassthroughTranslator,
    PrecomputedCacheTranslator,
    Translator,
    create_translator,
)


def test_translation_cache_lookup(tmp_path: Path):
    cache_file = tmp_path / "translations.jsonl"
    cache = TranslationCache(cache_file)
    cache.set("đau ngực", "en", "chest pain")

    # Reload from disk
    cache2 = TranslationCache(cache_file)
    assert cache2.get("đau ngực", "en") == "chest pain"
    assert cache2.get("đau ngực", "zh") is None


def test_query_builder_fallback_when_translation_missing(tmp_path: Path):
    cache_file = tmp_path / "translations.jsonl"
    cache = TranslationCache(cache_file)
    cache.set("sốt cao", "en", "high fever")
    # ZH is missing
    translator = PrecomputedCacheTranslator(cache)
    builder = MultilingualQueryBuilder(translator=translator, target_languages=["en", "zh"])
    variants = builder.build(text_vi="sốt cao", text_for_translation="sốt cao")

    assert variants.vi == "sốt cao"
    assert variants.en == "high fever"
    assert variants.zh is None  # Graceful fallback: zh stream will simply be skipped


def test_passthrough_translator():
    translator = PassthroughTranslator()
    assert isinstance(translator, Translator)
    assert translator.translate("đau đầu", "vi", "en") is None


def test_create_translator_factory(tmp_path: Path):
    cache_file = tmp_path / "trans.jsonl"
    cfg = {
        "query_understanding": {
            "translation": {
                "provider": "precomputed_cache",
                "cache_path": str(cache_file),
            }
        }
    }
    t = create_translator(cfg)
    assert isinstance(t, PrecomputedCacheTranslator)
    assert isinstance(t, Translator)


def test_translation_cache_thread_safety(tmp_path: Path):
    cache_file = tmp_path / "concurrent_cache.jsonl"
    cache = TranslationCache(cache_file)

    def write_worker(i: int):
        cache.set(f"thuốc {i}", "en", f"medicine {i}")
        return cache.get(f"thuốc {i}", "en")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_worker, range(50)))

    for i, res in enumerate(results):
        assert res == f"medicine {i}"

    # Verify reloading file from disk has all 50 entries
    cache_reloaded = TranslationCache(cache_file)
    for i in range(50):
        assert cache_reloaded.get(f"thuốc {i}", "en") == f"medicine {i}"


def test_translation_cache_corrupt_lines(tmp_path: Path):
    cache_file = tmp_path / "corrupt_cache.jsonl"
    # Write some valid lines and some corrupt lines
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write('{"key": "k1", "source_text": "đau tim", "target_lang": "en", "translated_text": "heart pain"}\n')
        f.write("CORRUPT NOT JSON LINE {[\n")
        f.write("\n")  # Empty line
        f.write('{"key": "k2", "source_text": "chóng mặt", "target_lang": "en", "translated_text": "dizziness"}\n')

    cache = TranslationCache(cache_file)
    assert cache.get("đau tim", "en") == "heart pain"
    assert cache.get("chóng mặt", "en") == "dizziness"
    assert cache.get("khó thở", "en") is None


def test_translation_cache_in_memory_mode():
    cache = TranslationCache(cache_path=None)
    cache.set("mất ngủ", "en", "insomnia")
    assert cache.get("mất ngủ", "en") == "insomnia"
    assert len(cache) == 1
    assert ("mất ngủ", "en") in cache


def test_precomputed_cache_translator_edge_cases(tmp_path: Path):
    cache_file = tmp_path / "edge_cache.jsonl"
    cache = TranslationCache(cache_file)
    cache.set("ho khan", "en", "dry cough")
    cache.set("rỗng", "en", "   ")  # whitespace only translation
    cache.set("ngắn", "en", "a")    # 1 character translation

    translator = PrecomputedCacheTranslator(cache)
    # Identical source and target
    assert translator.translate("ho khan", "vi", "vi") == "ho khan"
    assert translator.translate("", "vi", "en") is None
    # Valid translation
    assert translator.translate("ho khan", "vi", "en") == "dry cough"
    # Invalid translation (whitespace or too short)
    assert translator.translate("rỗng", "vi", "en") is None
    assert translator.translate("ngắn", "vi", "en") is None


def test_local_hf_translator_mocked():
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"translation_text": "diabetes type 2"}]

    translator = LocalHfTranslator(pipeline=mock_pipeline)
    assert isinstance(translator, Translator)
    result = translator.translate("tiểu đường type 2", "vi", "en")
    assert result == "diabetes type 2"
    mock_pipeline.assert_called_once()

    # Test error handling when pipeline throws exception
    failing_pipeline = MagicMock(side_effect=RuntimeError("GPU OOM"))
    failing_translator = LocalHfTranslator(pipeline=failing_pipeline)
    assert failing_translator.translate("tiểu đường", "vi", "en") is None


def test_api_translator_mocked():
    mock_client = MagicMock()
    mock_client.translate.return_value = "hypertension"

    translator = ApiTranslator(client=mock_client)
    assert isinstance(translator, Translator)
    result = translator.translate("tăng huyết áp", "vi", "en")
    assert result == "hypertension"

    # Test error handling when API fails
    failing_client = MagicMock()
    failing_client.translate.side_effect = ConnectionError("API timeout")
    failing_translator = ApiTranslator(client=failing_client)
    assert failing_translator.translate("tăng huyết áp", "vi", "en") is None


def test_create_translator_all_providers(tmp_path: Path):
    # passthrough
    t_pass = create_translator({"provider": "passthrough"})
    assert isinstance(t_pass, PassthroughTranslator)

    # disabled
    t_disabled = create_translator({"enabled": False, "provider": "precomputed_cache"})
    assert isinstance(t_disabled, PassthroughTranslator)

    # local_hf
    t_hf = create_translator({"provider": "local_hf", "local_hf_model": "test-model"})
    assert isinstance(t_hf, LocalHfTranslator)
    assert t_hf.model_name == "test-model"

    # api
    t_api = create_translator({"provider": "api", "api_model": "test-api"})
    assert isinstance(t_api, ApiTranslator)
    assert t_api.model == "test-api"

    # unsupported
    with pytest.raises(ValueError, match="Unsupported translation provider"):
        create_translator({"provider": "unknown_provider"})


def test_query_builder_with_processed_query(tmp_path: Path):
    cache_file = tmp_path / "trans.jsonl"
    cache = TranslationCache(cache_file)
    # Translate condensed text
    cache.set("đau tức ngực trái lan ra vai", "en", "left chest pain radiating to shoulder")
    cache.set("đau tức ngực trái lan ra vai", "zh", "左胸疼痛向肩部放射")

    translator = PrecomputedCacheTranslator(cache)
    builder = MultilingualQueryBuilder(translator=translator)

    router = AdaptiveQueryRouter(length_threshold=10, strip_clinical_boilerplate=True)
    query_text = (
        "Bệnh nhân nam 54 tuổi vào viện vì lý do đau tức ngực trái lan ra vai sau khi gắng sức."
    )
    routed = router.route(query_text, query_id="q100")
    assert routed.is_clinical_case is True
    # Verify condensed_vi was extracted
    assert routed.condensed_vi is not None

    # MultilingualQueryBuilder should use condensed_vi as default text_for_translation
    variants = builder.build(routed, text_for_translation="đau tức ngực trái lan ra vai")
    assert variants.vi == routed.normalized_text
    assert variants.vi_condensed == routed.condensed_vi
    assert variants.en == "left chest pain radiating to shoulder"
    assert variants.zh == "左胸疼痛向肩部放射"


def test_query_builder_fault_tolerance():
    # Translator that throws an unexpected exception on translation
    class CrashingTranslator:
        def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
            if target_lang == "en":
                raise RuntimeError("Catastrophic translation crash")
            return "正常中文翻译"

    builder = MultilingualQueryBuilder(translator=CrashingTranslator(), target_languages=["en", "zh"])
    # Should not crash the pipeline! en falls back to None, zh is preserved
    variants = builder.build("triệu chứng lâm sàng")
    assert variants.vi == "triệu chứng lâm sàng"
    assert variants.en is None
    assert variants.zh == "正常中文翻译"

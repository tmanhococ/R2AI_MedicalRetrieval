from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.query.query_processor import ProcessedQuery, QueryVariants
from src.query.translation_cache import TranslationCache

logger = logging.getLogger(__name__)


@runtime_checkable
class Translator(Protocol):
    """Protocol defining a translation service for query expansions."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """Translate text from source language to target language.

        Returns translated string, or None if translation is unavailable or fails.
        """
        ...


class PrecomputedCacheTranslator:
    """Translator backed by an offline precomputed JSONL translation cache."""

    def __init__(self, cache: TranslationCache | str | Path) -> None:
        if isinstance(cache, TranslationCache):
            self.cache = cache
        else:
            self.cache = TranslationCache(cache)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        if not text or not text.strip():
            return None
        clean_src = source_lang.strip().lower()
        clean_tgt = target_lang.strip().lower()
        if clean_src == clean_tgt:
            return text.strip()

        result = self.cache.get(text, target_lang=clean_tgt, source_lang=clean_src)
        if result is not None:
            clean_res = result.strip()
            # Invalidate empty or single-character translations
            if len(clean_res) >= 2:
                return clean_res
        return None


class PassthroughTranslator:
    """Null translator that skips translation (returns None).

    Used for Vietnamese-only retrieval baseline and ablation studies.
    """

    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        return None


class LocalHfTranslator:
    """Local HuggingFace translation pipeline (e.g. NLLB-200-distilled or MarianMT).

    Lazy loads pipeline on demand to avoid expensive startup cost.
    Completely fault-tolerant: falls back to None on any initialization or inference error.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        pipeline: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = pipeline

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import-not-found]

                self._pipeline = pipeline(
                    "translation",
                    model=self.model_name,
                    device=self.device,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load HuggingFace translation pipeline (%s): %s",
                    self.model_name,
                    exc,
                )
                return None
        return self._pipeline

    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        if not text or not text.strip():
            return None
        clean_src = source_lang.strip().lower()
        clean_tgt = target_lang.strip().lower()
        if clean_src == clean_tgt:
            return text.strip()

        pipe = self._get_pipeline()
        if pipe is None:
            return None

        try:
            nllb_lang_codes = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans"}
            src = nllb_lang_codes.get(clean_src, clean_src)
            tgt = nllb_lang_codes.get(clean_tgt, clean_tgt)

            output = pipe(text, src_lang=src, tgt_lang=tgt)
            if isinstance(output, list) and len(output) > 0:
                res = output[0].get("translation_text", "").strip()
                return res if len(res) >= 2 else None
            elif isinstance(output, str):
                res = output.strip()
                return res if len(res) >= 2 else None
            return None
        except Exception as exc:
            logger.warning("LocalHfTranslator translation failed for text '%s': %s", text, exc)
            return None


class ApiTranslator:
    """API-based translator (e.g. Gemini / OpenAI / Custom Translation API).

    Fault-tolerant: handles network errors and unexpected API responses gracefully.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.client = client

    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        if not text or not text.strip():
            return None
        clean_src = source_lang.strip().lower()
        clean_tgt = target_lang.strip().lower()
        if clean_src == clean_tgt:
            return text.strip()

        if self.client is None:
            logger.warning("ApiTranslator client is not configured; skipping translation.")
            return None

        try:
            if hasattr(self.client, "translate"):
                res = self.client.translate(text=text, source_lang=clean_src, target_lang=clean_tgt)
            elif callable(self.client):
                res = self.client(text, clean_src, clean_tgt)
            else:
                return None

            if res and isinstance(res, str):
                res = res.strip()
                return res if len(res) >= 2 else None
            return None
        except Exception as exc:
            logger.warning("ApiTranslator translation failed: %s", exc)
            return None


class MultilingualQueryBuilder:
    """Orchestrates query translation across multiple target languages.

    Produces QueryVariants with strict fault tolerance: if translation fails
    or is missing for any target language, that branch is set to None while
    preserving Vietnamese queries and other available translations.
    """

    def __init__(
        self,
        translator: Translator | None = None,
        target_languages: list[str] | None = None,
    ) -> None:
        self.translator: Translator = translator if translator is not None else PassthroughTranslator()
        self.target_languages: list[str] = (
            target_languages if target_languages is not None else ["en", "zh"]
        )

    def build(
        self,
        processed: ProcessedQuery | str | None = None,
        text_vi: str | None = None,
        text_for_translation: str | None = None,
        source_lang: str = "vi",
    ) -> QueryVariants:
        """Build QueryVariants from a processed query or raw Vietnamese text.

        If a ProcessedQuery is passed, condensed_vi is prioritized for translation
        when clinical boilerplate has been stripped.
        """
        if processed is not None:
            if isinstance(processed, ProcessedQuery):
                vi = processed.normalized_text
                vi_condensed = processed.condensed_vi
                if text_for_translation is None:
                    text_for_translation = processed.condensed_vi or processed.normalized_text
            elif isinstance(processed, str):
                vi = text_vi if text_vi is not None else processed
                vi_condensed = None
                if text_for_translation is None:
                    text_for_translation = processed
            else:
                raise TypeError(
                    f"Expected ProcessedQuery or str, got {type(processed).__name__}"
                )
        elif text_vi is not None:
            vi = text_vi
            vi_condensed = None
            if text_for_translation is None:
                text_for_translation = text_vi
        else:
            raise ValueError(
                "Either 'processed' or 'text_vi' must be provided to MultilingualQueryBuilder.build()"
            )

        translated: dict[str, str | None] = {}
        for lang in self.target_languages:
            if not text_for_translation or not text_for_translation.strip():
                translated[lang] = None
                continue
            try:
                res = self.translator.translate(
                    text=text_for_translation,
                    source_lang=source_lang,
                    target_lang=lang,
                )
                if res is not None:
                    res = res.strip()
                    if len(res) < 2:
                        res = None
                translated[lang] = res
            except Exception as exc:
                logger.warning(
                    "Translation to '%s' failed for query '%s': %s",
                    lang,
                    text_for_translation,
                    exc,
                )
                translated[lang] = None

        return QueryVariants(
            vi=vi,
            vi_condensed=vi_condensed,
            en=translated.get("en"),
            zh=translated.get("zh"),
        )


def create_translator(config: dict) -> Translator:
    """Factory function creating a Translator instance from configuration dict.

    Supports top-level config or nested query_understanding.translation config.
    """
    trans_cfg: dict = config
    if "query_understanding" in config and isinstance(config["query_understanding"], dict):
        trans_cfg = config["query_understanding"].get("translation", config["query_understanding"])
    elif "translation" in config and isinstance(config["translation"], dict):
        trans_cfg = config["translation"]

    if trans_cfg.get("enabled") is False:
        return PassthroughTranslator()

    provider = trans_cfg.get("provider", "passthrough")

    if provider == "precomputed_cache":
        cache_path = trans_cfg.get("cache_path", "artifacts/cache/translations.jsonl")
        return PrecomputedCacheTranslator(cache_path)
    elif provider in ("passthrough", "none", None):
        return PassthroughTranslator()
    elif provider == "local_hf":
        model_name = trans_cfg.get("local_hf_model", "facebook/nllb-200-distilled-600M")
        return LocalHfTranslator(model_name=model_name)
    elif provider == "api":
        model_name = trans_cfg.get("api_model", "gemini-1.5-flash")
        return ApiTranslator(model=model_name)
    else:
        raise ValueError(
            f"Unsupported translation provider: '{provider}'. "
            f"Must be one of 'precomputed_cache', 'passthrough', 'local_hf', 'api'."
        )

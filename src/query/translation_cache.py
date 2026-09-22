from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import threading
from src.query.query_processor import normalize_vi_query

logger = logging.getLogger(__name__)


class TranslationCache:
    """Thread-safe, append-only JSONL translation cache with O(1) in-memory lookup.

    Persists translations to a JSONL file and maintains an in-memory dictionary
    for fast retrieval during query expansion.
    """

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache_path: Path | None = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

        if self.cache_path and self.cache_path.is_file():
            self._load_cache()

    def _make_key(self, text: str, target_lang: str, source_lang: str = "vi") -> str:
        clean = normalize_vi_query(text).strip()
        text_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        return f"{text_hash}_{target_lang.strip().lower()}"

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        loaded_count = 0
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed line %d in translation cache %s: %s",
                        line_idx,
                        self.cache_path,
                        exc,
                    )
                    continue

                translated_text = data.get("translated_text")
                if translated_text is not None:
                    key = data.get("key")
                    if key:
                        self._cache[key] = translated_text
                    source_text = data.get("source_text")
                    target_lang = data.get("target_lang")
                    source_lang = data.get("source_lang", "vi")
                    if source_text and target_lang:
                        computed_key = self._make_key(source_text, target_lang, source_lang)
                        self._cache[computed_key] = translated_text
                    loaded_count += 1
        logger.debug("Loaded %d translation cache entries from %s", loaded_count, self.cache_path)

    def get(self, text: str, target_lang: str, source_lang: str = "vi") -> str | None:
        """Retrieve cached translation for query text.

        Returns None if translation is not found in cache.
        """
        if not text or not text.strip():
            return None
        key = self._make_key(text, target_lang, source_lang)
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                return val
            # Fallback for case variation (e.g. non-acronym lowercase)
            if not text.isupper():
                lower_key = self._make_key(text.lower(), target_lang, source_lang)
                val = self._cache.get(lower_key)
                if val is not None:
                    return val
            return None

    def set(
        self,
        text: str,
        target_lang: str,
        translated_text: str,
        source_lang: str = "vi",
    ) -> None:
        """Store translation in memory and append to disk cache."""
        if not text or not text.strip() or translated_text is None:
            return
        key = self._make_key(text, target_lang, source_lang)
        with self._lock:
            if self._cache.get(key) == translated_text:
                return
            self._cache[key] = translated_text
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "key": key,
                    "source_text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "translated_text": translated_text,
                }
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, item: tuple[str, str]) -> bool:
        text, target_lang = item
        return self.get(text, target_lang) is not None

import re
from typing import Callable

# Optional dependency imports
try:
    from pyvi import ViTokenizer  # type: ignore
    _PYVI_AVAILABLE = True
except ImportError:
    _PYVI_AVAILABLE = False

try:
    import jieba  # type: ignore
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

try:
    from nltk.stem import PorterStemmer  # type: ignore
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


_VI_SYLLABLE_PATTERN = re.compile(r"\w+", re.UNICODE)
_EN_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_ZH_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", re.UNICODE)


def _vi_syllable_tokenizer(text: str) -> list[str]:
    if not text:
        return []
    return _VI_SYLLABLE_PATTERN.findall(text.lower())


def _vi_pyvi_tokenizer(text: str) -> list[str]:
    if not text:
        return []
    if _PYVI_AVAILABLE:
        tokenized = ViTokenizer.tokenize(text).lower()
        return [t for t in tokenized.split() if t]
    return _vi_syllable_tokenizer(text)


def _en_whitespace_tokenizer(text: str) -> list[str]:
    if not text:
        return []
    return _EN_WORD_PATTERN.findall(text.lower())


_PORTER_STEMMER = PorterStemmer() if _NLTK_AVAILABLE else None


def _en_stem_tokenizer(text: str) -> list[str]:
    if not text:
        return []
    words = _EN_WORD_PATTERN.findall(text.lower())
    if _PORTER_STEMMER is not None:
        return [_PORTER_STEMMER.stem(w) for w in words]
    return words


def _zh_char_tokenizer(text: str) -> list[str]:
    if not text:
        return []
    return _ZH_CHAR_PATTERN.findall(text.lower())


def _zh_jieba_tokenizer(text: str) -> list[str]:
    if not text:
        return []
    if _JIEBA_AVAILABLE:
        return [w.strip() for w in jieba.lcut(text) if w.strip()]
    return _zh_char_tokenizer(text)


def _generic_tokenizer(text: str) -> list[str]:
    if not text:
        return []
    return _VI_SYLLABLE_PATTERN.findall(text.lower())


def get_tokenizer(lang: str, mode: str | None = None) -> Callable[[str], list[str]]:
    """Return a tokenizer function for the specified language and mode.

    Supported languages:
    - 'vi': mode 'syllable' (default) or 'pyvi' (fallback to syllable if unavailable)
    - 'en': mode 'whitespace' (default) or 'nltk_stem'
    - 'zh': mode 'jieba' (default, fallback to char if unavailable) or 'char'
    - other: generic unicode word tokenizer
    """
    lang = lang.lower() if lang else "generic"

    if lang == "vi":
        mode_val = mode.lower() if mode else "syllable"
        if mode_val == "pyvi":
            return _vi_pyvi_tokenizer
        return _vi_syllable_tokenizer

    elif lang == "en":
        mode_val = mode.lower() if mode else "whitespace"
        if mode_val == "nltk_stem":
            return _en_stem_tokenizer
        return _en_whitespace_tokenizer

    elif lang == "zh":
        mode_val = mode.lower() if mode else "jieba"
        if mode_val == "jieba":
            return _zh_jieba_tokenizer
        elif mode_val == "char":
            return _zh_char_tokenizer
        return _zh_char_tokenizer

    return _generic_tokenizer

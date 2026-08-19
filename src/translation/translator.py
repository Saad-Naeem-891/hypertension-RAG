"""Lightweight Arabic detection and translation with no Gemini quota usage.

Uses the free Google Translate web interface via deep-translator.
No API key required. Adds ~200-300 ms per translation call.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Arabic Unicode block: U+0600–U+06FF (covers all Arabic script letters).
# We also include extended Arabic supplement U+0750–U+077F and
# Arabic presentation forms U+FE70–U+FEFF so that common diacritics
# and ligatures are detected correctly.
_ARABIC_RANGES = (
    ("\u0600", "\u06FF"),
    ("\u0750", "\u077F"),
    ("\uFE70", "\uFEFF"),
)

# Minimum fraction of non-whitespace characters that must be Arabic for
# the text to be classified as an Arabic question.
_ARABIC_THRESHOLD = 0.20


def is_arabic(text: str) -> bool:
    """Return True when *text* contains significant Arabic content.

    The check operates purely on Unicode code points — no API call is
    made, so this function adds zero latency and consumes no quota.
    """
    stripped = text.replace(" ", "").replace("\n", "")
    if not stripped:
        return False
    arabic_count = sum(
        1
        for ch in stripped
        if any(lo <= ch <= hi for lo, hi in _ARABIC_RANGES)
    )
    return (arabic_count / len(stripped)) >= _ARABIC_THRESHOLD


def translate_ar_to_en(text: str) -> str:
    """Translate *text* from Arabic to English.

    Uses the free Google Translate web API via deep-translator.
    No API key needed; does not consume Gemini quota.

    Raises:
        RuntimeError: if deep-translator is not installed or the
            translation service is temporarily unreachable.
    """
    return _translate(text, source="ar", target="en")


def translate_en_to_ar(text: str) -> str:
    """Translate *text* from English to Arabic.

    Uses the free Google Translate web API via deep-translator.
    No API key needed; does not consume Gemini quota.

    Raises:
        RuntimeError: if deep-translator is not installed or the
            translation service is temporarily unreachable.
    """
    return _translate(text, source="en", target="ar")


def _translate(text: str, *, source: str, target: str) -> str:
    """Internal helper: translate *text* from *source* to *target* language."""
    if not text or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "deep-translator is not installed. "
            "Run: pip install deep-translator"
        ) from exc

    try:
        translated: str = GoogleTranslator(
            source=source, target=target
        ).translate(text)
        return translated or text
    except Exception as exc:
        logger.warning(
            "Translation %s→%s failed (%s). Returning original text.",
            source,
            target,
            exc,
        )
        # Graceful degradation: return original text so the pipeline
        # never crashes because of a translation failure.
        return text

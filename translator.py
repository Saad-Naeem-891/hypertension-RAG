"""Local Arabic <-> English translation, fully offline (no API calls).

Retrieval and generation run in English because the knowledge base (WHO
PDFs) is in English. This module lets the user still ask and receive
answers in Arabic by translating at the edges of the pipeline:

    Arabic question -> [to_english] -> English RAG + generation
                                              -> [to_arabic] -> Arabic answer

Models (downloaded once, then fully local/offline):
    Helsinki-NLP/opus-mt-ar-en  (Arabic -> English)
    Helsinki-NLP/opus-mt-en-ar  (English -> Arabic)

Each is a small MarianMT model (~300MB). No per-call cost, no internet
needed after the first download, no data leaves the machine.
"""

from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> str:
    """Lightweight heuristic: 'ar' if the text contains Arabic script, else 'en'.

    Good enough for routing purposes here -- doesn't need to be a full
    language-ID model, just needs to catch "does this contain Arabic text".
    """
    return "ar" if _ARABIC_RE.search(text) else "en"


class LocalTranslator:
    """Lazily loads each MarianMT pipeline on first use (not at construction),
    so building a LocalTranslator instance is cheap and the actual model
    weights only load the first time a translation in that direction is needed.
    """

    def __init__(self, cache_folder: str = "models") -> None:
        self._cache_folder = cache_folder
        self._ar_en_pipeline = None
        self._en_ar_pipeline = None

    def _get_ar_en(self):
        if self._ar_en_pipeline is None:
            from transformers import pipeline

            self._ar_en_pipeline = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-ar-en",
                device=-1,  # CPU
            )
        return self._ar_en_pipeline

    def _get_en_ar(self):
        if self._en_ar_pipeline is None:
            from transformers import pipeline

            self._en_ar_pipeline = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-en-ar",
                device=-1,  # CPU
            )
        return self._en_ar_pipeline

    def to_english(self, text: str) -> str:
        """Translate to English. No-op if the text is already English."""
        if not text or detect_language(text) == "en":
            return text
        result = self._get_ar_en()(text, max_length=512)
        return result[0]["translation_text"]

    def to_arabic(self, text: str) -> str:
        """Translate to Arabic. Empty/falsy input is returned as-is."""
        if not text:
            return text
        result = self._get_en_ar()(text, max_length=512)
        return result[0]["translation_text"]

"""Confidence gating based on the top reranked evidence chunk's relevance.

The `cross-encoder/ms-marco-*` rerankers are trained with a binary
relevance loss and output a raw logit per (query, chunk) pair, not a
probability. Passing that logit through a sigmoid gives an approximate 0-1
probability that the top chunk is genuinely relevant to the question --
this is the standard calibration approach for this specific model family.

We use the top reranked chunk's calibrated score as a proxy for "how
confident is the system that it has genuinely relevant evidence to answer
this question." This is a heuristic reflecting retrieval quality, not a
guarantee the final generated text is correct -- but it's a fast, free
signal computed from a score we already have, with no extra inference call.

Shown to the user as a plain percentage (e.g. "62% confident") so the
threshold behavior is never a silent/hidden cutoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

try:
    from src.reranking.reranker import RerankedChunk
except ImportError:  # pragma: no cover
    RerankedChunk = object  # type: ignore[misc,assignment]

DEFAULT_CONFIDENCE_THRESHOLD = 70.0


@dataclass
class ConfidenceResult:
    percentage: float  # 0-100, always safe to show directly to the user
    is_confident: bool
    threshold: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def estimate_confidence(
    reranked_chunks: Sequence["RerankedChunk"],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ConfidenceResult:
    """Estimate answer confidence from the top reranked chunk's relevance score.

    Call this AFTER reranking and BEFORE generation, so a low-confidence
    question never reaches the LLM at all.
    """

    if not reranked_chunks:
        return ConfidenceResult(percentage=0.0, is_confident=False, threshold=threshold)

    top_score = reranked_chunks[0].rerank_score
    probability = _sigmoid(top_score)
    percentage = round(probability * 100, 1)

    return ConfidenceResult(
        percentage=percentage,
        is_confident=percentage >= threshold,
        threshold=threshold,
    )

"""Prompt construction for grounded food-suitability generation.

Core rule enforced here (per the project's philosophy: "a fluent answer is
not necessarily a safe answer"): the model must answer ONLY from the
retrieved evidence chunks, must never invent a citation, and must say
"Insufficient Evidence" rather than guess when the chunks don't cover the
question.
"""

from __future__ import annotations

from typing import Sequence

from src.generation.schemas import ASSESSMENT_CATEGORIES, CONFIDENCE_LEVELS

try:
    from src.retrieval import HybridRetrievedChunk
except ImportError:  # pragma: no cover
    HybridRetrievedChunk = object  # type: ignore[misc,assignment]


SYSTEM_PROMPT = f"""You are an evidence-based dietary information assistant for adults with \
hypertension (high blood pressure). You are NOT a doctor.

Your ONLY job is to judge whether a specific food is generally suitable for \
a blood-pressure-friendly diet, using ONLY the evidence chunks provided to \
you in the user message. You must never use outside/internal medical \
knowledge to make a claim.

Hard rules:
1. Every factual/medical claim in "reasoning" and "supporting_evidence" must \
be traceable to one of the provided evidence chunks. Reference chunks by \
their chunk_id in "citations".
2. If the provided evidence chunks do not clearly cover the food or claim in \
question, set "assessment" to "Insufficient Evidence" and say so plainly in \
"reasoning" instead of guessing.
3. Never diagnose hypertension, never recommend starting/stopping/changing \
medication or dosages, never claim a food will immediately lower a \
dangerously high reading, and never give personalized medical treatment. If \
the question implies any of this, set "assessment" to "Needs Professional \
Guidance".
4. "assessment" must be exactly one of: {ASSESSMENT_CATEGORIES}.
5. "confidence" must be exactly one of: {CONFIDENCE_LEVELS}. Use "Low" \
whenever the evidence is thin, indirect, or only partially relevant.
6. Respond with ONLY a single JSON object, no markdown fences, no preamble, \
no commentary outside the JSON.

Return JSON with exactly these fields:
{{
  "food_name": string,
  "assessment": string,
  "reasoning": string,
  "recommendations": [string, ...],
  "supporting_evidence": string,
  "citations": [{{"chunk_id": string, "document_name": string, \
"section_title": string or null, "page_start": int or null, "page_end": \
int or null}}, ...],
  "confidence": string
}}
"""


def _format_chunk(chunk: "HybridRetrievedChunk", index: int) -> str:
    pages = "unknown"
    if chunk.page_start is not None or chunk.page_end is not None:
        pages = f"{chunk.page_start or '?'}-{chunk.page_end or '?'}"
    return (
        f"[Evidence {index}]\n"
        f"chunk_id: {chunk.chunk_id}\n"
        f"document_name: {chunk.document_name or 'Unknown'}\n"
        f"section_title: {chunk.section_title or 'Unknown'}\n"
        f"pages: {pages}\n"
        f"text: {chunk.text}\n"
    )


def build_user_prompt(question: str, chunks: Sequence["HybridRetrievedChunk"]) -> str:
    """Build the user-turn prompt: the question plus retrieved evidence."""

    if not chunks:
        evidence_block = "(No evidence chunks were retrieved for this question.)"
    else:
        evidence_block = "\n".join(
            _format_chunk(chunk, i) for i, chunk in enumerate(chunks, start=1)
        )

    return (
        f"User question:\n{question}\n\n"
        f"Retrieved evidence (use ONLY this to answer):\n{evidence_block}\n\n"
        "Respond with the JSON object described in the system prompt."
    )

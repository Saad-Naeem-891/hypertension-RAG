"""Structured output types for grounded food-suitability generation.

Mirrors the fields defined in the project overview's "Expected Application
Output" section exactly, so the UI and any future API layer can render a
consistent shape regardless of which chunks were retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Keep in sync with "Food Suitability Categories" in the project overview.
ASSESSMENT_CATEGORIES = [
    "Suitable",
    "Generally Suitable",
    "Consume in Moderation",
    "Limit",
    "Insufficient Evidence",
    "Needs Professional Guidance",
]

CONFIDENCE_LEVELS = ["High", "Medium", "Low"]


@dataclass
class Citation:
    document_name: str
    chunk_id: str
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class FoodAssessment:
    food_name: str
    assessment: str
    reasoning: str
    recommendations: list[str] = field(default_factory=list)
    supporting_evidence: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "Low"
    safety_message: str | None = None

    def __post_init__(self) -> None:
        if self.assessment not in ASSESSMENT_CATEGORIES:
            raise ValueError(
                f"assessment must be one of {ASSESSMENT_CATEGORIES}, "
                f"got {self.assessment!r}"
            )
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence must be one of {CONFIDENCE_LEVELS}, "
                f"got {self.confidence!r}"
            )

    def to_dict(self) -> dict:
        return {
            "food_name": self.food_name,
            "assessment": self.assessment,
            "reasoning": self.reasoning,
            "recommendations": self.recommendations,
            "supporting_evidence": self.supporting_evidence,
            "citations": [
                {
                    "document_name": c.document_name,
                    "chunk_id": c.chunk_id,
                    "section_title": c.section_title,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                }
                for c in self.citations
            ],
            "confidence": self.confidence,
            "safety_message": self.safety_message,
        }

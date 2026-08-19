"""Pre-retrieval safety triage for the hypertension food assistant.

This runs BEFORE retrieval/generation. Its only job is to decide whether a
user's question is safe to answer as a normal "is this food okay" question,
or whether it needs to be redirected to a safety message instead.

This is a fast, cheap, rule-based first line of defense (keyword matching in
English and Arabic). It is intentionally conservative: false positives
(flagging a borderline-safe question) are far less costly than false
negatives (answering a genuine emergency as if it were a food question).

This is a baseline. For production you would likely add a small LLM
classifier as a second pass on top of these keyword rules -- a stub for that
is included at the bottom (`llm_fallback_classify`) but not wired up, since
it needs your generation client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SafetyCategory(str, Enum):
    OK = "ok"
    EMERGENCY = "emergency"
    MEDICATION_CHANGE = "medication_change"
    CLINICAL_SPECIALIST = "clinical_specialist"
    COMPLEX_CONDITION = "complex_condition"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class SafetyCheckResult:
    category: SafetyCategory
    is_safe_to_answer_normally: bool
    matched_terms: list[str] = field(default_factory=list)
    safety_message: str | None = None


# NOTE: keep these lists lowercase; matching is done on a lowercased copy of
# the query. Arabic terms are included as-is (Arabic has no case folding).
_EMERGENCY_TERMS = [
    # English
    "chest pain", "can't breathe", "cannot breathe", "shortness of breath",
    "severe headache", "worst headache", "blurred vision", "vision loss",
    "confusion", "slurred speech", "numbness", "fainted", "fainting",
    "seizure",
    # Arabic
    "ألم في الصدر", "صعوبة في التنفس", "ضيق تنفس شديد", "صداع شديد",
    "تنميل", "تشنج", "فقدان الوعي", "دوخة شديدة", "تشوش الرؤية",
]

# These words are common in general WHO guideline questions, so they are only
# treated as emergency signals when the question is framed as a personal case.
_PERSONAL_EMERGENCY_TERMS = [
    "stroke", "heart attack", "hypertensive crisis", "180/", "220/",
    "سكتة", "جلطة",
]

_PERSONAL_CONTEXT_TERMS = [
    # English
    "i have", "i am", "i'm", "my ", "for me", "can i", "should i",
    "my mother", "my father", "my child", "my wife", "my husband",
    # Arabic
    "أنا", "انا", "عندي", "لدي", "عند أمي", "عند امي", "عند أبي",
    "عند ابي", "طفلي", "زوجي", "زوجتي",
]

_MEDICATION_TERMS = [
    # English
    "stop taking my medication", "stop my medication", "skip my dose",
    "double my dose", "increase my dose", "reduce my dose",
    "change my medication", "switch my medication", "come off my medication",
    # Arabic
    "أوقف الدواء", "اوقف الدواء", "أوقف علاج", "تغيير جرعة",
    "زود الجرعة", "قلل الجرعة", "أغير دوائي",
]

# Anything asking for dosage numbers, drug-specific guidance, lab-result
# interpretation, or a diagnosis. These always get redirected to a doctor,
# even if the retrieved evidence happens to contain a relevant number --
# a RAG chunk about a population-level dosage guideline is not the same as
# a safe personal recommendation, and this app must never imply otherwise.
_CLINICAL_SPECIALIST_TERMS = [
    # English
    "mg of", "milligram", "how many mg", "what dose", "what dosage",
    "safe dose", "correct dosage", "drug interaction", "interacts with",
    "lab result", "blood test result", "creatinine level", "gfr",
    "egfr", "diagnose me", "do i have hypertension", "what medication should i",
    "which drug should i", "prescribe", "prescription",
    # Arabic
    "كام مليجرام", "ملغ", "الجرعة المناسبة", "جرعة آمنة", "تداخل دوائي",
    "نتيجة تحليل", "نتيجة الدم", "تحليل الكرياتينين", "هل عندي ضغط",
    "شخصلي", "شخّصلي", "أي دواء يناسبني", "وصفة طبية",
]

_COMPLEX_CONDITION_TERMS = [
    # English
    "kidney disease", "kidney failure", "dialysis", "renal",
    "pregnant", "pregnancy", "diabetic ketoacidosis", "heart failure",
    "liver disease", "on blood thinners",
    # Arabic
    "مرض الكلى", "فشل كلوي", "غسيل كلى", "حامل", "الحمل",
    "فشل القلب", "مرض الكبد",
]

_SAFETY_MESSAGES: dict[SafetyCategory, str] = {
    SafetyCategory.EMERGENCY: (
        "This sounds like it could be a medical emergency, not a food "
        "question. Please contact emergency services or go to the nearest "
        "emergency room immediately. I can't assess symptoms or provide "
        "emergency guidance."
    ),
    SafetyCategory.MEDICATION_CHANGE: (
        "I can't advise on starting, stopping, or changing blood pressure "
        "medication or dosages -- that decision needs to be made with your "
        "doctor or pharmacist. I can still help with general food and diet "
        "questions related to hypertension."
    ),
    SafetyCategory.CLINICAL_SPECIALIST: (
        "This looks like a question for a doctor or pharmacist rather than "
        "a food question -- dosages, drug interactions, lab results, and "
        "diagnosis need to be assessed by a qualified professional who "
        "knows your medical history. I can't answer this safely, even if "
        "related background information exists in my sources, because a "
        "general guideline is not the same as a personal medical "
        "recommendation. Please consult a healthcare professional."
    ),
    SafetyCategory.COMPLEX_CONDITION: (
        "You've mentioned a condition (e.g. kidney disease, pregnancy, heart "
        "failure) that can substantially change what's safe to eat -- "
        "general hypertension dietary guidance may not apply and could even "
        "be inappropriate for your situation. Please get personalized "
        "guidance from your doctor or a registered dietitian."
    ),
}


def check_query(text: str) -> SafetyCheckResult:
    """Classify a user query before it reaches retrieval/generation."""

    lowered = text.lower()

    has_personal_context = any(term.lower() in lowered for term in _PERSONAL_CONTEXT_TERMS)
    emergency_hits = [term for term in _EMERGENCY_TERMS if term in lowered]
    if has_personal_context:
        emergency_hits.extend(
            term for term in _PERSONAL_EMERGENCY_TERMS if term in lowered
        )
    if emergency_hits:
        return SafetyCheckResult(
            category=SafetyCategory.EMERGENCY,
            is_safe_to_answer_normally=False,
            matched_terms=emergency_hits,
            safety_message=_SAFETY_MESSAGES[SafetyCategory.EMERGENCY],
        )

    # Checked BEFORE retrieval runs at all, and independent of what the
    # knowledge base contains -- dosage/clinical questions are redirected
    # unconditionally, per the project's safety rules.
    clinical_hits = [term for term in _CLINICAL_SPECIALIST_TERMS if term in lowered]
    if clinical_hits:
        return SafetyCheckResult(
            category=SafetyCategory.CLINICAL_SPECIALIST,
            is_safe_to_answer_normally=False,
            matched_terms=clinical_hits,
            safety_message=_SAFETY_MESSAGES[SafetyCategory.CLINICAL_SPECIALIST],
        )

    medication_hits = [term for term in _MEDICATION_TERMS if term in lowered]
    if medication_hits:
        return SafetyCheckResult(
            category=SafetyCategory.MEDICATION_CHANGE,
            is_safe_to_answer_normally=False,
            matched_terms=medication_hits,
            safety_message=_SAFETY_MESSAGES[SafetyCategory.MEDICATION_CHANGE],
        )

    complex_hits = [term for term in _COMPLEX_CONDITION_TERMS if term in lowered]
    if complex_hits and has_personal_context:
        return SafetyCheckResult(
            category=SafetyCategory.COMPLEX_CONDITION,
            is_safe_to_answer_normally=False,
            matched_terms=complex_hits,
            safety_message=_SAFETY_MESSAGES[SafetyCategory.COMPLEX_CONDITION],
        )

    return SafetyCheckResult(
        category=SafetyCategory.OK,
        is_safe_to_answer_normally=True,
    )


def llm_fallback_classify(text: str, generator) -> SafetyCheckResult:  # pragma: no cover
    """Optional second-pass classifier using the LLM generation client.

    Not wired into the default pipeline. Use this only if the rule-based
    `check_query` above turns out to be too permissive in your own testing
    (e.g. paraphrased emergencies it doesn't catch). `generator` should be an
    instance of `src.generation.generator.FoodAssessmentGenerator` or any
    object exposing a `.classify_safety(text) -> str` method you implement.
    """
    raise NotImplementedError(
        "Wire this up to your generation client if keyword rules prove "
        "insufficient during testing."
    )

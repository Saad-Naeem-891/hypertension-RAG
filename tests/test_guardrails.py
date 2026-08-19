from types import SimpleNamespace
import unittest

from src.guardrails import SafetyCategory, check_query, estimate_confidence


class SafetyCheckerTests(unittest.TestCase):
    def test_emergency_symptoms_are_blocked(self) -> None:
        result = check_query("I have chest pain and shortness of breath")

        self.assertFalse(result.is_safe_to_answer_normally)
        self.assertEqual(result.category, SafetyCategory.EMERGENCY)
        self.assertIn("emergency", result.safety_message.lower())

    def test_personal_complex_condition_is_blocked(self) -> None:
        result = check_query("I have kidney disease. How much potassium can I eat?")

        self.assertFalse(result.is_safe_to_answer_normally)
        self.assertEqual(result.category, SafetyCategory.COMPLEX_CONDITION)

    def test_general_guideline_question_about_complex_condition_is_allowed(self) -> None:
        result = check_query(
            "What does WHO recommend for adults with chronic kidney disease?"
        )

        self.assertTrue(result.is_safe_to_answer_normally)
        self.assertEqual(result.category, SafetyCategory.OK)

    def test_general_stroke_risk_question_is_not_treated_as_an_emergency(self) -> None:
        result = check_query("How does hypertension affect stroke risk?")

        self.assertTrue(result.is_safe_to_answer_normally)

    def test_personal_hypertensive_crisis_reading_is_blocked(self) -> None:
        result = check_query("My blood pressure is 180/120. What should I do?")

        self.assertFalse(result.is_safe_to_answer_normally)
        self.assertEqual(result.category, SafetyCategory.EMERGENCY)

    def test_personal_medication_change_is_blocked(self) -> None:
        result = check_query("Should I stop taking my medication?")

        self.assertFalse(result.is_safe_to_answer_normally)
        self.assertEqual(result.category, SafetyCategory.MEDICATION_CHANGE)


class ConfidenceGuardTests(unittest.TestCase):
    def test_empty_evidence_is_not_confident(self) -> None:
        result = estimate_confidence([])

        self.assertEqual(result.percentage, 0.0)
        self.assertFalse(result.is_confident)

    def test_raw_reranker_logit_is_calibrated_and_gated(self) -> None:
        result = estimate_confidence([SimpleNamespace(rerank_score=1.0)])

        self.assertEqual(result.percentage, 73.1)
        self.assertTrue(result.is_confident)

    def test_extreme_negative_score_is_numerically_stable(self) -> None:
        result = estimate_confidence([SimpleNamespace(rerank_score=-10_000.0)])

        self.assertEqual(result.percentage, 0.0)
        self.assertFalse(result.is_confident)

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            estimate_confidence([], threshold=101)


if __name__ == "__main__":
    unittest.main()

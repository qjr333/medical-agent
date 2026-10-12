"""Finite clarification and task reporting regressions; no live model or database."""

import unittest
from unittest.mock import Mock

from agent import extractor, reporting, state_manager
from medical_tools.safety import DANGER_PATTERNS, tool_assess_risk


def make_state():
    return state_manager.create_patient_state(
        "scheduled",
        DEMO_PATIENT={"patient_id": "task-v3-test", "patient_name": "模拟患者", "age": 70,
                      "conditions": ["高血压"]},
        DEMO_BP_HISTORY=[],
        TRIGGER_CONFIGS={"scheduled": {"followup_reason": "模拟随访", "bp_systolic": 135,
                                      "bp_diastolic": 85}},
    )


def ask(state, topic="medication", kind="medication_taken", flags=None):
    state["current_goal"] = {"medication": "check_medication", "bp": "collect_bp",
                             "current_symptoms": "check_symptoms"}.get(topic, "")
    state["question_context"] = {
        "goal": state["current_goal"], "topic": topic, "kind": kind,
        "question": "模拟单项问题", "asked_symptoms": ["头晕"] if kind == "symptoms" else [],
        "asked_red_flags": list(flags or []),
    }


def answer(state, text):
    model = Mock(return_value={})
    patch = extractor.extract_user_patch(text, state, call_json_model=model)
    state_manager.merge_user_patch(state, patch, text)
    return patch, model


class FiniteAnswerStateTests(unittest.TestCase):
    def test_new_fields_are_compatible_defaults_and_do_not_fill_current_facts(self):
        state = make_state()
        self.assertEqual(state["deferred_topics"], {})
        self.assertEqual(state["topic_answer_attempts"], {})
        self.assertEqual(state["task_errors"], [])
        self.assertEqual(state["latest_patient_message"], "")
        self.assertEqual(state["interruption_reason"], "")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertEqual(state["symptom_status"], "unknown")

    def test_refused_medication_remains_unknown_and_does_not_use_model(self):
        state = make_state()
        ask(state)
        patch, model = answer(state, "我不想回答")
        self.assertEqual(patch["answer_status"], "refused")
        model.assert_not_called()
        self.assertEqual(state["deferred_topics"], {"medication": "refused"})
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["medication_status"], "unknown")
        self.assertEqual(state["topic_status"]["medication"], "clarification_needed")

    def test_cannot_measure_defers_bp_without_fabricated_measurement(self):
        state = make_state()
        state.update(bp_systolic=None, bp_diastolic=None)
        state["coverage"]["bp"] = False
        state["topic_status"]["bp"] = "unanswered"
        ask(state, "bp", "bp")
        patch, model = answer(state, "我不会测血压")
        self.assertEqual(patch["answer_status"], "unable")
        model.assert_not_called()
        self.assertIsNone(state["bp_systolic"])
        self.assertFalse(state["coverage"]["bp"])
        self.assertEqual(state["deferred_topics"]["bp"], "unable")

    def test_symptom_refusal_also_defers_asked_red_flags_without_ruling_them_out(self):
        state = make_state()
        ask(state, "current_symptoms", "symptoms", DANGER_PATTERNS)
        answer(state, "这些问题我不愿意回答")
        self.assertEqual(state["deferred_topics"], {"current_symptoms": "refused", "red_flags": "refused"})
        self.assertEqual(state["symptom_status"], "unknown")
        self.assertEqual(state["red_flag_status"], "unknown")
        self.assertEqual(state["denied_symptoms"], [])
        self.assertEqual(state["denied_red_flags"], [])
        self.assertFalse(state["coverage"]["current_symptoms"])
        self.assertFalse(state["coverage"]["red_flags"])

    def test_three_unclear_answers_defer_the_pending_topic(self):
        state = make_state()
        ask(state)
        for count in range(1, 4):
            patch, _ = answer(state, "不确定")
            self.assertEqual(patch["answer_status"], "unclear")
            self.assertEqual(state["topic_answer_attempts"]["medication"], count)
            self.assertEqual("medication" in state["deferred_topics"], count == 3)
        self.assertEqual(state["deferred_topics"]["medication"], "unclear_limit")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["medication_status"], "unknown")

    def test_unrelated_answer_counts_as_unresolved_for_current_topic(self):
        state = make_state()
        ask(state)
        patch, _ = answer(state, "血压135/85")
        self.assertEqual(patch["answer_status"], "unclear")
        self.assertEqual(state["topic_answer_attempts"]["medication"], 1)
        self.assertTrue(state["coverage"]["bp"])
        self.assertFalse(state["coverage"]["medication"])

    def test_new_partial_medication_fact_reopens_deferred_topic_for_timing_clarification(self):
        state = make_state()
        ask(state)
        answer(state, "不想回答")
        patch, _ = answer(state, "还没吃")
        self.assertNotEqual(patch["answer_status"], "unclear")
        self.assertNotIn("medication", state["deferred_topics"])
        self.assertNotIn("medication", state["topic_answer_attempts"])
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertFalse(state["coverage"]["medication"])

    def test_clear_current_fact_clears_deferred_topic_and_counter(self):
        state = make_state()
        ask(state)
        for _ in range(3):
            answer(state, "不确定")
        answer(state, "今天已经吃了降压药")
        self.assertEqual(state["medication_status"], "taken")
        self.assertTrue(state["coverage"]["medication"])
        self.assertNotIn("medication", state["deferred_topics"])
        self.assertNotIn("medication", state["topic_answer_attempts"])

    def test_past_medication_fact_does_not_clear_current_unknown_or_refusal(self):
        state = make_state()
        ask(state)
        answer(state, "不想回答")
        answer(state, "昨天按时服药了")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["deferred_topics"]["medication"], "refused")

    def test_refusal_never_erases_a_new_danger_sign(self):
        state = make_state()
        ask(state, "current_symptoms", "symptoms", DANGER_PATTERNS)
        answer(state, "我不想回答，但是现在呼吸困难")
        self.assertIn("呼吸困难", state["red_flags"])
        self.assertEqual(state["red_flag_status"], "reported")
        self.assertTrue(state["coverage"]["red_flags"])
        self.assertEqual(tool_assess_risk(state)["risk_level"], "high")

    def test_merge_retains_latest_message_and_does_not_reset_handled_question(self):
        state = make_state()
        ask(state)
        question = "能不能多吃一片降压药？"
        state["last_medication_question"] = question
        answer(state, question)
        self.assertEqual(state["latest_patient_message"], question)
        self.assertEqual(state["latest_patient_question"], question)
        self.assertEqual(state["last_medication_question"], question)
        answer(state, "已经吃了")
        self.assertEqual(state["latest_patient_question"], "")


class TaskReportTests(unittest.TestCase):
    def test_report_extends_existing_fields_with_confirmation_and_error_summary(self):
        state = make_state()
        state.update(followup_status="escalated", risk_level="high", required_topics=["bp", "medication"],
                     deferred_topics={"medication": "refused"}, task_errors=[
                         {"tool": "RAG", "error_type": "TimeoutError", "recoverable": True,
                          "secret_exception_text": "must not be copied"},
                     ])
        report = reporting.generate_report(state)
        self.assertIn("patient", report)
        self.assertIn("blood_pressure", report)
        self.assertIn("knowledge", report)
        completion = report["task_completion"]
        self.assertEqual(completion["status"], "escalated")
        self.assertEqual(completion["unconfirmed_topics"], ["medication"])
        self.assertFalse(completion["all_required_information_confirmed"])
        self.assertEqual(completion["error_summary"], [{"tool": "RAG", "error_type": "TimeoutError", "recoverable": True}])

    def test_completed_report_is_reused_without_a_new_generated_timestamp(self):
        state = make_state()
        state["followup_status"] = "completed"
        first = reporting.generate_report(state)
        self.assertIs(reporting.generate_report(state), first)
        self.assertEqual(reporting.generate_report(state)["generated_at"], first["generated_at"])

    def test_interrupted_and_failed_tasks_do_not_generate_a_report(self):
        for status in ("interrupted", "failed"):
            with self.subTest(status=status):
                state = make_state()
                state["followup_status"] = status
                self.assertIsNone(reporting.generate_report(state))
                self.assertIsNone(state["report"])


if __name__ == "__main__":
    unittest.main()

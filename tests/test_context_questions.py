"""Context extraction, semantic question progress and safety regressions, offline."""

import unittest
from unittest.mock import Mock

from agent import extractor, planner, question_planner, state_manager
from medical_tools.safety import DANGER_PATTERNS, tool_assess_risk


def make_state():
    return state_manager.create_patient_state(
        "scheduled",
        DEMO_PATIENT={"patient_id": "context-test", "patient_name": "模拟患者", "age": 70,
                      "conditions": ["高血压"]},
        DEMO_BP_HISTORY=[],
        TRIGGER_CONFIGS={"scheduled": {"followup_reason": "测试随访",
                                      "bp_systolic": 150, "bp_diastolic": 90}},
    )


class ContextQuestionTests(unittest.TestCase):
    def reply(self, state, message):
        model = Mock(return_value={})
        extracted = extractor.extract_user_patch(message, state, call_json_model=model)
        state_manager.merge_user_patch(state, extracted, message)
        tool_assess_risk(state)
        return extracted, model

    def confirm_medication(self, state):
        question_planner.generate_question(state)
        self.reply(state, "已经吃了")
        self.assertTrue(state["coverage"]["medication"])

    def test_default_medication_question_is_single_and_context_is_recorded(self):
        state = make_state()
        question = question_planner.generate_question(state)
        self.assertIn("今天", question)
        self.assertNotIn("有没有忘记", question)
        self.assertEqual(state["current_goal"], "check_medication")
        self.assertEqual(state["last_agent_question"], question)
        self.assertEqual(state["question_context"]["question"], question)
        self.assertEqual(state["question_context"]["kind"], "medication_taken")
        self.assertEqual(state["question_attempts"]["medication.medication_taken"], 1)

    def test_not_yet_replies_clarify_timing_without_repeating_original_question(self):
        for answer in ("没有呢", "还没吃"):
            with self.subTest(answer=answer):
                state = make_state()
                original = question_planner.generate_question(state)
                extracted, model = self.reply(state, answer)
                self.assertEqual(extracted["medication_status"], "not_taken_yet")
                self.assertEqual(state["medication_status"], "not_taken_yet")
                self.assertFalse(state["coverage"]["medication"])
                self.assertEqual(state["topic_status"]["medication"], "clarification_needed")
                self.assertEqual(planner.select_next_action(state, []), "ASK_USER")
                next_question = question_planner.generate_question(state)
                self.assertNotEqual(next_question, original)
                self.assertIn("还没到", next_question)
                self.assertIn("超过", next_question)
                self.assertEqual(state["current_goal"], "clarify_medication_timing")
                self.assertEqual(state["question_context"]["kind"], "timing")
                model.assert_not_called()

    def test_three_rounds_progress_by_confirmed_topics_and_different_question_kinds(self):
        state = make_state()
        questions = [question_planner.generate_question(state)]
        kinds = [state["question_context"]["kind"]]
        self.reply(state, "还没吃")
        questions.append(question_planner.generate_question(state))
        kinds.append(state["question_context"]["kind"])
        self.reply(state, "还没到吃药时间")
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertEqual(state["medication_timing"], "not_due")
        self.assertTrue(state["coverage"]["medication"])
        self.assertEqual(state["topic_status"]["medication"], "answered")
        questions.append(question_planner.generate_question(state))
        kinds.append(state["question_context"]["kind"])
        self.assertEqual(state["current_goal"], "check_symptoms")
        self.reply(state, "都没有不舒服")
        self.assertEqual(state["symptom_status"], "denied")
        self.assertEqual(state["red_flag_status"], "denied")
        self.assertTrue(state["coverage"]["current_symptoms"])
        self.assertTrue(state["coverage"]["red_flags"])
        self.assertEqual(planner.select_next_action(state, ["CHECK_BP_HISTORY"]), "FINISH")
        self.assertEqual(len(set(questions)), 3)
        self.assertEqual(kinds, ["medication_taken", "timing", "symptoms"])

    def test_taken_yes_and_explicit_missed_move_to_symptoms(self):
        for answer, expected in (("已经吃了", "taken"), ("是的", "taken"),
                                 ("今天忘记吃药了", "missed")):
            with self.subTest(answer=answer):
                state = make_state()
                question_planner.generate_question(state)
                self.reply(state, answer)
                self.assertEqual(state["medication_status"], expected)
                self.assertTrue(state["coverage"]["medication"])
                question = question_planner.generate_question(state)
                self.assertNotIn("服用了降压药", question)
                self.assertEqual(state["current_goal"], "check_symptoms")

    def test_confirmed_overdue_does_not_invent_a_missed_dose(self):
        state = make_state()
        question_planner.generate_question(state)
        self.reply(state, "还没吃")
        question_planner.generate_question(state)
        self.reply(state, "已经超过规定的服药时间了")
        self.assertEqual(state["medication_timing"], "overdue")
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertTrue(state["coverage"]["medication"])
        question_planner.generate_question(state)
        self.assertEqual(state["current_goal"], "check_symptoms")

    def test_denied_adherence_issue_does_not_mean_not_taken(self):
        for answer in ("没有", "没有漏服"):
            with self.subTest(answer=answer):
                state = make_state()
                original = question_planner.record_question(
                    state, "您今天有没有漏服或者自行停药？", topic="medication",
                    kind="adherence", goal="check_medication",
                )
                self.reply(state, answer)
                self.assertEqual(state["medication_status"], "unknown")
                self.assertFalse(state["coverage"]["medication"])
                question = question_planner.generate_question(state)
                self.assertNotEqual(question, original)
                self.assertIn("实际服用", question)
                self.assertEqual(state["question_context"]["kind"], "taken_clarify")

    def test_uncertainty_uses_clarifications_and_never_marks_confirmed(self):
        state = make_state()
        original = question_planner.generate_question(state)
        self.reply(state, "不确定")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertEqual(state["topic_status"]["medication"], "clarification_needed")
        first_clarification = question_planner.generate_question(state)
        self.assertEqual(state["question_context"]["kind"], "taken_clarify")
        self.reply(state, "不确定")
        next_clarification = question_planner.generate_question(state)
        self.assertEqual(state["question_context"]["kind"], "medication_details")
        self.assertEqual(state["current_goal"], "clarify_medication_taken")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(len({original, first_clarification, next_clarification}), 3)

    def test_unclear_timing_requests_original_schedule_instead_of_repeating_binary_question(self):
        state = make_state()
        question_planner.generate_question(state)
        self.reply(state, "还没吃")
        timing_question = question_planner.generate_question(state)
        self.reply(state, "不确定")
        clarification = question_planner.generate_question(state)
        self.assertNotEqual(timing_question, clarification)
        self.assertIn("医生原来安排", clarification)
        self.assertEqual(state["question_context"]["kind"], "medication_details")
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertEqual(state["medication_timing"], "unknown")

    def test_same_short_denial_means_different_things_for_different_goals(self):
        medication = make_state()
        question_planner.generate_question(medication)
        self.reply(medication, "没有")
        self.assertEqual(medication["medication_status"], "not_taken_yet")
        self.assertEqual(medication["symptom_status"], "unknown")

        symptoms = make_state()
        self.confirm_medication(symptoms)
        question_planner.generate_question(symptoms)
        self.reply(symptoms, "没有")
        self.assertEqual(symptoms["medication_status"], "taken")
        self.assertEqual(symptoms["symptom_status"], "denied")
        self.assertEqual(symptoms["red_flag_status"], "denied")

    def test_partial_symptom_denial_only_covers_asked_danger_signs(self):
        state = make_state()
        self.confirm_medication(state)
        question_planner.record_question(
            state, "您有没有头晕、胸痛或者呼吸困难？", goal="check_symptoms",
            topic="current_symptoms", kind="symptoms", asked_symptoms=["头晕"],
            asked_red_flags=["胸痛", "呼吸困难"],
        )
        self.reply(state, "都没有")
        self.assertEqual(set(state["denied_red_flags"]), {"胸痛", "呼吸困难"})
        self.assertTrue(state["coverage"]["current_symptoms"])
        self.assertFalse(state["coverage"]["red_flags"])
        self.assertEqual(state["red_flag_status"], "unknown")
        self.assertEqual(planner.select_next_action(state, ["CHECK_BP_HISTORY"]), "ASK_USER")
        question = question_planner.generate_question(state)
        self.assertNotIn("胸痛", question)
        self.assertNotIn("呼吸困难", question)
        self.assertNotIn("头晕", question)
        self.assertEqual(set(state["question_context"]["asked_red_flags"]),
                         set(DANGER_PATTERNS) - {"胸痛", "呼吸困难"})
        self.assertEqual(state["question_context"]["topic"], "red_flags")
        self.reply(state, "没有")
        self.assertTrue(state["coverage"]["red_flags"])
        self.assertEqual(state["red_flag_status"], "denied")

    def test_full_symptom_question_covers_all_six_categories_including_consciousness(self):
        state = make_state()
        self.confirm_medication(state)
        question = question_planner.generate_question(state)
        self.assertIn("意识不清", question)
        self.assertEqual(set(state["question_context"]["asked_red_flags"]), set(DANGER_PATTERNS))

    def test_measurement_yes_answers_one_unambiguous_question(self):
        state = make_state()
        self.confirm_medication(state)
        question_planner.generate_question(state)
        self.reply(state, "都没有")
        state["risk_level"] = "attention"
        self.assertEqual(planner.select_next_action(state, ["CHECK_BP_HISTORY"]), "ASK_USER")
        question = question_planner.generate_question(state)
        self.assertNotIn("刚活动", question)
        self.assertEqual(state["current_goal"], "check_measurement_context")
        self.reply(state, "是的")
        self.assertTrue(state["coverage"]["measurement_context"])
        self.assertEqual(state["topic_status"]["measurement_context"], "answered")

    def test_high_risk_wins_over_clarifications_and_ordinary_tools(self):
        state = make_state()
        question_planner.generate_question(state)
        self.reply(state, "还没吃")
        self.reply(state, "没有胸痛，但是呼吸困难")
        self.assertIn("胸痛", state["denied_symptoms"])
        self.assertIn("呼吸困难", state["red_flags"])
        self.assertEqual(state["symptom_status"], "reported")
        self.assertEqual(state["red_flag_status"], "reported")
        self.assertEqual(state["risk_level"], "high")
        self.assertEqual(planner.select_next_action(state, []), "ESCALATE")

    def test_past_medication_memory_never_completes_current_topic(self):
        state = make_state()
        state["patient_memory"] = {
            "medication": {"last_7_days": {"issue_report_count": 2}},
            "last_followup": {"medication_status": "taken"},
        }
        question = question_planner.generate_question(state)
        self.assertIn("这次", question)
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.reply(state, "是的")
        self.assertEqual(state["medication_status"], "taken")
        self.assertTrue(state["coverage"]["medication"])


if __name__ == "__main__":
    unittest.main()

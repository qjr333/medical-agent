"""重复追问回归：不联网，不操作已有患者数据库。"""

import unittest
from unittest.mock import Mock

from agent import extractor, planner, question_planner, state_manager
from medical_tools.safety import tool_assess_risk


def make_state():
    return state_manager.create_patient_state(
        "scheduled",
        DEMO_PATIENT={
            "patient_id": "repeat-test",
            "patient_name": "模拟患者",
            "age": 70,
            "conditions": ["高血压"],
        },
        DEMO_BP_HISTORY=[],
        TRIGGER_CONFIGS={
            "scheduled": {
                "followup_reason": "模拟随访",
                "bp_systolic": 150,
                "bp_diastolic": 90,
            },
        },
    )


class RepeatFollowupTests(unittest.TestCase):
    def reply(self, state, text):
        model = Mock(return_value={})
        patch = extractor.extract_user_patch(text, state, call_json_model=model)
        state_manager.merge_user_patch(state, patch, text)
        tool_assess_risk(state)
        return patch, model

    def test_short_not_due_moves_to_symptoms(self):
        state = make_state()
        question_planner.generate_question(state)
        self.reply(state, "我今天没有服药")
        before = question_planner.generate_question(state)
        self.assertIn("还没到", before)
        patch, model = self.reply(state, "没到")
        self.assertEqual(patch["medication_timing"], "not_due")
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertEqual(state["medication_timing"], "not_due")
        self.assertTrue(state["coverage"]["medication"])
        self.assertEqual(state["topic_status"]["medication"], "answered")
        next_step = planner.select_next_decision(state, [])
        self.assertEqual(next_step["goal"], "check_symptoms")
        next_question = question_planner.generate_question(state, goal="check_symptoms")
        self.assertNotEqual(next_question, before)
        self.assertIn("不舒服", next_question)
        model.assert_not_called()

    def test_cannot_recall_schedule_does_not_repeat_time(self):
        for answer in ("记不清", "我不知道", "不太清楚", "想不起来"):
            with self.subTest(answer=answer):
                state = make_state()
                question_planner.generate_question(state)
                self.reply(state, "还没吃")
                question_planner.generate_question(state)
                self.reply(state, "不确定")
                time_question = question_planner.generate_question(state)
                self.assertIn("医生原来安排", time_question)
                patch, model = self.reply(state, answer)
                self.assertEqual(patch["answer_status"], "unable")
                self.assertEqual(state["deferred_topics"]["medication"], "unable")
                self.assertFalse(state["coverage"]["medication"])
                self.assertEqual(state["medication_status"], "not_taken_yet")
                self.assertEqual(state["medication_timing"], "unknown")
                choice = planner.select_next_decision(state, [])
                self.assertEqual(choice["goal"], "check_symptoms")
                next_question = question_planner.generate_question(state, goal=choice["goal"])
                self.assertNotEqual(next_question, time_question)
                self.assertIn("不舒服", next_question)
                model.assert_not_called()

    def test_general_uncertainty_remains_clarification(self):
        state = make_state()
        question_planner.generate_question(state)
        self.reply(state, "还没吃")
        question_planner.generate_question(state)
        patch, _ = self.reply(state, "不确定")
        self.assertEqual(patch["answer_status"], "unclear")
        self.assertNotIn("medication", state["deferred_topics"])

    def test_short_answer_outside_medication_context_does_not_update_medication(self):
        patch = extractor.extract_user_patch(
            "没到",
            {
                "current_goal": "check_symptoms",
                "question_context": {
                    "goal": "check_symptoms",
                    "topic": "current_symptoms",
                    "kind": "symptoms",
                    "question": "您现在有没有不舒服？",
                },
            },
            call_json_model=Mock(return_value={}),
        )
        self.assertEqual(patch["medication_status"], "unknown")
        self.assertEqual(patch["medication_observations"], [])

    def test_danger_sign_still_preempts_medication_followup(self):
        state = make_state()
        question_planner.generate_question(state)
        self.reply(state, "还没吃")
        question_planner.generate_question(state)
        self.reply(state, "不知道，但现在喘不上气")
        self.assertEqual(state["risk_level"], "high")
        self.assertEqual(planner.select_next_decision(state, [])["action"], "ESCALATE")


if __name__ == "__main__":
    unittest.main()

"""Deterministic Planner V3 decisions; no live model or production data writes."""

import unittest
from unittest.mock import patch

from agent import planner, question_planner, state_manager
from medical_tools import guidance
from medical_tools.safety import DANGER_PATTERNS, tool_assess_risk


def make_state():
    state = state_manager.create_patient_state(
        "scheduled",
        DEMO_PATIENT={"patient_id": "planner-v3", "patient_name": "模拟患者", "age": 72,
                      "conditions": ["高血压"]},
        DEMO_BP_HISTORY=[],
        TRIGGER_CONFIGS={"scheduled": {"followup_reason": "测试随访",
                                      "bp_systolic": 130, "bp_diastolic": 80}},
    )
    tool_assess_risk(state)
    return state


def confirm_symptoms(state):
    state.update(symptom_status="denied", red_flag_status="denied",
                 denied_red_flags=list(DANGER_PATTERNS), denied_symptoms=["头晕"])
    for topic in ("current_symptoms", "red_flags"):
        state["coverage"][topic] = True
        state["topic_status"][topic] = "answered"


def complete_information(state):
    confirm_symptoms(state)
    state.update(medication_status="taken")
    for topic in state["coverage"]:
        state["coverage"][topic] = True
        state["topic_status"][topic] = "answered"


def record_tool(state, action, query=None, status="ok"):
    state.setdefault("tool_execution", {})[action] = {
        "fingerprint": guidance.execution_fingerprint(state, action, query),
        "status": status,
    }


class PlannerV3Tests(unittest.TestCase):
    def decision(self, state, actions=()):
        result = planner.select_next_decision(state, list(actions))
        self.assertTrue({"action", "goal", "reason", "priority", "required_tools", "basis"} <= result.keys())
        self.assertTrue(result["reason"])
        self.assertNotIn("patient_memory", result["basis"])
        self.assertNotIn("trace", result["basis"])
        return result

    def test_high_risk_has_absolute_priority_over_questions_and_deferred_topics(self):
        state = make_state()
        state.update(red_flags=["胸痛"], latest_patient_question="能不能多吃一片药？")
        state["deferred_topics"] = {"current_symptoms": "refused", "red_flags": "refused"}
        tool_assess_risk(state)
        decision = self.decision(state)
        self.assertEqual(decision["action"], "ESCALATE")
        self.assertEqual(decision["priority"], "critical")
        self.assertEqual(decision["required_tools"], ["CREATE_ALERT"])
        self.assertEqual(planner.select_next_action(state, []), "ESCALATE")

    def test_abnormal_bp_prioritizes_unconfirmed_symptoms_before_medication(self):
        state = make_state()
        state.update(bp_systolic=162, bp_diastolic=96)
        tool_assess_risk(state)
        result = self.decision(state)
        self.assertEqual((result["action"], result["goal"], result["priority"]),
                         ("ASK_USER", "check_symptoms", "high"))
        self.assertEqual(state["medication_status"], "unknown")
        question = question_planner.generate_question(state, goal=result["goal"])
        self.assertIn("胸痛", question)
        self.assertEqual(state["question_context"]["goal"], "check_symptoms")
        self.assertEqual(state["current_goal"], "check_symptoms")

    def test_abnormal_bp_safety_screen_precedes_an_active_medication_question(self):
        state = make_state()
        state.update(bp_systolic=162, latest_patient_question="忘记吃药能补两片吗？")
        tool_assess_risk(state)
        self.assertEqual(self.decision(state)["goal"], "check_symptoms")
        confirm_symptoms(state)
        self.assertEqual(self.decision(state)["action"], "RETRIEVE_MEDICAL_GUIDANCE")

    def test_current_symptom_raises_safety_question_priority(self):
        state = make_state()
        state.update(symptoms=["头晕"], symptom_status="reported")
        state["coverage"]["current_symptoms"] = True
        tool_assess_risk(state)
        self.assertEqual(self.decision(state)["goal"], "check_symptoms")

    def test_recent_continuous_rise_from_memory_or_analysis_prioritizes_symptoms(self):
        for change in (
            {"bp_analysis": {"consecutive_rise": True}},
            {"bp_trend": "rising"},
            {"patient_memory": {"blood_pressure": {"last_7_days": {
                "consecutive_rise_last_3_readings": True}}}},
            {"bp_history": [{"systolic": 110, "diastolic": 65},
                            {"systolic": 120, "diastolic": 70}]},
        ):
            with self.subTest(change=change):
                state = make_state()
                state.update(change)
                decision = self.decision(state)
                self.assertEqual(decision["goal"], "check_symptoms")
                self.assertTrue(decision["basis"]["rising_history"])
                self.assertEqual(state["symptom_status"], "unknown")

    def test_historical_missed_report_only_changes_question_reason_and_wording(self):
        state = make_state()
        state["patient_memory"] = {"medication": {"last_7_days": {"issue_report_count": 2}},
                                   "last_followup": {"medication_status": "missed"}}
        decision = self.decision(state)
        self.assertEqual(decision["goal"], "check_medication")
        self.assertIn("历史", decision["reason"])
        question = question_planner.generate_question(state, decision["goal"])
        self.assertIn("这次", question)
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])

    def test_not_taken_yet_needs_timing_not_an_invented_missed_dose(self):
        state = make_state()
        state.update(medication_status="not_taken_yet", medication_timing="unknown")
        state["topic_status"]["medication"] = "clarification_needed"
        decision = self.decision(state)
        self.assertEqual(decision["goal"], "clarify_medication_timing")
        question = question_planner.generate_question(state, decision["goal"])
        self.assertIn("还没到", question)
        self.assertIn("超过", question)
        self.assertEqual(state["medication_status"], "not_taken_yet")

    def test_plain_normal_bp_keeps_medication_first_and_does_not_call_history_first(self):
        state = make_state()
        result = self.decision(state)
        self.assertEqual((result["action"], result["goal"]), ("ASK_USER", "check_medication"))
        self.assertEqual(result["required_tools"], [])

    def test_all_information_normal_state_analyzes_once_then_finishes(self):
        state = make_state()
        complete_information(state)
        self.assertEqual(self.decision(state)["action"], "CHECK_BP_HISTORY")
        record_tool(state, "CHECK_BP_HISTORY")
        decision = self.decision(state)
        self.assertEqual(decision["action"], "FINISH")
        self.assertEqual(decision["basis"]["pending_topics"], [])

    def test_attention_state_needs_evidence_and_visit_advice_before_report(self):
        state = make_state()
        complete_information(state)
        state.update(bp_systolic=162)
        tool_assess_risk(state)
        record_tool(state, "CHECK_BP_HISTORY")
        decision = self.decision(state)
        self.assertEqual(decision["action"], "RETRIEVE_MEDICAL_GUIDANCE")
        self.assertEqual(decision["goal"], "generate_visit_advice")
        query = guidance.build_guidance_query(state)
        record_tool(state, "RETRIEVE_MEDICAL_GUIDANCE", query, status="empty")
        self.assertEqual(self.decision(state)["action"], "GENERATE_VISIT_ADVICE")
        record_tool(state, "GENERATE_VISIT_ADVICE")
        self.assertEqual(self.decision(state)["action"], "FINISH")

    def test_empty_rag_is_completed_attempt_then_safe_medication_tool(self):
        state = make_state()
        question = "降压药忘记吃了可以多吃一片吗？"
        state["latest_patient_question"] = question
        state["retrieved_guidance"] = []
        record_tool(state, "RETRIEVE_MEDICAL_GUIDANCE", question, status="empty")
        self.assertEqual(self.decision(state)["action"], "MEDICATION_GUIDANCE")

    def test_non_medication_question_can_be_answered_before_missing_bp(self):
        state = make_state()
        state.update(bp_systolic=None, bp_diastolic=None,
                     latest_patient_question="家庭血压应该怎么测？")
        state["coverage"]["bp"] = False
        state["topic_status"]["bp"] = "unanswered"
        question = state["latest_patient_question"]
        self.assertEqual(self.decision(state)["action"], "RETRIEVE_MEDICAL_GUIDANCE")
        record_tool(state, "RETRIEVE_MEDICAL_GUIDANCE", question)
        self.assertEqual(self.decision(state)["action"], "GENERATE_VISIT_ADVICE")
        record_tool(state, "GENERATE_VISIT_ADVICE", question)
        state["handled_patient_question"] = {
            "question": question,
            "fingerprint": guidance.execution_fingerprint(state, "ANSWER_PATIENT_QUESTION", question),
        }
        self.assertEqual(self.decision(state)["goal"], "collect_bp")

    def test_handled_question_is_ignored_but_changed_state_makes_it_new(self):
        state = make_state()
        confirm_symptoms(state)
        question = "降压药漏服应该怎么办？"
        state["latest_patient_question"] = question
        state["handled_patient_question"] = {
            "question": question,
            "fingerprint": guidance.execution_fingerprint(state, "ANSWER_PATIENT_QUESTION", question),
        }
        self.assertEqual(self.decision(state)["goal"], "check_medication")
        state["medication_status"] = "missed"
        self.assertEqual(self.decision(state)["action"], "RETRIEVE_MEDICAL_GUIDANCE")

    def test_stale_tool_entry_cannot_be_hidden_by_old_query_marker(self):
        state = make_state()
        question = "降压药漏服应该怎么办？"
        state.update(latest_patient_question=question, last_guidance_query=question)
        record_tool(state, "RETRIEVE_MEDICAL_GUIDANCE", question)
        state["medication_detail"] = "新的服药情况"
        self.assertEqual(self.decision(state, ["RETRIEVE_MEDICAL_GUIDANCE"])["action"],
                         "RETRIEVE_MEDICAL_GUIDANCE")

    def test_knowledge_revision_invalidates_a_previous_same_question_attempt(self):
        state = make_state()
        question = "降压药漏服应该怎么办？"
        state["latest_patient_question"] = question
        with patch.object(guidance, "knowledge_revision", return_value="revision-1"):
            record_tool(state, "RETRIEVE_MEDICAL_GUIDANCE", question)
        with patch.object(guidance, "knowledge_revision", return_value="revision-2"):
            self.assertEqual(self.decision(state)["action"], "RETRIEVE_MEDICAL_GUIDANCE")

    def test_refused_required_information_interrupts_instead_of_finishing(self):
        state = make_state()
        complete_information(state)
        state["coverage"]["medication"] = False
        state["topic_status"]["medication"] = "clarification_needed"
        state["deferred_topics"] = {"medication": "refused"}
        self.assertEqual(self.decision(state)["action"], "INTERRUPT")
        self.assertFalse(state["coverage"]["medication"])

    def test_abnormal_bp_with_deferred_safety_information_interrupts_before_tools(self):
        state = make_state()
        state.update(bp_systolic=162, latest_patient_question="降压药怎么吃？")
        state["deferred_topics"] = {"current_symptoms": "refused", "red_flags": "refused"}
        tool_assess_risk(state)
        result = self.decision(state)
        self.assertEqual(result["action"], "INTERRUPT")
        self.assertEqual(result["goal"], "check_symptoms")

    def test_deferred_medication_is_not_asked_again_when_symptoms_can_be_collected(self):
        state = make_state()
        state["deferred_topics"] = {"medication": "unclear_limit"}
        result = self.decision(state)
        self.assertEqual(result["goal"], "check_symptoms")
        question_planner.generate_question(state, result["goal"])
        self.assertEqual(state["question_context"]["topic"], "current_symptoms")

    def test_explicit_measurement_goal_does_not_fall_back_to_medication_order(self):
        state = make_state()
        question = question_planner.generate_question(state, "check_measurement_context")
        self.assertIn("安静坐着休息", question)
        self.assertEqual(state["current_goal"], "check_measurement_context")

    def test_explicit_symptom_goal_only_asks_unconfirmed_red_flags(self):
        state = make_state()
        state.update(symptom_status="denied", denied_red_flags=["胸痛", "呼吸困难"],
                     denied_symptoms=["头晕"])
        state["coverage"]["current_symptoms"] = True
        state["topic_status"]["current_symptoms"] = "answered"
        question = question_planner.generate_question(state, "check_symptoms")
        self.assertNotIn("胸痛", question)
        self.assertNotIn("呼吸困难", question)
        self.assertEqual(state["question_context"]["topic"], "red_flags")

    def test_resolved_explicit_goal_does_not_repeat_the_answered_topic(self):
        state = make_state()
        confirm_symptoms(state)
        question = question_planner.generate_question(state, "check_symptoms")
        self.assertIn("服用了降压药", question)
        self.assertEqual(state["current_goal"], "check_medication")

    def test_completed_state_decision_does_not_schedule_tools_again(self):
        state = make_state()
        state.update(followup_status="completed", report={"existing": True})
        result = self.decision(state)
        self.assertEqual(result["action"], "FINISH")
        self.assertEqual(result["goal"], "completed")
        self.assertEqual(state["report"], {"existing": True})


if __name__ == "__main__":
    unittest.main()

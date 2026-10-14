"""V5 minimum completion gates, using offline structured semantic evidence."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from agent.completion import completion_requirements, evaluate_completion
from agent.question_planner import generate_question
from backend.utils import today_local_date


def stable_state(trigger="scheduled"):
    answer = "今天量的是138/82，药按原来的安排吃了，现在没有不舒服。"
    return {
        "agent_version": "V5.0", "trigger_type": trigger,
        "risk_level": "normal", "red_flags": [],
        "bp_systolic": 138, "bp_diastolic": 82,
        "medication_status": "taken", "medication_timing": "unknown",
        "medication_date": today_local_date(),
        "symptom_status": "denied", "red_flag_status": "screened",
        "denied_red_flags": [], "measurement_context": "",
        "coverage": {"bp": True, "medication": True, "current_symptoms": True,
                     "red_flags": True, "measurement_context": False},
        "topic_status": {}, "deferred_topics": {}, "trace": [],
        "latest_patient_message": answer, "latest_patient_question": "",
        "_semantic": {"screening": {
            "adequate": True, "evidence": "现在没有不舒服",
            "question": "今天有没有新出现的明显不舒服，尤其胸痛或喘不过气？",
            "scope": ["胸痛", "呼吸困难"],
        }},
    }


def proposal(outcome="completed", **extra):
    return {"outcome": outcome, "reason": "本次目标已核实，无需无关追问", "pending_items": [],
            "next_goal": "", **extra}


class LLMCompletionAutonomyTests(unittest.TestCase):
    def evaluate(self, state, decision=None, context=None):
        model = Mock(return_value=proposal() if decision is None else decision)
        result = evaluate_completion(state, context or {}, model_call=model)
        return result, model

    def test_stable_contextual_screen_can_finish_without_enumerating_all_red_flags(self):
        state = stable_state()
        result, model = self.evaluate(state)
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(state["denied_red_flags"], [])
        self.assertNotIn("measurement_context", result["pending_items"])
        prompt, payload = model.call_args.args[1:]
        self.assertIn("不是固定问卷", prompt)
        self.assertEqual(payload["verified_semantics"], state["_semantic"])

    def test_legacy_required_fields_do_not_create_unrelated_v5_questionnaire(self):
        state = stable_state()
        state["required_topics"] = ["bp", "medication", "red_flags", "measurement_context"]
        self.assertNotIn("measurement_context", completion_requirements(state))
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "completed")

    def test_generic_wellbeing_without_screening_context_cannot_clear_risk(self):
        for incomplete in ({}, {"adequate": True, "evidence": "挺好", "question": "", "scope": []}):
            with self.subTest(screening=incomplete):
                state = stable_state()
                state["latest_patient_message"] = "挺好"
                state["_semantic"]["screening"] = incomplete
                result, _ = self.evaluate(state)
                self.assertEqual(result["outcome"], "continue")
                self.assertIn("red_flags", result["pending_items"])

    def test_screening_requires_the_verified_original_patient_quote(self):
        state = stable_state()
        state["_semantic"]["screening"]["evidence"] = "我没有任何医学危险症状"
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("red_flags", result["pending_items"])

    def test_inadequate_screening_is_not_completed_by_coverage_alone(self):
        state = stable_state()
        state["_semantic"]["screening"]["adequate"] = False
        result, _ = self.evaluate(state)
        self.assertIn("red_flags", result["pending_items"])

    def test_actual_high_bp_still_needs_relevant_measurement_context(self):
        state = stable_state("high_bp")
        result, _ = self.evaluate(state)
        self.assertIn("measurement_context", result["pending_items"])
        state["coverage"]["measurement_context"] = True
        state["measurement_context"] = "今天坐着休息五分钟后测量"
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "completed")

    def test_verified_colloquial_monitoring_commitment_does_not_need_fixed_phrase(self):
        state = stable_state("missed_measurement")
        state.update(bp_systolic=None, bp_diastolic=None)
        state["coverage"]["bp"] = False
        quote = "明儿让闺女给我量一下，再记到本子里"
        state["trace"] = [{"event": "USER_MESSAGE", "detail": {"message": quote}}]
        state["_semantic"]["monitoring_plan"] = {"confirmed": True, "evidence": quote}
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(result["objective_evidence"]["monitoring_plan"], quote)
        self.assertIsNone(state["bp_systolic"])

    def test_a_fresh_real_reading_resolves_missing_measurement_without_future_promise(self):
        state = stable_state("missed_measurement")
        self.assertNotIn("monitoring_plan", completion_requirements(state))
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "completed")

    def test_completion_model_cannot_create_a_monitoring_plan_from_an_original_quote(self):
        for quote in ("以前每天都量", "如果明天有空就看看", "明天不想量", "医生让我明天量"):
            with self.subTest(quote=quote):
                state = stable_state("missed_measurement")
                state.update(bp_systolic=None, bp_diastolic=None)
                state["trace"] = [{"event": "USER_MESSAGE", "detail": {"message": quote}}]
                result, _ = self.evaluate(state, proposal(objective_evidence={"monitoring_plan": quote}))
                self.assertEqual(result["outcome"], "continue")
                self.assertIn("monitoring_plan", result["pending_items"])
                self.assertNotIn("objective_evidence", result)

    def test_unverified_or_invented_semantic_plan_is_rejected(self):
        for value in ({"confirmed": False, "evidence": "我会继续测量"},
                      {"confirmed": True, "evidence": "我会继续测量"}):
            with self.subTest(value=value):
                state = stable_state("missed_measurement")
                state.update(bp_systolic=None, bp_diastolic=None)
                state["_semantic"]["monitoring_plan"] = value
                result, _ = self.evaluate(state)
                self.assertIn("monitoring_plan", result["pending_items"])

    def test_llm_can_revisit_a_completed_topic_when_new_information_warrants_it(self):
        state = stable_state()
        decision = proposal("continue", pending_items=["new_medication_question"],
                            next_goal="核实刚新增的用药疑问")
        result, _ = self.evaluate(state, decision)
        self.assertEqual(result["outcome"], "continue")
        self.assertEqual(result["next_goal"], "核实刚新增的用药疑问")
        self.assertIn("new_medication_question", result["pending_items"])

    def test_new_verified_end_intent_can_close_as_partial_with_missing_facts(self):
        state = stable_state()
        state["medication_status"] = "unknown"
        quote = "今天先歇会儿，咱们别聊了"
        state["latest_patient_message"] = quote
        state["_semantic"]["patient_requested_end"] = {"confirmed": True, "evidence": quote}
        result, model = self.evaluate(state, proposal("partial"))
        self.assertEqual(result["outcome"], "partial")
        self.assertIn("medication", result["pending_items"])
        self.assertTrue(model.call_args.args[2]["patient_requested_end"])

    def test_old_end_flags_or_historical_quotes_do_not_override_current_intent(self):
        state = stable_state()
        state["medication_status"] = "unknown"
        state["v41_patient_requested_end"] = True
        state["_semantic"]["patient_requested_end"] = {"confirmed": True, "evidence": "昨天结束吧"}
        state["trace"] = [{"event": "USER_MESSAGE", "detail": {"message": "昨天结束吧"}}]
        result, model = self.evaluate(state, proposal("partial"))
        self.assertEqual(result["outcome"], "continue")
        self.assertFalse(model.call_args.args[2]["patient_requested_end"])

    def test_safety_and_unresolved_tool_errors_keep_priority_over_model_finish(self):
        state = stable_state()
        state["red_flags"] = ["呼吸困难"]
        result, model = self.evaluate(state)
        self.assertEqual(result["outcome"], "escalated")
        model.assert_not_called()
        context = {"tool_results_this_turn": [{"tool": "MEDICATION_GUIDANCE", "status": "error"}]}
        result, _ = self.evaluate(stable_state(), context=context)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("tool:MEDICATION_GUIDANCE", result["pending_items"])

    def test_completion_does_not_mutate_verified_patient_facts(self):
        state = stable_state()
        before = deepcopy(state)
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(state, before)

    def test_v5_question_context_records_the_model_choice_even_on_an_answered_topic(self):
        state = stable_state()
        question = "您刚提到换了药盒，能再核实一下今天实际服下的是哪一种吗？"
        value = generate_question(state, model_proposal={
            "question": question, "topic": "medication", "kind": "supplement",
            "goal": "核实患者刚提出的新用药信息",
        })
        self.assertEqual(value, question)
        self.assertEqual(state["question_context"]["question"], question)
        self.assertTrue(state["coverage"]["medication"])

    def test_v5_does_not_fallback_to_a_fixed_question_list_without_a_model_choice(self):
        state = stable_state()
        state["coverage"]["bp"] = False
        with self.assertRaisesRegex(ValueError, "v5_requires_model_question"):
            generate_question(state)


if __name__ == "__main__":
    unittest.main()

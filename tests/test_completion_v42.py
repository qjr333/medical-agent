"""Offline completion-evaluator tests: no patient database or model network."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from agent.completion import completion_requirements, evaluate_completion
from medical_tools.safety import DANGER_PATTERNS


def make_state(kind="scheduled"):
    return {
        "trigger_type": kind, "risk_level": "normal", "red_flags": [],
        "bp_systolic": 145, "bp_diastolic": 85,
        "medication_status": "taken", "medication_timing": "unknown",
        "symptom_status": "denied", "red_flag_status": "denied",
        "denied_red_flags": list(DANGER_PATTERNS),
        "measurement_context": "安静休息后测量",
        "coverage": {"bp": True, "medication": True, "current_symptoms": True,
                     "red_flags": True, "measurement_context": True},
        "topic_status": {"bp": "answered", "medication": "answered", "current_symptoms": "answered",
                         "red_flags": "answered", "measurement_context": "answered"},
        "latest_patient_message": "今天按原医嘱服药，目前没有不舒服。",
        "latest_patient_question": "", "deferred_topics": {}, "trace": [],
    }


def proposal(outcome="completed", pending=None, **extra):
    return {"outcome": outcome, "reason": "依据当前事实评估本次目标", "pending_items": pending or [],
            "next_goal": "" if outcome != "continue" else "confirm_remaining_information", **extra}


class CompletionV42Tests(unittest.TestCase):
    def evaluate(self, state, decision=None, context=None):
        model = Mock(return_value=proposal() if decision is None else decision)
        result = evaluate_completion(state, context or {}, model_call=model)
        return result, model

    def test_requirements_follow_each_trigger_objective(self):
        self.assertEqual(completion_requirements(make_state()),
                         ["bp", "medication", "current_symptoms", "red_flags"])
        self.assertEqual(completion_requirements(make_state("high_bp")),
                         ["bp", "medication", "current_symptoms", "red_flags", "measurement_context"])
        missing = make_state("missed_measurement")
        missing["required_topics"] = ["bp", "medication", "current_symptoms", "red_flags"]
        self.assertEqual(completion_requirements(missing),
                         ["medication", "current_symptoms", "red_flags", "monitoring_plan"])

    def test_high_risk_escalates_without_waiting_for_completion_model(self):
        state = make_state()
        state["risk_level"] = "high"
        result, model = self.evaluate(state)
        self.assertEqual(result["outcome"], "escalated")
        self.assertEqual(result["model_status"], "skipped")
        model.assert_not_called()

    def test_red_flags_override_a_model_that_would_complete(self):
        state = make_state()
        state["red_flags"] = ["胸痛"]
        result, model = self.evaluate(state)
        self.assertEqual(result["outcome"], "escalated")
        model.assert_not_called()

    def test_scheduled_task_completes_with_confirmed_current_information(self):
        result, model = self.evaluate(make_state())
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(result["model_status"], "success")
        self.assertEqual(result["pending_items"], [])
        purpose, system, payload = model.call_args.args
        self.assertEqual(purpose, "COMPLETE")
        self.assertIn("外部数据", system)
        self.assertIn("当前", system)
        self.assertEqual(payload["trigger_type"], "scheduled")

    def test_completion_cannot_override_missing_required_coverage(self):
        for topic in ("bp", "medication", "current_symptoms", "red_flags"):
            with self.subTest(topic=topic):
                state = make_state()
                state["coverage"][topic] = False
                result, _ = self.evaluate(state)
                self.assertEqual(result["outcome"], "continue")
                self.assertIn(topic, result["pending_items"])

    def test_coverage_alone_cannot_replace_verified_current_facts(self):
        changes = [("bp_systolic", None, "bp"), ("medication_status", "unknown", "medication"),
                   ("symptom_status", "unknown", "current_symptoms"),
                   ("red_flag_status", "unknown", "red_flags")]
        for key, value, topic in changes:
            with self.subTest(key=key):
                state = make_state()
                state[key] = value
                result, _ = self.evaluate(state)
                self.assertEqual(result["outcome"], "continue")
                self.assertIn(topic, result["pending_items"])

    def test_unknown_not_taken_timing_still_needs_confirmation(self):
        state = make_state()
        state["medication_status"] = "not_taken_yet"
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("medication", result["pending_items"])

    def test_positive_red_flag_screen_without_details_is_not_risk_clearance(self):
        state = make_state()
        state["red_flag_status"] = "reported"
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("red_flags", result["pending_items"])

    def test_missing_or_partial_denial_records_cannot_be_replaced_by_coverage(self):
        for denied in (None, [], ["胸痛", "呼吸困难"], list(DANGER_PATTERNS)[:-1]):
            with self.subTest(denied=denied):
                state = make_state()
                state["denied_red_flags"] = denied
                result, _ = self.evaluate(state)
                self.assertEqual(result["outcome"], "continue")
                self.assertIn("red_flags", result["pending_items"])

    def test_absent_denial_field_in_legacy_state_cannot_complete_safety_screen(self):
        state = make_state()
        del state["denied_red_flags"]
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("red_flags", result["pending_items"])

    def test_high_bp_requires_measurement_context_and_valid_current_reading(self):
        state = make_state("high_bp")
        state["measurement_context"] = ""
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("measurement_context", result["pending_items"])
        state["measurement_context"] = "刚走动后测量"
        state["bp_systolic"] = 40
        result, _ = self.evaluate(state)
        self.assertIn("bp", result["pending_items"])

    def test_missed_measurement_can_finish_without_reading_with_grounded_recovery_plan(self):
        state = make_state("missed_measurement")
        state.update(bp_systolic=None, bp_diastolic=None,
                     latest_patient_message="今天按时服药，身体没有不舒服，我明天让女儿帮我测血压并记录。")
        state["coverage"]["bp"] = False
        decision = proposal(objective_evidence={"monitoring_plan": "我明天让女儿帮我测血压并记录"})
        result, _ = self.evaluate(state, decision)
        self.assertEqual(result["outcome"], "completed")
        self.assertNotIn("bp", result["pending_items"])
        self.assertEqual(result["objective_evidence"]["monitoring_plan"], "我明天让女儿帮我测血压并记录")
        self.assertIsNone(state["bp_systolic"])

    def test_monitoring_plan_cannot_be_fabricated_by_model_or_coverage(self):
        state = make_state("missed_measurement")
        state["coverage"]["monitoring_plan"] = True
        result, _ = self.evaluate(state, proposal(objective_evidence={"monitoring_plan": "我明天会测血压"}))
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("monitoring_plan", result["pending_items"])
        self.assertNotIn("objective_evidence", result)

    def test_past_measurement_and_refused_measurement_are_not_a_recovery_plan(self):
        for quote in ("我昨天测血压了", "我明天不想测血压", "我不会测血压", "医生建议我测血压"):
            with self.subTest(quote=quote):
                state = make_state("missed_measurement")
                state["latest_patient_message"] = quote
                result, _ = self.evaluate(state, proposal(objective_evidence={"monitoring_plan": quote}))
                self.assertEqual(result["outcome"], "continue")
                self.assertIn("monitoring_plan", result["pending_items"])

    def test_original_patient_plan_in_recent_trace_can_supply_grounded_evidence(self):
        state = make_state("missed_measurement")
        state["trace"] = [{"event": "USER_MESSAGE", "detail": {"message": "我准备每天测血压并记录。"}}]
        result, _ = self.evaluate(state, proposal(objective_evidence={"monitoring_plan": "我准备每天测血压并记录"}))
        self.assertEqual(result["outcome"], "completed")

    def test_unresolved_patient_question_prevents_normal_completion(self):
        state = make_state()
        state["v41_pending_patient_question"] = "为什么按时吃药还是血压高？"
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("patient_question", result["pending_items"])
        result, _ = self.evaluate(state, context={"patient_question_resolved": True,
                                                "grounded_reply": "现有依据提示需要结合测量情况评估。"})
        self.assertEqual(result["outcome"], "completed")

    def test_medication_tool_followup_cannot_be_ignored(self):
        state = make_state()
        state["medication_guidance"] = {"needs_followup": True}
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("medication_details", result["pending_items"])

    def test_current_tool_failure_cannot_be_reported_as_completed(self):
        context = {"tool_results_this_turn": [{"tool": "MEDICATION_GUIDANCE", "status": "error"}]}
        result, _ = self.evaluate(make_state(), context=context)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("tool:MEDICATION_GUIDANCE", result["pending_items"])

    def test_successful_retry_resolves_effective_previous_tool_error(self):
        context = {"previous_tool_results": [{"tool": "MEDICATION_GUIDANCE", "status": "error"}],
                   "tool_results_this_turn": [{"tool": "MEDICATION_GUIDANCE", "status": "success"}]}
        result, _ = self.evaluate(make_state(), context=context)
        self.assertEqual(result["outcome"], "completed")
        context["tool_results_this_turn"] = [{"tool": "CHECK_BP_HISTORY", "status": "success"}]
        result, _ = self.evaluate(make_state(), context=context)
        self.assertEqual(result["outcome"], "continue")

    def test_explicitly_unanswerable_necessary_topic_gives_partial_not_normal_completion(self):
        state = make_state()
        state["coverage"]["medication"] = False
        state["medication_status"] = "unknown"
        state["deferred_topics"] = {"medication": "unable"}
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "partial")
        self.assertIn("medication", result["pending_items"])
        self.assertIn("不能据此排除风险", result["reason"])

    def test_other_available_topics_continue_when_one_topic_is_deferred(self):
        state = make_state()
        state["coverage"].update(medication=False, current_symptoms=False)
        state["deferred_topics"] = {"medication": "refused"}
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertEqual(result["next_goal"], "current_symptoms")

    def test_explicit_request_to_end_is_partial_with_unconfirmed_items(self):
        state = make_state()
        state["coverage"]["red_flags"] = False
        state["latest_patient_message"] = "今天先结束随访"
        result, _ = self.evaluate(state, proposal("partial"))
        self.assertEqual(result["outcome"], "partial")
        self.assertIn("red_flags", result["pending_items"])
        self.assertIn("不能据此排除风险", result["reason"])

    def test_denied_ending_does_not_end_the_task_even_with_an_old_request_flag(self):
        statements = ("不要结束随访，我还能继续", "我不想结束随访", "别暂停本次随访",
                      "我不是要结束随访，请继续", "我不会结束随访，我还想回答")
        for statement in statements:
            with self.subTest(statement=statement):
                state = make_state()
                state["coverage"]["red_flags"] = False
                state["latest_patient_message"] = statement
                state["v41_patient_requested_end"] = True
                result, model = self.evaluate(state, proposal("partial"))
                self.assertEqual(result["outcome"], "continue")
                self.assertFalse(model.call_args.args[2]["patient_requested_end"])

    def test_questions_and_hypothetical_ending_do_not_authorize_partial_completion(self):
        statements = ("什么时候结束随访？", "现在可以结束随访吗", "是否结束本次随访",
                      "如果结束随访，我以后怎么办？", "结束随访后我该做什么？")
        for statement in statements:
            with self.subTest(statement=statement):
                state = make_state()
                state["coverage"]["red_flags"] = False
                state["latest_patient_message"] = statement
                result, model = self.evaluate(state, proposal("partial"))
                self.assertEqual(result["outcome"], "continue")
                self.assertFalse(model.call_args.args[2]["patient_requested_end"])

    def test_positive_ending_request_in_a_later_clause_is_honored(self):
        statements = ("请结束本次随访", "今天先到这里", "我不想继续", "我今天不能继续随访",
                      "刚才我说不要结束随访，但现在请结束本次随访")
        for statement in statements:
            with self.subTest(statement=statement):
                state = make_state()
                state["coverage"]["red_flags"] = False
                state["latest_patient_message"] = statement
                result, model = self.evaluate(state, proposal("partial"))
                self.assertEqual(result["outcome"], "partial")
                self.assertTrue(model.call_args.args[2]["patient_requested_end"])

    def test_historical_ending_request_does_not_override_current_intent_to_continue(self):
        state = make_state()
        state["coverage"]["red_flags"] = False
        state["trace"] = [{"event": "USER_MESSAGE", "detail": {"message": "今天先结束随访"}}]
        state["latest_patient_message"] = "我现在还能继续，请继续随访"
        result, _ = self.evaluate(state, proposal("partial"))
        self.assertEqual(result["outcome"], "continue")

    def test_model_alone_cannot_claim_patient_is_unable_or_wants_to_end(self):
        result, _ = self.evaluate(make_state(), proposal("partial"))
        self.assertEqual(result["outcome"], "continue")

    def test_no_patient_answer_cannot_be_replaced_by_history(self):
        state = make_state()
        state.update(latest_patient_message="", patient_memory={"last_followup": {"completed": True}})
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("patient_answer", result["pending_items"])

    def test_model_failure_and_exception_preserve_task_without_normal_completion(self):
        for model in (Mock(return_value=None), Mock(side_effect=TimeoutError("private model credential"))):
            with self.subTest(model=model):
                result = evaluate_completion(make_state(), {}, model_call=model)
                self.assertEqual(result["outcome"], "continue")
                self.assertEqual(result["model_status"], "failed")
                self.assertIn("模型暂时不可用", result["reason"])
                self.assertNotIn("private", str(result))

    def test_strict_schema_rejects_bad_action_fields_and_types(self):
        invalid = [[], {}, proposal("invented"), proposal(code="print(1)"),
                   proposal(pending_items="bp"), proposal(next_goal=123), proposal(reason=""),
                   proposal(objective_evidence={"bp": "145/85"}),
                   proposal(objective_evidence={"monitoring_plan": ["明天测"]}),
                   proposal(pending_items=[{}]), proposal(reason="x" * 401)]
        for decision in invalid:
            with self.subTest(decision=decision):
                result, _ = self.evaluate(make_state(), decision)
                self.assertEqual(result["outcome"], "continue")
                self.assertEqual(result["model_status"], "invalid")

    def test_many_patient_turns_do_not_force_finish_or_partial_completion(self):
        state = make_state()
        state["coverage"]["medication"] = False
        state["trace"] = [{"event": "USER_MESSAGE", "detail": {"message": "补充%d" % index}}
                          for index in range(30)]
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("medication", result["pending_items"])

    def test_deferred_next_goal_is_redirected_to_available_information(self):
        state = make_state()
        state["coverage"].update(medication=False, red_flags=False)
        state["deferred_topics"] = {"medication": "unable"}
        result, _ = self.evaluate(state, proposal("continue", next_goal="medication"))
        self.assertEqual(result["next_goal"], "red_flags")

    def test_unclear_counter_does_not_authorize_partial_completion(self):
        state = make_state()
        state["coverage"]["medication"] = False
        state["deferred_topics"] = {"medication": "unclear_limit"}
        result, _ = self.evaluate(state)
        self.assertEqual(result["outcome"], "continue")

    def test_context_and_state_are_not_mutated_by_evaluation(self):
        state, context = make_state(), {"recent_dialogue": [{"role": "patient", "message": "测试"}]}
        previous_state, previous_context = deepcopy(state), deepcopy(context)
        def model(purpose, system, payload):
            payload["evaluation_context"]["recent_dialogue"][0]["message"] = "模型试图改写"
            return proposal()
        result = evaluate_completion(state, context, model_call=model)
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(state, previous_state)
        self.assertEqual(context, previous_context)


if __name__ == "__main__":
    unittest.main()

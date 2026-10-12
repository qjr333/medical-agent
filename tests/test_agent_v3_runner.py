"""Agent Loop V3 failure, bound and tool freshness regressions (Mock models)."""

import contextlib
import copy
import io
import os
import unittest
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1"}):
    from backend.config import load_demo_data
    from agent import runner, state_manager
    from medical_tools import guidance
    from medical_tools.registry import get_tool_registry
    from medical_tools.safety import DANGER_PATTERNS


EVIDENCE = [{
    "block_id": "mock_medication_1", "source_id": "mock",
    "source_title": "Mock用药证据", "source_pages": "1", "year": 2026,
    "text": "遵循已有处方，不自行改变降压药剂量，具体方案向医生或药师核实。",
}]


def make_state(*, complete=True):
    data = load_demo_data()
    configs = copy.deepcopy(data["trigger_configs"])
    configs["scheduled"].update(bp_systolic=125, bp_diastolic=78)
    state = state_manager.create_patient_state(
        "scheduled", DEMO_PATIENT=data["patient"],
        DEMO_BP_HISTORY=[{"label": f"历史{i}", "systolic": 125, "diastolic": 78} for i in range(3)],
        TRIGGER_CONFIGS=configs,
    )
    state.update(medication_status="taken", medication_timing="unknown",
                 symptom_status="denied", red_flag_status="denied",
                 denied_red_flags=list(DANGER_PATTERNS),
                 measurement_context="安静休息5分钟后测量")
    state["coverage"] = dict.fromkeys(state["coverage"], True)
    state["topic_status"] = dict.fromkeys(state["coverage"], "answered")
    if not complete:
        state["coverage"].update(current_symptoms=False, red_flags=False)
        state["topic_status"].update(current_symptoms="unanswered", red_flags="unanswered")
        state.update(symptom_status="unknown", red_flag_status="unknown", denied_red_flags=[])
    return state


def medication_answer(state):
    return {"status": "ok", "message": "请按医生原来安排服药；具体方案向医生或药师确认。",
            "needs_followup": False, "followup_question": "",
            "needs_professional_review": True, "reason": "Mock证据回答", "source_blocks": []}


class AgentV3RunnerRegressionTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def tools(self):
        return {name: Mock(side_effect=function) for name, function in
                get_tool_registry(call_json_model=lambda **kwargs: {}).items()}

    def test_maximum_steps_interrupts_without_report_then_resumes_remaining_work(self):
        state = make_state()
        tools = self.tools()
        response = runner.run_agent(state, tool_registry=tools, max_steps=1)
        self.assertEqual(state["followup_status"], "interrupted")
        self.assertTrue(state["pending_tasks"])
        self.assertFalse(response.get("report"))
        self.assertFalse(state.get("report"))
        self.assertNotIn("FINISH", [event["event"] for event in state["trace"]])
        self.assertTrue(any("LIMIT" in event["event"] for event in state["trace"]))
        self.assertTrue(response["message"])
        history_calls = tools["CHECK_BP_HISTORY"].call_count
        resumed = runner.run_agent(state, tool_registry=tools)
        self.assertEqual(state["followup_status"], "completed")
        self.assertTrue(resumed["report"])
        self.assertEqual(tools["CHECK_BP_HISTORY"].call_count, history_calls)
        self.assertEqual([event["event"] for event in state["trace"]].count("FINISH"), 1)

    def test_empty_rag_result_is_attempted_once_and_uses_conservative_response(self):
        state = make_state(complete=False)
        question = "今天忘记吃药了，可以自行补吃两片降压药吗？"
        state["latest_patient_question"] = question
        tools = self.tools()
        with patch.object(guidance, "retrieve_medical_guidance", return_value=[]) as retrieve:
            first = runner.run_agent(state, tool_registry=tools)
            self.assertEqual(retrieve.call_count, 1)
            self.assertEqual(state["retrieved_guidance"], [])
            self.assertTrue(state["medication_guidance"]["needs_professional_review"])
            self.assertIn("不能", first["message"])
            state["latest_patient_question"] = question
            runner.run_agent(state, tool_registry=tools)
            self.assertEqual(retrieve.call_count, 1)
            self.assertEqual(tools["MEDICATION_GUIDANCE"].call_count, 1)
        self.assertNotEqual(state["followup_status"], "completed")

    def test_repeated_question_reuses_tools_but_changed_patient_facts_allow_new_evidence(self):
        state = make_state(complete=False)
        question = "我可以自行增加降压药剂量吗？"
        state["latest_patient_question"] = question
        tools = self.tools()
        tools["MEDICATION_GUIDANCE"] = Mock(side_effect=medication_answer)
        with patch.object(guidance, "retrieve_medical_guidance", return_value=EVIDENCE) as retrieve:
            runner.run_agent(state, tool_registry=tools)
            self.assertEqual(retrieve.call_count, 1)
            self.assertEqual(tools["MEDICATION_GUIDANCE"].call_count, 1)
            state["latest_patient_question"] = question
            runner.run_agent(state, tool_registry=tools)
            self.assertEqual(retrieve.call_count, 1)
            self.assertEqual(tools["MEDICATION_GUIDANCE"].call_count, 1)
            state.update(bp_systolic=162, bp_diastolic=96,
                         symptom_status="denied", red_flag_status="denied",
                         denied_red_flags=list(DANGER_PATTERNS), latest_patient_question=question)
            state["coverage"].update(current_symptoms=True, red_flags=True)
            state["topic_status"].update(current_symptoms="answered", red_flags="answered")
            runner.run_agent(state, tool_registry=tools)
            self.assertGreaterEqual(retrieve.call_count, 2)
            self.assertEqual(tools["MEDICATION_GUIDANCE"].call_count, 2)

    def test_old_patient_question_is_consumed_and_plain_reply_does_not_reinvoke_medication_tool(self):
        state = make_state(complete=False)
        state["latest_patient_question"] = "我可以自行增加降压药剂量吗？"
        tools = self.tools()
        tools["MEDICATION_GUIDANCE"] = Mock(side_effect=medication_answer)
        with patch.object(guidance, "retrieve_medical_guidance", return_value=EVIDENCE) as retrieve:
            runner.run_agent(state, tool_registry=tools)
            self.assertFalse(state.get("latest_patient_question"))
            state_manager.merge_user_patch(state, {"medication_status": "taken"}, "今天按时吃药了")
            runner.run_agent(state, tool_registry=tools)
            self.assertEqual(retrieve.call_count, 1)
            self.assertEqual(tools["MEDICATION_GUIDANCE"].call_count, 1)

    def test_rag_timeout_or_exception_is_traced_without_repeated_retry(self):
        for failure in (TimeoutError("offline timeout"), RuntimeError("offline retrieval error")):
            with self.subTest(error=type(failure).__name__):
                state = make_state(complete=False)
                state["latest_patient_question"] = "我可以自行增加降压药剂量吗？"
                tools = self.tools()
                tools["RETRIEVE_MEDICAL_GUIDANCE"] = Mock(side_effect=failure)
                response = runner.run_agent(state, tool_registry=tools)
                tools["RETRIEVE_MEDICAL_GUIDANCE"].assert_called_once()
                self.assertTrue(response["message"])
                self.assertFalse(state.get("retrieved_guidance"))
                errors = [event["detail"] for event in state["trace"] if event["event"] == "TOOL_ERROR"]
                self.assertTrue(errors)
                error = next(item for item in errors if item["tool"] == "RETRIEVE_MEDICAL_GUIDANCE")
                self.assertEqual(error["error_type"], type(failure).__name__)
                self.assertTrue(error["recoverable"])
                self.assertGreaterEqual(error["duration_ms"], 0)
                self.assertEqual(state["symptom_status"], "unknown")
                self.assertNotEqual(state["followup_status"], "completed")

    def test_model_timeout_uses_safe_tool_fallback_and_records_failure(self):
        state = make_state(complete=False)
        state["latest_patient_question"] = "我可以自行增加降压药剂量吗？"
        model = Mock(side_effect=TimeoutError("Mock DeepSeek timeout"))
        with patch.object(guidance, "retrieve_medical_guidance", return_value=EVIDENCE):
            response = runner.run_agent(state, call_json_model=model)
        self.assertTrue(response["message"])
        self.assertEqual(state["symptom_status"], "unknown")
        self.assertTrue(state["medication_guidance"]["needs_professional_review"])
        self.assertTrue(any(event["event"] == "TOOL_ERROR" for event in state["trace"]))
        self.assertNotEqual(state["followup_status"], "completed")

    def test_recoverable_rag_failure_can_retry_the_same_question_on_a_later_turn(self):
        state = make_state(complete=False)
        question = "我可以自行增加降压药剂量吗？"
        state["latest_patient_question"] = question
        tools = self.tools()
        tools["MEDICATION_GUIDANCE"] = Mock(side_effect=medication_answer)
        original = get_tool_registry(call_json_model=lambda **kwargs: {})["RETRIEVE_MEDICAL_GUIDANCE"]
        retrieve = Mock(side_effect=[TimeoutError("offline first attempt"), EVIDENCE])

        def retrieval(state, query):
            result = retrieve()
            with patch.object(guidance, "retrieve_medical_guidance", return_value=result):
                return original(state, query)

        tools["RETRIEVE_MEDICAL_GUIDANCE"] = Mock(side_effect=retrieval)
        runner.run_agent(state, tool_registry=tools)
        self.assertEqual(retrieve.call_count, 1)
        self.assertFalse(state.get("retrieved_guidance"))
        state["latest_patient_question"] = question
        runner.run_agent(state, tool_registry=tools)
        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(state["retrieved_guidance"], EVIDENCE)
        self.assertEqual(tools["MEDICATION_GUIDANCE"].call_count, 2)

    def test_risk_engine_failure_interrupts_without_assuming_normal_risk(self):
        state = make_state(complete=False)
        tools = self.tools()
        tools["ASSESS_RISK"] = Mock(side_effect=RuntimeError("offline safety failure"))
        response = runner.run_agent(state, tool_registry=tools)
        self.assertIn(state["followup_status"], {"failed", "interrupted"})
        self.assertFalse(response.get("report"))
        self.assertTrue(response["message"])
        for name in ("CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            tools[name].assert_not_called()
        self.assertTrue(any(event["event"] == "TOOL_ERROR" for event in state["trace"]))

    def test_high_risk_bypasses_step_limit_and_regular_tool_failures(self):
        state = make_state(complete=False)
        state.update(red_flags=["胸痛"], red_flag_status="reported", symptoms=["胸痛"], symptom_status="reported")
        tools = self.tools()
        for name in ("CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            tools[name].side_effect = AssertionError("ordinary work cannot precede safety")
        response = runner.run_agent(state, tool_registry=tools, max_steps=1)
        self.assertEqual(state["followup_status"], "escalated")
        self.assertEqual(state["risk_level"], "high")
        self.assertEqual(response["report"]["risk"]["level"], "high")
        tools["ASSESS_RISK"].assert_called_once()
        tools["CREATE_ALERT"].assert_called_once()

    def test_inconsistent_injected_risk_result_cannot_hide_existing_high_bp_rule(self):
        state = make_state(complete=False)
        state.update(bp_systolic=180, bp_diastolic=120)
        tools = self.tools()
        tools["ASSESS_RISK"] = Mock(return_value={"risk_level": "normal", "reasons": []})
        response = runner.run_agent(state, tool_registry=tools)
        self.assertEqual(state["followup_status"], "escalated")
        self.assertEqual(state["risk_level"], "high")
        self.assertEqual(response["report"]["risk"]["level"], "high")
        for name in ("CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            tools[name].assert_not_called()

    def test_tool_execution_trace_contains_compact_results_and_duration(self):
        state = make_state()
        response = runner.run_agent(state, call_json_model=lambda **kwargs: {})
        self.assertEqual(state["followup_status"], "completed")
        executions = [event["detail"] for event in state["trace"] if event["event"] == "TOOL_RESULT"]
        self.assertTrue(executions)
        for detail in executions:
            self.assertTrue({"tool", "status", "result_summary", "duration_ms", "next_status"} <= detail.keys())
            self.assertGreaterEqual(detail["duration_ms"], 0)
            self.assertNotIn("state", detail)
            self.assertNotIn("patient_state", detail)
        self.assertEqual(response["trace"], state["trace"])


if __name__ == "__main__":
    unittest.main()

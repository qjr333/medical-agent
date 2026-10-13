"""Offline V4.1 tool-loop regressions: no live model, patient DB or network.

Real registered tools are used where their behavior is under test. Slow workers
are synchronized with events so timeout isolation needs no long sleep.
"""

import contextlib
import io
import json
import unittest
from collections import deque
from copy import deepcopy
from threading import Event
from unittest.mock import Mock, patch

from agent import llm_v4, state_manager
from medical_tools import bp_trend, guidance, medication, registry


EVIDENCE = [{
    "block_id": "fixture-knowledge-1", "source_id": "fixture-source",
    "source_title": "模拟高血压管理依据", "year": 2026, "source_pages": "2",
    "text": "遵照原医嘱服药，记录血压与服药情况，具体处方问题请咨询医生或药师。",
}]


def make_state(*, complete=False):
    state = state_manager.create_patient_state(
        "scheduled",
        DEMO_PATIENT={"patient_id": "v41-tools-fixture", "patient_name": "模拟患者",
                      "age": 72, "conditions": ["高血压"]},
        DEMO_BP_HISTORY=[{"systolic": 140, "diastolic": 85},
                         {"systolic": 142, "diastolic": 86},
                         {"systolic": 145, "diastolic": 85}],
        TRIGGER_CONFIGS={"scheduled": {"followup_reason": "模拟随访",
                                      "bp_systolic": 145, "bp_diastolic": 85}},
    )
    if complete:
        state.update(medication_status="taken", symptom_status="denied", red_flag_status="denied",
                     denied_red_flags=list(llm_v4.DANGER_LABELS),
                     measurement_context="安静休息后测量", latest_patient_message="今天按时服药，没有不舒服")
        state["coverage"] = dict.fromkeys(state["coverage"], True)
        state["topic_status"] = dict.fromkeys(state["coverage"], "answered")
    return state


def tool(action, **arguments):
    return {"action": action, "arguments": arguments, "reason": "模拟模型按当前信息选择工具"}


def ask(question="您还有什么情况想补充？", **extra):
    return {"action": "ASK_USER", "question": question, "topic": "additional_information",
            "reason": "继续确认患者情况", **extra}


class FakeModel:
    def __init__(self, *plans, medical_result=None, visit_result=None):
        self.plans = deque(plans)
        self.planner_inputs = []
        self.tool_inputs = []
        self.medical_result = medical_result if medical_result is not None else {
            "patient_message": "请核对原医嘱，可以设置服药提醒；具体处方问题请咨询医生或药师。",
            "needs_followup": False, "followup_question": "", "needs_professional_review": True,
            "reason": "结合患者依从性及实际检索证据",
        }
        self.visit_result = visit_result if visit_result is not None else {
            "level": "contact_clinician", "advice": "请携带实际血压及服药记录联系负责治疗的医生。",
            "reason": ["依从性问题需要进一步核实"], "professional_review": True,
            "continue_home_monitoring": True, "preparation": ["实际血压和服药记录"],
        }

    def __call__(self, *, system_prompt, user_prompt, temperature=0.1):
        payload = json.loads(user_prompt)
        if "trigger_event" in payload:
            self.planner_inputs.append(payload)
            if not self.plans:
                raise AssertionError("Unexpected extra planner call")
            decision = self.plans.popleft()
            return decision(payload) if callable(decision) else deepcopy(decision)
        if "tool_input" in payload:
            self.tool_inputs.append({"system_prompt": system_prompt, **payload})
            if "用药管理辅助模块" in system_prompt:
                return deepcopy(self.medical_result)
            if "就医指导模块" in system_prompt:
                return deepcopy(self.visit_result)
        raise AssertionError("Unexpected model purpose")


class AgentV41ToolsTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.retrieval = self.stack.enter_context(patch.object(
            guidance, "retrieve_medical_guidance", return_value=deepcopy(EVIDENCE),
        ))
        self.stack.enter_context(patch.object(guidance, "knowledge_revision", return_value="fixture-rev-1"))

    def run_model(self, state, model, **kwargs):
        return llm_v4.run_agent_v4(state, call_json_model=model, **kwargs)

    def seed_evidence(self, state):
        state.update(retrieved_guidance=deepcopy(EVIDENCE), last_guidance_query="用药依从性管理")
        state["guidance_fingerprint"] = guidance.execution_fingerprint(
            state, "RETRIEVE_MEDICAL_GUIDANCE", state["last_guidance_query"],
        )

    def assert_no_completion(self, response):
        self.assertNotEqual(response["state"]["followup_status"], "completed")
        self.assertFalse(response.get("report"))
        self.assertFalse(any(event["event"] == "FINISH" for event in response["trace"]))

    def test_medication_and_visit_tools_are_autonomously_chained_with_real_evidence(self):
        state = make_state(complete=True)
        state.update(medication_status="irregular", medication_detail="最近经常忘记服药",
                     latest_patient_question="经常忘记吃降压药怎么办？")

        def reply_from_tool_results(payload):
            data = {result["tool"]: result for result in payload["tool_results_this_turn"]}
            return ask(reply=data["MEDICATION_GUIDANCE"]["data"]["message"] +
                       data["GENERATE_VISIT_ADVICE"]["data"]["advice"],
                       reply_sources=[data["RETRIEVE_MEDICAL_GUIDANCE"]["sources"][0]["block_id"]])

        model = FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="高血压用药依从性提醒"),
                          tool("MEDICATION_GUIDANCE"), tool("GENERATE_VISIT_ADVICE"),
                          reply_from_tool_results)
        response = self.run_model(state, model)
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertEqual(len(model.tool_inputs), 2)
        for entry in model.tool_inputs:
            self.assertIn(EVIDENCE[0]["text"], entry["tool_input"])
            self.assertIn("外部", entry["system_prompt"])
            self.assertIn("工具权限", entry["system_prompt"])
        results = model.planner_inputs[-1]["tool_results_this_turn"]
        self.assertEqual([result["tool"] for result in results], [
            "RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE",
        ])
        self.assertTrue(all(result["status"] == "success" and result["sources"] for result in results))
        self.assertIn("设置服药提醒", response["message"])
        self.assertIn("实际血压及服药记录", response["message"])
        self.assertFalse(medication.contains_unsafe_medication_instruction(response["message"]))
        purposes = {event["detail"].get("purpose") for event in state["trace"]
                    if event["event"] == "LLM_CALL" and event["detail"].get("ok")}
        self.assertTrue({"PLAN", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"} <= purposes)

    def test_tool_order_is_model_selected_and_not_a_fixed_script(self):
        state = make_state(complete=True)
        model = FakeModel(tool("CHECK_BP_HISTORY"), ask())
        response = self.run_model(state, model)
        self.assertEqual(state["bp_analysis"]["record_count"], 3)
        self.assertEqual(model.planner_inputs[-1]["tool_results_this_turn"][0]["data"]["history"],
                         state["bp_analysis"]["history"])
        self.assertEqual([entry["detail"]["tool"] for entry in response["trace"]
                          if entry["event"] == "V4_TOOL_CALL"], ["CHECK_BP_HISTORY"])
        self.retrieval.assert_not_called()
        self.assertFalse(model.tool_inputs)

    def test_invalid_actions_never_execute_arbitrary_or_unregistered_tools(self):
        for invalid in ("EXECUTE_PYTHON", "__import__", "DELETE_PATIENT", None, ["CHECK_BP_HISTORY"]):
            with self.subTest(action=invalid):
                spies = {name: Mock() for name in registry.TOOL_REGISTRY}
                model = FakeModel({"action": invalid, "arguments": {}}, ask())
                response = self.run_model(make_state(), model, tool_registry=spies)
                self.assert_no_completion(response)
                self.assertTrue(any(entry["event"] == "LLM_VALIDATION_ERROR" for entry in response["trace"]))
                for spy in spies.values():
                    spy.assert_not_called()

    def test_invalid_arguments_never_reach_registered_execution(self):
        invalid_decisions = [
            tool("CHECK_BP_HISTORY", patient_id="other-patient"),
            tool("MEDICATION_GUIDANCE", dose=10),
            tool("GENERATE_VISIT_ADVICE", code="open('/secret')"),
            tool("RETRIEVE_MEDICAL_GUIDANCE", query=""),
            tool("RETRIEVE_MEDICAL_GUIDANCE", query=99),
            tool("RETRIEVE_MEDICAL_GUIDANCE", query="x" * 451),
            tool("RETRIEVE_MEDICAL_GUIDANCE", query="家庭血压", url="https://invalid.example"),
            {"action": "CHECK_BP_HISTORY", "arguments": []},
            {"action": "CHECK_BP_HISTORY", "arguments": {}, "python": "print(1)"},
            {"action": "RETRIEVE_MEDICAL_GUIDANCE", "arguments": {"query": "query1"}, "query": "query2"},
        ]
        for decision in invalid_decisions:
            with self.subTest(decision=decision):
                spies = {name: Mock() for name in registry.TOOL_REGISTRY}
                response = self.run_model(make_state(), FakeModel(decision, ask()), tool_registry=spies)
                self.assertTrue(any(entry["event"] == "LLM_VALIDATION_ERROR" for entry in response["trace"]))
                for spy in spies.values():
                    spy.assert_not_called()

    def test_identical_tool_inputs_execute_once_then_pause_and_can_resume(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        model = FakeModel(tool("CHECK_BP_HISTORY"), tool("CHECK_BP_HISTORY"), tool("CHECK_BP_HISTORY"))
        response = self.run_model(state, model, tool_registry={"CHECK_BP_HISTORY": history})
        history.assert_called_once()
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertEqual(state["loop_state"]["reason"], "repeated_tool")
        self.assert_no_completion(response)
        resumed = FakeModel(tool("CHECK_BP_HISTORY"), ask("请补充今天的用药情况"))
        self.run_model(state, resumed, tool_registry={"CHECK_BP_HISTORY": history})
        history.assert_called_once()
        self.assertEqual(resumed.planner_inputs[-1]["tool_results_this_turn"][0]["status"], "cached")
        self.assertEqual(state["followup_status"], "waiting_for_user")

    def test_completed_tool_results_survive_request_budget_and_enter_next_plan(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        self.run_model(state, FakeModel(tool("CHECK_BP_HISTORY")),
                       tool_registry={"CHECK_BP_HISTORY": history}, max_steps=1)
        model = FakeModel(ask())
        self.run_model(state, model, tool_registry={"CHECK_BP_HISTORY": history})
        result = model.planner_inputs[0]["previous_tool_results"][0]
        self.assertEqual(result["tool"], "CHECK_BP_HISTORY")
        self.assertEqual(result["data"]["record_count"], 3)
        history.assert_called_once()

    def test_changed_measurement_invalidates_cached_history(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        self.run_model(state, FakeModel(tool("CHECK_BP_HISTORY"), ask("第一问")),
                       tool_registry={"CHECK_BP_HISTORY": history})
        state.update(bp_systolic=155, bp_diastolic=90)
        model = FakeModel(tool("CHECK_BP_HISTORY"), ask("第二问"))
        self.run_model(state, model, tool_registry={"CHECK_BP_HISTORY": history})
        self.assertEqual(history.call_count, 2)
        self.assertEqual(state["bp_analysis"]["record_count"], 4)
        self.assertEqual(state["bp_analysis"]["history"][-1]["systolic"], 155)
        self.assertFalse(model.planner_inputs[0]["previous_tool_results"])

    def test_changed_knowledge_revision_invalidates_cached_retrieval(self):
        state = make_state()
        self.run_model(state, FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="家庭监测"), ask("第一问")))
        with patch.object(guidance, "knowledge_revision", return_value="fixture-rev-2"):
            model = FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="家庭监测"), ask("第二问"))
            self.run_model(state, model)
        self.assertEqual(self.retrieval.call_count, 2)
        self.assertFalse(model.planner_inputs[0]["previous_tool_results"])
        self.assertFalse(model.planner_inputs[0]["retrieval_evidence"])

    def test_changed_patient_facts_invalidate_cached_retrieval(self):
        state = make_state()
        self.run_model(state, FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="家庭监测"), ask("第一问")))
        state["medication_detail"] = "患者补充最近漏服一次"
        self.run_model(state, FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="家庭监测"), ask("第二问")))
        self.assertEqual(self.retrieval.call_count, 2)

    def test_query_a_b_a_cache_restores_the_matching_live_evidence(self):
        state = make_state()
        def retrieve(query, **kwargs):
            return [{**EVIDENCE[0], "block_id": query, "text": "模拟依据：" + query}]
        self.retrieval.side_effect = retrieve
        for query, question in (("query-A", "第一问"), ("query-B", "第二问")):
            self.run_model(state, FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query=query), ask(question)))
        model = FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="query-A"), ask("第三问"))
        self.run_model(state, model)
        self.assertEqual(self.retrieval.call_count, 2)
        result = model.planner_inputs[-1]["tool_results_this_turn"][0]
        self.assertEqual(result["status"], "cached")
        self.assertEqual(result["data"]["results"][0]["block_id"], "query-A")
        self.assertEqual(state["last_guidance_query"], "query-A")
        self.assertEqual(state["retrieved_guidance"][0]["block_id"], "query-A")
        self.assertEqual(model.planner_inputs[-1]["retrieval_evidence"][0]["block_id"], "query-A")

    def test_retrieval_failure_is_error_feedback_and_blocks_finish(self):
        state = make_state(complete=True)
        self.retrieval.side_effect = RuntimeError("private retrieval password")
        model = FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="高血压管理"),
                          {"action": "FINISH"}, ask())
        response = self.run_model(state, model)
        failure = model.planner_inputs[1]["tool_results_this_turn"][0]
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["error_type"], "RuntimeError")
        self.assert_no_completion(response)
        self.assertFalse(state["retrieved_guidance"])
        self.assertTrue(any(event["event"] == "FINISH_BLOCKED" for event in state["trace"]))
        self.assertNotIn("private retrieval password", json.dumps(response, ensure_ascii=False))
        self.assertTrue(all(event["detail"]["status"] == "error" for event in state["trace"]
                            if event["event"] == "V4_TOOL_CALL"))

    def test_failed_tool_is_visible_next_request_and_cannot_be_bypassed_by_finish(self):
        state = make_state(complete=True)
        self.retrieval.side_effect = TimeoutError("private retrieval failure")
        self.run_model(state, FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="高血压管理"), ask("第一问")))
        model = FakeModel({"action": "FINISH"}, ask("第二问"))
        response = self.run_model(state, model)
        previous = model.planner_inputs[0]["previous_tool_results"]
        self.assertEqual(previous[0]["tool"], "RETRIEVE_MEDICAL_GUIDANCE")
        self.assertEqual(previous[0]["status"], "error")
        self.assert_no_completion(response)
        self.retrieval.side_effect = None
        self.retrieval.return_value = deepcopy(EVIDENCE)
        recovered = self.run_model(state, FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="高血压管理"),
                                                    {"action": "FINISH", "reply": "本次随访记录已完成。"}))
        self.assertEqual(self.retrieval.call_count, 2)
        self.assertEqual(recovered["state"]["followup_status"], "completed")

    def test_stale_evidence_cannot_be_used_by_medication_or_visit_tools(self):
        for action in ("MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            with self.subTest(action=action):
                state = make_state(complete=True)
                self.seed_evidence(state)
                state.update(medication_status="missed", medication_detail="新补充的漏服情况")
                model = FakeModel(tool(action), ask())
                self.run_model(state, model)
                result = model.planner_inputs[-1]["tool_results_this_turn"][0]
                self.assertFalse(result["sources"])
                self.assertFalse(model.tool_inputs)
                if action == "MEDICATION_GUIDANCE":
                    self.assertEqual(result["status"], "fallback")
                else:
                    self.assertIn("检索结果不足", result["data"]["advice"])

    def test_medication_model_failure_is_not_a_successful_fallback_assessment(self):
        state = make_state(complete=True)
        self.seed_evidence(state)
        model = FakeModel(tool("MEDICATION_GUIDANCE"), {"action": "FINISH"}, ask(), medical_result={})
        response = self.run_model(state, model)
        failure = model.planner_inputs[1]["tool_results_this_turn"][0]
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["error_type"], "ToolModelFailure")
        self.assertIsNone(state["medication_guidance"])
        self.assert_no_completion(response)
        calls = [event["detail"] for event in state["trace"] if event["event"] == "LLM_CALL"
                 and event["detail"]["purpose"] == "MEDICATION_GUIDANCE"]
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["ok"])

    def test_invalid_visit_model_output_is_error_and_not_a_generated_advice(self):
        state = make_state(complete=True)
        self.seed_evidence(state)
        model = FakeModel(tool("GENERATE_VISIT_ADVICE"), ask(),
                          visit_result={"level": "invented_level", "advice": "虚构分级"})
        response = self.run_model(state, model)
        self.assertEqual(model.planner_inputs[-1]["tool_results_this_turn"][0]["status"], "error")
        self.assertIsNone(state["visit_advice"])
        self.assert_no_completion(response)

    def test_invalid_tool_result_does_not_commit_shadow_state_or_claim_success(self):
        state = make_state()
        def invalid_history(shadow):
            shadow["bp_systolic"] = 999
            shadow["bp_analysis"] = {"made_up": True}
            return {"record_count": "three", "trend": "diagnosed"}
        model = FakeModel(tool("CHECK_BP_HISTORY"), ask())
        response = self.run_model(state, model, tool_registry={"CHECK_BP_HISTORY": invalid_history})
        self.assertEqual(state["bp_systolic"], 145)
        self.assertIsNone(state["bp_analysis"])
        self.assertEqual(model.planner_inputs[-1]["tool_results_this_turn"][0]["status"], "error")
        self.assert_no_completion(response)

    def test_empty_retrieval_has_empty_status_and_explicit_insufficient_evidence_reply(self):
        state = make_state()
        state["latest_patient_question"] = "按时吃药为什么血压仍高？"
        self.retrieval.return_value = []
        model = FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="按时服药血压仍高"),
                          ask(reply="根据《虚构医学指南》，已经明确患者具体病因。"))
        response = self.run_model(state, model)
        result = model.planner_inputs[-1]["tool_results_this_turn"][0]
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["sources"], [])
        self.assertIn("没有足够", response["message"])
        self.assertNotIn("虚构医学指南", response["message"])
        self.assertFalse(medication.contains_unsafe_medication_instruction(response["message"]))

    def test_medication_without_evidence_is_marked_fallback_and_calls_no_medical_model(self):
        state = make_state()
        model = FakeModel(tool("MEDICATION_GUIDANCE"), ask())
        self.run_model(state, model)
        result = model.planner_inputs[-1]["tool_results_this_turn"][0]
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["data"]["status"], "insufficient_evidence")
        self.assertFalse(model.tool_inputs)
        self.assertFalse(result["sources"])

    def test_finish_is_rejected_until_necessary_current_information_is_confirmed(self):
        state = make_state()
        state["latest_patient_message"] = "我提供了本次血压"
        model = FakeModel({"action": "FINISH", "reply": "随访已完成"}, ask("今天服药了吗？"))
        response = self.run_model(state, model)
        self.assert_no_completion(response)
        blocked = next(event["detail"] for event in state["trace"] if event["event"] == "FINISH_BLOCKED")
        self.assertIn("medication", blocked["pending_topics"])
        self.assertTrue(blocked["has_patient_answer"])

    def test_finish_with_confirmed_information_completes_without_forcing_all_tools(self):
        state = make_state(complete=True)
        response = self.run_model(state, FakeModel({"action": "FINISH", "reply": "本次记录已完成。"}))
        self.assertEqual(state["followup_status"], "completed")
        self.assertTrue(response["report"])
        self.assertEqual(state["pending_tasks"], [])
        self.assertFalse(any(event["event"] == "V4_TOOL_CALL" for event in state["trace"]))
        self.retrieval.assert_not_called()

    def test_patient_requested_end_is_honored_without_claiming_unknown_risk_is_excluded(self):
        state = make_state()
        state.update(v41_patient_requested_end=True, latest_patient_message="今天先结束随访")
        response = self.run_model(state, FakeModel({"action": "FINISH", "reply": "好的，先结束。"}))
        self.assertEqual(state["followup_status"], "completed")
        self.assertIn("仍有信息未确认", response["message"])
        self.assertFalse(response["report"]["task_completion"]["all_required_information_confirmed"])

    def test_dangerous_state_overrides_even_zero_step_and_time_budgets(self):
        state = make_state()
        state.update(red_flags=["胸痛", "呼吸困难"], last_action="CHECK_BP_HISTORY")
        model = Mock(side_effect=AssertionError("Safety must not depend on a model"))
        history = Mock()
        response = self.run_model(state, model, tool_registry={"CHECK_BP_HISTORY": history},
                                  max_steps=0, max_seconds=0)
        model.assert_not_called()
        history.assert_not_called()
        self.assertEqual(state["followup_status"], "escalated")
        self.assertIn("立即", response["message"])
        self.assertTrue(response["alerts"])
        self.assertTrue(any(event["event"] == "SAFETY_OVERRIDE" for event in state["trace"]))

    def test_model_cannot_create_alert_for_an_ordinary_state(self):
        state = make_state()
        create_alert = Mock()
        model = FakeModel(tool("CREATE_ALERT"), ask())
        response = self.run_model(state, model, tool_registry={"CREATE_ALERT": create_alert})
        create_alert.assert_not_called()
        self.assertEqual(model.planner_inputs[-1]["tool_results_this_turn"][0]["status"], "blocked")
        self.assert_no_completion(response)

    def test_timed_out_tool_cannot_publish_late_state_or_trace(self):
        state = make_state()
        entered, release, finished = Event(), Event(), Event()
        def late_history(shadow):
            entered.set()
            try:
                if not release.wait(2):
                    raise AssertionError("Test worker was not released")
                shadow["bp_systolic"] = 999
                shadow["bp_history"].append({"systolic": 999, "diastolic": 998})
                state_manager.add_trace(shadow, "LATE_WORKER", {"invalid": True})
                return {"trend": "stable", "record_count": 99, "history": shadow["bp_history"]}
            finally:
                finished.set()
        try:
            response = self.run_model(state, FakeModel(tool("CHECK_BP_HISTORY")),
                                      tool_registry={"CHECK_BP_HISTORY": late_history},
                                      max_seconds=0.1)
            self.assertTrue(entered.is_set())
            self.assertEqual(state["followup_status"], "waiting_for_user")
            self.assertEqual(state["loop_state"]["reason"], "internal_time_budget")
            release.set()
            self.assertTrue(finished.wait(1))
            self.assertEqual(state["bp_systolic"], 145)
            self.assertIsNone(state["bp_analysis"])
            self.assertFalse(any(row["systolic"] == 999 for row in state["bp_history"]))
            self.assertFalse(any(event["event"] == "LATE_WORKER" for event in state["trace"]))
            self.assert_no_completion(response)
        finally:
            release.set()
            finished.wait(1)

    def test_many_patient_turns_do_not_force_a_completion(self):
        state = make_state()
        for number in range(12):
            state["latest_patient_message"] = "我还有第%d项情况" % number
            response = self.run_model(state, FakeModel(ask("请补充第%d项内容" % number)))
            self.assertEqual(state["followup_status"], "waiting_for_user")
            self.assert_no_completion(response)

    def test_unverified_source_and_fabricated_bp_are_not_presented_as_grounded_facts(self):
        for reply in ({"reply": "依据当前来源解释", "reply_sources": ["invented-source"]},
                      {"reply": "历史显示血压199/111", "reply_sources": [EVIDENCE[0]["block_id"]]}):
            with self.subTest(reply=reply):
                state = make_state()
                model = FakeModel(tool("RETRIEVE_MEDICAL_GUIDANCE", query="家庭监测"), ask(**reply))
                response = self.run_model(state, model)
                self.assertTrue(any(event["event"] == "OUTPUT_GROUNDING_OVERRIDE" for event in state["trace"]))
                self.assertNotIn("199/111", response["message"])
                self.assertNotIn("依据当前来源解释", response["message"])


if __name__ == "__main__":
    unittest.main()

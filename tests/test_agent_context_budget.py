"""Deterministic execution-boundary checks, not proof of live LLM autonomy."""
import contextlib
import io
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

# Test discovery must not load local .env mode flags or credentials into other
# legacy API fixtures. Import the real modules under the same offline boundary.
with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1"}), patch("dotenv.load_dotenv", return_value=False):
    from agent import followup_agent
    from agent.completion import evaluate_completion
    from agent.state_manager import add_trace, merge_user_patch
    from medical_tools import bp_trend, guidance
    from tests.test_agent_v50 import Model, ask, make_state, tool
    from tests.test_llm_completion_autonomy import stable_state, proposal
    from tests.test_validation_recovery_v5 import ExtractionModel, TEXT, state as extraction_state



class ContextAndExecutionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(guidance, "knowledge_revision", return_value="context-test"))

    def test_actual_question_is_not_duplicated_by_its_ask_execution_event(self):
        state = {"agent_version": "V5.0", "trace": [
            {"event": "ASK_USER", "detail": {"question": "现在还头晕吗？"}},
            {"event": "AGENT_RESPONSE", "detail": {"message": "谢谢您告诉我。\n\n现在还头晕吗？"}},
            {"event": "USER_MESSAGE", "detail": {"message": "站起来会有一点"}},
        ]}
        self.assertEqual(followup_agent._conversation(state), [
            {"role": "agent", "message": "谢谢您告诉我。\n\n现在还头晕吗？"},
            {"role": "patient", "message": "站起来会有一点"},
        ])

    def test_partial_legacy_trace_keeps_actual_question_without_response_event(self):
        state = {"agent_version": "V5.0", "trace": [
            {"event": "ASK_USER", "detail": {"question": "今天服药了吗？"}},
            {"event": "USER_MESSAGE", "detail": {"message": "还没呢"}},
        ]}
        self.assertEqual([row["message"] for row in followup_agent._conversation(state)], ["今天服药了吗？", "还没呢"])

    def test_processing_notice_does_not_consume_real_dialogue_window(self):
        state = {"agent_version": "V5.0", "trace": []}
        for index in range(5):
            state["trace"].extend([
                {"event": "ASK_USER", "detail": {"question": f"实际问题{index}？"}},
                {"event": "AGENT_RESPONSE", "detail": {"message": "自动处理中", "followup_status": "processing"}},
                {"event": "AGENT_RESPONSE", "detail": {"message": f"实际问题{index}？", "followup_status": "waiting_for_user"}},
                {"event": "USER_MESSAGE", "detail": {"message": f"真实回答{index}"}},
            ])
        rows = followup_agent._conversation(state)
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(row["role"] == "patient" for row in rows), 5)
        self.assertNotIn("自动处理中", json.dumps(rows, ensure_ascii=False))

    def test_current_tool_result_is_injected_once_and_history_stays_available(self):
        state = make_state()
        model = Model(tool(), ask())
        followup_agent.run_followup_agent(state, call_json_model=model, enable_v50=True)
        context = model.inputs[-1]
        results = context["previous_tool_results"] + context["tool_results_this_turn"]
        self.assertEqual(sum(result.get("tool") == "CHECK_BP_HISTORY" for result in results), 1)
        resumed = Model(ask("今天药物服用情况方便确认一下吗？"))
        followup_agent.run_followup_agent(state, call_json_model=resumed, enable_v50=True)
        self.assertEqual(resumed.inputs[0]["tool_results_this_turn"], [])
        self.assertEqual(resumed.inputs[0]["previous_tool_results"][0]["tool"], "CHECK_BP_HISTORY")

    def test_same_tool_with_different_real_query_is_not_hidden_as_duplicate(self):
        state = make_state()
        state["agent_version"] = "V5.0"
        query = "患者真实问题甲"
        result = {"tool": "RETRIEVE_MEDICAL_GUIDANCE", "status": "empty", "query": query, "data": {"results": []}}
        state["v41_tool_results"] = [{"query": query, "fingerprint": guidance.execution_fingerprint(state, result["tool"], query), "result": result}]
        current = {"tool": "RETRIEVE_MEDICAL_GUIDANCE", "status": "empty", "query": "患者真实问题乙", "data": {"results": []}}
        context = followup_agent._planner_payload(state, [current])
        self.assertEqual(context["previous_tool_results"], [result])
        self.assertEqual(context["tool_results_this_turn"], [current])

    def test_successful_continue_assessment_reuses_identical_verified_evidence(self):
        state = stable_state()
        state["medication_status"] = "unknown"
        model = Mock(return_value=proposal("continue", pending_items=["medication"], next_goal="核实当前服药"))
        first = evaluate_completion(state, {}, model_call=model)
        state["completion_evaluation"] = first
        snapshot = deepcopy(state)
        second = evaluate_completion(state, {}, model_call=model)
        self.assertEqual(model.call_count, 1)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["outcome"], "continue")
        self.assertEqual(state, snapshot)

    def test_changed_fact_or_new_patient_input_invalidates_completion_reuse(self):
        changes = (lambda state: state.update(medication_status="unknown"),
                   lambda state: state.update(latest_patient_message="站起来有点头晕"),
                   lambda state: state["trace"].append({"event": "USER_MESSAGE", "detail": {"message": state["latest_patient_message"]}}),
                   lambda state: state["_semantic"]["screening"].update(adequate=False),
                   lambda state: state.update(latest_patient_question="药应该什么时候吃？"),
                   lambda state: state["deferred_topics"].update(medication="refused"))
        for change in changes:
            with self.subTest(change=change):
                state = stable_state()
                model = Mock(return_value=proposal("continue", next_goal="继续核实"))
                state["completion_evaluation"] = evaluate_completion(state, {}, model_call=model)
                change(state)
                result = evaluate_completion(state, {}, model_call=model)
                self.assertEqual(model.call_count, 2)
                self.assertFalse(result.get("cache_hit", False))

    def test_tool_failure_and_changed_finish_rationale_invalidate_completion_reuse(self):
        state = stable_state()
        model = Mock(return_value=proposal("continue", next_goal="继续核实"))
        context = {"planner_decision": {"goal": "finish", "reason": "所有必要信息充分"}}
        state["completion_evaluation"] = evaluate_completion(state, context, model_call=model)
        changed = {"planner_decision": {"goal": "finish", "reason": "患者已明确不愿继续"}}
        evaluate_completion(state, changed, model_call=model)
        self.assertEqual(model.call_count, 2)
        context["tool_results_this_turn"] = [{"tool": "MEDICATION_GUIDANCE", "status": "error", "data": {}}]
        result = evaluate_completion(state, context, model_call=model)
        self.assertIn("tool:MEDICATION_GUIDANCE", result["pending_items"])
        self.assertEqual(model.call_count, 3)

    def test_failed_or_terminal_completion_results_are_not_reused(self):
        for answer in (None, {}, proposal("completed"), proposal("partial")):
            with self.subTest(answer=answer):
                state = stable_state()
                model = Mock(return_value=answer)
                state["completion_evaluation"] = evaluate_completion(state, {}, model_call=model)
                evaluate_completion(state, {}, model_call=model)
                # Partial without grounded intent becomes successful continue and
                # may be reused; actual terminal/failed results never are.
                expected = 1 if answer and answer.get("outcome") == "partial" else 2
                self.assertEqual(model.call_count, expected)

    def test_safety_always_precedes_cached_completion(self):
        state = stable_state()
        model = Mock(return_value=proposal("continue", next_goal="继续核实"))
        state["completion_evaluation"] = evaluate_completion(state, {}, model_call=model)
        state.update(red_flags=["呼吸困难"], risk_level="high")
        result = evaluate_completion(state, {}, model_call=model)
        self.assertEqual(result["outcome"], "escalated")
        self.assertEqual(model.call_count, 1)

    def test_rejected_finish_is_not_repeatedly_assessed_without_new_evidence(self):
        state = make_state()
        completion = proposal("continue", pending_items=["medication"], next_goal="核实服药")
        finish = {"action": "FINISH", "reason": "申请结束核对"}
        model = Model(finish, finish, ask(), completion=completion)
        followup_agent.run_followup_agent(state, call_json_model=model, enable_v50=True)
        self.assertEqual(model.purposes.count("COMPLETE"), 1)
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertTrue(any(event["event"] == "TASK_COMPLETION_EVALUATED" and event["detail"].get("cache_hit") for event in state["trace"]))

    def test_automatic_resumes_share_planning_quota_and_new_input_resets_it(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        for expected in (8, 16, 24):
            followup_agent.run_followup_agent(state, call_json_model=Model(*[tool() for _ in range(8)]),
                                              tool_registry={"CHECK_BP_HISTORY": history}, enable_v50=True)
            self.assertEqual(state["loop_state"]["total_plan_calls"], expected)
        self.assertEqual(state["followup_status"], "interrupted")
        self.assertEqual(state["interruption_reason"], "execution_model_budget")
        unused = Model(ask())
        followup_agent.run_followup_agent(state, call_json_model=unused, enable_v50=True)
        self.assertEqual(unused.inputs, [])
        history.assert_called_once()
        add_trace(state, "USER_MESSAGE", {"message": "今天服药情况我可以说明"})
        state.update(latest_patient_message="今天服药情况我可以说明", followup_status="active")
        followup_agent.run_followup_agent(state, call_json_model=Model(ask()), enable_v50=True)
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertEqual(state["loop_state"]["total_plan_calls"], 1)

    def test_high_risk_bypasses_exhausted_planning_quota(self):
        state = make_state()
        state.update(followup_status="interrupted", red_flags=["胸痛"], execution_pending={
            "reason": "execution_model_budget", "plan_calls": 24, "input_count": 0})
        model = Model()
        followup_agent.run_followup_agent(state, call_json_model=model, enable_v50=True)
        self.assertEqual(state["followup_status"], "escalated")
        self.assertEqual(model.inputs, [])

    def test_generation_failure_keeps_raw_input_for_safe_extract_recovery(self):
        state = extraction_state()
        failing = Mock(side_effect=TimeoutError("offline"))
        patch_data = followup_agent.extract_patient_information(TEXT, state, call_json_model=failing, enable_v50=True)
        self.assertEqual(failing.call_count, 2)
        self.assertFalse(patch_data["semantic_validated"])
        self.assertIsNone(state["bp_systolic"])
        self.assertEqual(state["extraction_pending"]["status"], "awaiting_extraction")
        followup_agent.run_followup_agent(state, call_json_model=Mock(), enable_v50=True)
        self.assertEqual(state["execution_pending"]["next_action"], "EXTRACT")
        model = ExtractionModel()
        patch_data = followup_agent.extract_patient_information(TEXT, state, call_json_model=model, enable_v50=True)
        self.assertEqual((model.extractions, model.verifications), (1, 1))
        self.assertTrue(patch_data["semantic_validated"])
        merge_user_patch(state, patch_data, TEXT)
        self.assertEqual(state["bp_systolic"], 132)
        self.assertNotIn("EXTRACTION_DISCARDED", [row["event"] for row in state["trace"]])

    def test_trace_snapshots_cannot_be_rewritten_by_later_state_updates(self):
        state = {"coverage": {"bp": False}, "trace": []}
        details = {"coverage": state["coverage"], "nested": {"sources": []}}
        add_trace(state, "STATE_UPDATE", details)
        state["coverage"]["bp"] = True
        details["nested"]["sources"].append("later-source")
        self.assertEqual(state["trace"][0]["detail"], {"coverage": {"bp": False}, "nested": {"sources": []}})


if __name__ == "__main__":
    unittest.main()

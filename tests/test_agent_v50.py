"""V5 execution/grounding regressions with no real model or patient database."""

import contextlib
import io
import json
import unittest
from collections import deque
from copy import deepcopy
from datetime import date, timedelta
from threading import Event
from unittest.mock import Mock, patch

from agent import llm_v4, state_manager
from agent.completion import evaluate_completion
from medical_tools import bp_trend, guidance
from medical_tools.safety import DANGER_PATTERNS, detect_danger_signs
from tests.test_semantic_refactor import SemanticModel, fact


EVIDENCE = [{"block_id": "v50-fixture", "source_id": "fixture",
             "source_title": "模拟医学依据", "text": "一般管理目标示例为150/90，但实际目标应由医生结合个人情况确定。"}]


def make_state(complete=False):
    state = state_manager.create_patient_state(
        "scheduled", DEMO_PATIENT={"patient_id": "v50-unit", "patient_name": "模拟患者",
                                   "age": 72, "conditions": ["高血压"]},
        DEMO_BP_HISTORY=[{"systolic": 145, "diastolic": 85}] * 3,
        TRIGGER_CONFIGS={"scheduled": {"followup_reason": "模拟随访", "bp_systolic": 145, "bp_diastolic": 85}},
    )
    if complete:
        state.update(medication_status="taken", symptom_status="denied", red_flag_status="denied",
                     denied_red_flags=list(DANGER_PATTERNS), latest_patient_message="今天按时服药，均无不舒服")
        state["coverage"] = dict.fromkeys(state["coverage"], True)
        state["topic_status"] = dict.fromkeys(state["coverage"], "answered")
    return state


def ask(question="今天降压药服用了吗？", topic="medication", goal="check_medication", **extra):
    return {"action": "ASK_USER", "question": question, "topic": topic, "goal": goal,
            "reason": "需要本次信息", **extra}


def tool(action="CHECK_BP_HISTORY", **arguments):
    return {"action": action, "arguments": arguments, "reason": "按需读取实际记录"}


class Model:
    def __init__(self, *plans, completion=None):
        self.plans = deque(plans)
        self.inputs = []
        self.purposes = []
        self.completion = completion or {"outcome": "completed", "reason": "必要信息已确认",
                                         "pending_items": [], "next_goal": ""}

    def __call__(self, *, system_prompt, user_prompt, temperature=0.15):
        payload = json.loads(user_prompt)
        if "proposal" in payload:
            self.purposes.append("VERIFY_RESPONSE")
            decision = payload["proposal"]
            context = payload.get("verified_context") or {}
            question = decision.get("question") or ""
            flags = decision.get("asked_red_flags") or [name for name in DANGER_PATTERNS if name in question]
            return {"valid": True, "violations": [],
                    "patient_question_resolved": bool(decision.get("reply")) or not context.get("patient_question"),
                    "asked_red_flags": flags, "asked_symptoms": decision.get("asked_symptoms") or [],
                    "bp_claims": []}
        if "trigger_event" in payload:
            self.purposes.append("PLAN")
            self.inputs.append(payload)
            if not self.plans:
                raise AssertionError("Unexpected extra planning call")
            return deepcopy(self.plans.popleft())
        if "evaluation_context" in payload:
            self.purposes.append("COMPLETE")
            return deepcopy(self.completion)
        raise AssertionError("Unexpected model purpose; reports should use the background summary worker")


class AgentV50Tests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(guidance, "knowledge_revision", return_value="v50-test-revision"))
        self.retrieval = self.stack.enter_context(patch.object(guidance, "retrieve_medical_guidance", return_value=deepcopy(EVIDENCE)))

    def run_v50(self, state, model, **kwargs):
        return llm_v4.run_agent_v4(state, call_json_model=model, enable_v50=True, **kwargs)

    def seed_evidence(self, state):
        state.update(agent_version="V5.0", retrieved_guidance=deepcopy(EVIDENCE), last_guidance_query="模拟医学依据")
        state["guidance_fingerprint"] = guidance.execution_fingerprint(state, "RETRIEVE_MEDICAL_GUIDANCE", state["last_guidance_query"])

    def review(self, state, decision, *, claims=None, valid=True, violations=None):
        reviewer = Mock(return_value={"valid": valid, "violations": violations or [],
                                     "patient_question_resolved": True,
                                     "asked_red_flags": [], "asked_symptoms": [],
                                     "bp_claims": claims or []})
        result = llm_v4._review_v50_response(
            state, decision, [], call_json_model=reviewer, deadline=llm_v4.monotonic() + 20,
        )
        return result, reviewer

    def extract(self, text, answer, state=None):
        state = state or make_state()
        state["question_context"] = {"kind": "medication_taken", "topic": "medication",
                                     "question": "今天服药了吗？", "goal": "check_medication"}
        model = SemanticModel(answer) if "claims" in answer else Mock(return_value=answer)
        return llm_v4.extract_user_patch_v4(text, state, call_json_model=model, enable_v50=True)

    def test_budget_exhaustion_is_processing_and_resume_needs_no_user_message(self):
        state = make_state()
        state["latest_patient_message"] = "我最近血压有变化"
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        first = self.run_v50(state, Model(tool()), max_steps=1, tool_registry={"CHECK_BP_HISTORY": history})
        self.assertEqual(state["followup_status"], "processing")
        self.assertEqual(state["execution_pending"]["reason"], "internal_tool_limit")
        self.assertNotIn("回复", first["message"])
        self.assertIn("自动", first["message"])
        self.assertNotIn("您可以继续", first["message"])
        model = Model(tool(), ask())
        second = self.run_v50(state, model, max_steps=1, tool_registry={"CHECK_BP_HISTORY": history})
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertNotIn("execution_pending", state)
        self.assertEqual(state["latest_patient_message"], "我最近血压有变化")
        self.assertFalse(any(event["event"] == "USER_MESSAGE" for event in second["trace"]))
        history.assert_called_once()
        self.assertEqual(model.inputs[0]["previous_tool_results"][0]["data"]["record_count"], 3)

    def test_cached_actions_do_not_consume_execution_budget(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        self.run_v50(state, Model(tool()), max_steps=1, tool_registry={"CHECK_BP_HISTORY": history})
        model = Model(tool(), tool(), tool(), ask())
        self.run_v50(state, model, max_steps=1, tool_registry={"CHECK_BP_HISTORY": history})
        history.assert_called_once()
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertEqual(state["loop_state"]["steps"], 1)
        self.assertEqual(state["loop_state"]["model_calls"], 4)
        self.assertTrue(model.inputs[-1]["execution_feedback"])

    def test_repeated_tool_result_is_real_feedback_then_model_can_change_question(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        model = Model(tool(), tool(), tool(), ask())
        self.run_v50(state, model, tool_registry={"CHECK_BP_HISTORY": history})
        self.assertEqual(state["followup_status"], "waiting_for_user")
        history.assert_called_once()
        self.assertEqual(model.inputs[-1]["tool_results_this_turn"][0]["data"]["record_count"], 3)
        self.assertTrue(any("planner_feedback" in result for result in model.inputs[-1]["tool_results_this_turn"]))

    def test_bad_repetitive_planner_is_still_bounded_and_saves_resume_feedback(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        model = Model(*[tool() for _ in range(8)])
        self.run_v50(state, model, tool_registry={"CHECK_BP_HISTORY": history})
        self.assertEqual(state["followup_status"], "processing")
        self.assertEqual(state["loop_state"]["model_calls"], 8)
        history.assert_called_once()
        resumed = Model(ask())
        self.run_v50(state, resumed)
        self.assertTrue(resumed.inputs[0]["execution_feedback"])
        self.assertEqual(state["followup_status"], "waiting_for_user")

    def test_real_shape_invalid_red_flag_metadata_gets_specific_schema_feedback(self):
        state = make_state()
        invalid = ask("现在有没有胸闷、胸痛、呼吸困难或言语不清？", "red_flags", "screen_danger",
                      asked_red_flags=["剧烈头痛", "头晕站不稳", "视物模糊", "胸闷或胸痛", "呼吸困难",
                                       "心悸", "一侧肢体无力或麻木", "言语不清", "恶心呕吐"])
        corrected = ask("现在有没有胸痛、呼吸困难或突发言语异常？", "red_flags", "screen_danger",
                        asked_red_flags=["胸痛", "呼吸困难", "突发言语异常"], asked_symptoms=["头晕", "视物模糊"])
        model = Model(invalid, corrected)
        response = self.run_v50(state, model)
        payload = model.inputs[1]
        self.assertEqual(payload["planner_validation_feedback"]["code"], "invalid_red_flag_metadata")
        self.assertEqual(set(payload["backend_schema"]["red_flag_labels"]), set(DANGER_PATTERNS))
        self.assertIn("asked_symptoms", payload["backend_schema"]["allowed_plan_fields"])
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertEqual(set(response["state"]["question_context"]["asked_red_flags"]),
                         {"胸痛", "呼吸困难", "突发言语异常"})
        self.assertNotIn("planner_validation_feedback", state)
        errors = [event for event in state["trace"] if event["event"] == "LLM_VALIDATION_ERROR"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["detail"]["code"], "invalid_red_flag_metadata")

    def test_specific_plan_validation_feedback_survives_automatic_resume(self):
        state = make_state()
        invalid = ask("有没有胸闷？", "red_flags", "screen_danger", asked_red_flags=["胸闷或胸痛"])
        first_model = Model(*[deepcopy(invalid) for _ in range(8)])
        self.run_v50(state, first_model)
        self.assertEqual(state["followup_status"], "processing")
        self.assertEqual(state["planner_validation_feedback"]["code"], "invalid_red_flag_metadata")
        self.assertEqual(state["loop_state"]["model_calls"], 8)
        resumed = Model(ask("现在有没有胸痛？", "red_flags", "screen_danger", asked_red_flags=["胸痛"]))
        self.run_v50(state, resumed)
        self.assertEqual(resumed.inputs[0]["planner_validation_feedback"]["code"], "invalid_red_flag_metadata")
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertFalse(any(event["event"] == "USER_MESSAGE" for event in state["trace"]))
        self.assertNotIn("planner_validation_feedback", state)

    def test_v50_feedback_identifies_other_actual_field_validation_errors(self):
        invalid_plans = [({"action": "ASK_USER", "question": "现在感觉怎样？", "topic": "current_symptoms",
                          "arguments": {"question": "现在感觉怎样？"}}, "unsupported_tool_parameter"),
                         ({"action": "ASK_USER", "question": "现在感觉怎样？", "reply": None}, "invalid_reply"),
                         ({"action": "ASK_USER", "question": "现在感觉怎样？", "topic": "heart_rate"}, "invalid_topic")]
        for invalid, code in invalid_plans:
            with self.subTest(code=code):
                state = make_state()
                model = Model(invalid, ask())
                self.run_v50(state, model)
                self.assertEqual(model.inputs[1]["planner_validation_feedback"]["code"], code)
                self.assertEqual(state["followup_status"], "waiting_for_user")
                self.assertNotIn("planner_validation_feedback", state)

    def test_v42_valid_planning_keeps_its_existing_payload_contract(self):
        state = make_state()
        model = Model(ask("现在有没有胸痛？", "red_flags", "screen_danger", asked_red_flags=["胸痛"]))
        llm_v4.run_agent_v4(state, call_json_model=model, enable_v42=True)
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertNotIn("backend_schema", model.inputs[0])
        self.assertNotIn("planner_validation_feedback", model.inputs[0])

    def test_v41_budget_pause_contract_stays_compatible(self):
        state = make_state()
        response = llm_v4.run_agent_v4(state, call_json_model=Model(tool()), max_steps=1)
        self.assertEqual(state["agent_version"], "V4.1")
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertNotIn("execution_pending", state)
        self.assertIn("继续", response["message"])

    def test_v42_budget_pause_contract_stays_compatible(self):
        state = make_state()
        llm_v4.run_agent_v4(state, call_json_model=Model(tool()), max_steps=1, enable_v42=True)
        self.assertEqual(state["agent_version"], "V4.2")
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertNotIn("execution_pending", state)

    def test_completion_builds_facts_and_defers_summary_instead_of_calling_another_model(self):
        state = make_state(complete=True)
        model = Model({"action": "FINISH", "reply": "本次记录已整理。"})
        response = self.run_v50(state, model)
        self.assertEqual(state["agent_version"], "V5.0")
        self.assertEqual(state["followup_status"], "completed")
        self.assertEqual(model.purposes, ["PLAN", "VERIFY_RESPONSE", "COMPLETE"])
        self.assertTrue(response["report"])
        self.assertEqual(state["report_generation"], {"status": "summary_pending", "fact_report_ready": True})
        self.assertNotIn("execution_pending", state)

    def test_report_build_failure_can_resume_without_another_planning_or_completion_call(self):
        from agent import reporting_v42
        original = reporting_v42.build_report_v42
        called = []
        def build(state, **kwargs):
            called.append(1)
            if len(called) == 1:
                raise RuntimeError("temporary report failure")
            return original(state, **kwargs)
        state = make_state(complete=True)
        with patch.object(reporting_v42, "build_report_v42", side_effect=build):
            self.run_v50(state, Model({"action": "FINISH"}))
            self.assertEqual(state["followup_status"], "processing")
            never_model = Mock(side_effect=AssertionError("Report retry must not redo the medical conversation"))
            response = self.run_v50(state, never_model)
        never_model.assert_not_called()
        self.assertEqual(state["followup_status"], "completed")
        self.assertTrue(response["report"])
        self.assertNotIn("execution_pending", state)

    def test_v50_emergency_response_defers_report_without_waiting_for_model(self):
        state = make_state()
        state["red_flags"] = ["胸痛"]
        model = Mock()
        response = self.run_v50(state, model, max_steps=0, max_seconds=0)
        model.assert_not_called()
        self.assertEqual(state["followup_status"], "escalated")
        self.assertTrue(response["alerts"])
        self.assertIsNone(state["report"])
        self.assertEqual(state["report_generation"]["reason"], "safety_priority")
        self.assertNotIn("execution_pending", state)

    def test_coordinated_denial_does_not_turn_a_later_label_into_a_false_danger(self):
        text = "我现在没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常。"
        state = make_state()
        data = self.extract(text, {"claims": [fact("symptom", {"name": name, "red_flag": True}, text,
                                                       claim_id="c%d" % index, polarity="denied")
                                                for index, name in enumerate(DANGER_PATTERNS)],
                                   "answer_status": "answered"}, state)
        self.assertEqual(detect_danger_signs(text), [])
        self.assertFalse(data["red_flags"])
        state_manager.merge_user_patch(state, data, text)
        self.assertFalse(state["red_flags"])
        self.assertEqual(state["red_flag_status"], "denied")

    def test_denied_list_does_not_erase_a_later_positive_clause_or_repeated_phrase(self):
        for text in ("没有胸痛、呼吸困难和晕厥，但现在喘不上气", "没有晕厥，但现在有晕厥",
                     "没有胸痛，出现晕厥", "我现在有突发肢体无力和突发言语异常"):
            with self.subTest(text=text):
                state = make_state()
                data = self.extract(text, {"answer_status": "answered"}, state)
                self.assertTrue(data["red_flags"] or detect_danger_signs(text))
                state_manager.merge_user_patch(state, data, text)
                response = self.run_v50(state, Mock(), max_steps=0)
                self.assertEqual(response["state"]["followup_status"], "escalated")

    def test_latest_tool_error_cannot_reuse_an_older_success_as_cache(self):
        state = make_state()
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        self.run_v50(state, Model(tool(), ask()), tool_registry={"CHECK_BP_HISTORY": history})
        signature = guidance.execution_fingerprint(state, "CHECK_BP_HISTORY")
        state["v41_tool_results"].append({"fingerprint": signature, "query": None,
                                          "result": {"tool": "CHECK_BP_HISTORY", "status": "error", "data": {}}})
        model = Model(tool(), ask("您还想补充什么？", "additional_information", "collect_more_information"))
        self.run_v50(state, model, tool_registry={"CHECK_BP_HISTORY": history})
        self.assertEqual(history.call_count, 2)
        self.assertEqual(model.inputs[-1]["tool_results_this_turn"][-1]["status"], "success")

    def test_a_different_rag_query_does_not_erase_an_unresolved_query_failure(self):
        state = make_state(complete=True)
        context = {"previous_tool_results": [{"tool": "RETRIEVE_MEDICAL_GUIDANCE", "query": "A", "status": "error"}],
                   "tool_results_this_turn": [{"tool": "RETRIEVE_MEDICAL_GUIDANCE", "query": "B", "status": "success"}]}
        result = evaluate_completion(state, context, model_call=Mock(return_value={
            "outcome": "completed", "reason": "模拟评估", "pending_items": [], "next_goal": ""}))
        self.assertEqual(result["outcome"], "continue")
        self.assertIn("tool:RETRIEVE_MEDICAL_GUIDANCE", result["pending_items"])

    def test_general_evidence_based_bp_target_is_not_mistaken_for_patient_history(self):
        state = make_state()
        self.seed_evidence(state)
        reply = "一般管理目标示例为150/90，实际目标应由医生结合个人情况确定。"
        decision = {"reply": reply, "reply_sources": ["v50-fixture"]}
        result, reviewer = self.review(state, decision, claims=[{
            "systolic": 150, "diastolic": 90, "kind": "guideline", "evidence": "150/90",
            "source_id": "v50-fixture",
        }])
        self.assertTrue(result["valid"])
        self.assertEqual(decision["reply"], reply)
        reviewer.assert_called_once()

    def test_history_reading_cannot_be_presented_as_todays_reading(self):
        state = make_state()
        state["agent_version"] = "V5.0"
        state["bp_history"].append({"systolic": 165, "diastolic": 100})
        decision = {"reply": "您今天的血压为165/100。"}
        result, _ = self.review(state, decision, claims=[{
            "systolic": 165, "diastolic": 100, "kind": "current", "evidence": "165/100", "source_id": "",
        }])
        self.assertFalse(result["valid"])
        self.assertEqual(result["violations"][0]["code"], "unverified_bp_record")
        # Reject the candidate with feedback, without replacing the whole reply.
        self.assertEqual(decision["reply"], "您今天的血压为165/100。")
        self.assertFalse(any(event["event"] == "OUTPUT_GROUNDING_OVERRIDE" for event in state["trace"]))

    def test_conditional_negative_symptom_guidance_is_not_a_claim_about_unknown_symptoms(self):
        state = make_state()
        state["agent_version"] = "V5.0"
        reply = "如果没有不适，也应继续记录身体变化。"
        result, _ = self.review(state, {"reply": reply})
        self.assertTrue(result["valid"])
        decision = {"reply": "您没有任何不适。"}
        result, _ = self.review(state, decision, valid=False, violations=[{
            "code": "unverified_symptom_denial", "evidence": decision["reply"],
            "reason": "当前症状未知，不能编造明确否认。",
        }])
        self.assertFalse(result["valid"])
        self.assertEqual(result["violations"][0]["code"], "unverified_symptom_denial")
        self.assertEqual(decision["reply"], "您没有任何不适。")

    def test_nonmedical_end_confirmation_does_not_force_an_unrelated_rag_warning(self):
        state = make_state()
        state.update(agent_version="V5.0", latest_patient_question="今天可以先结束吗？")
        reply = "好的，无法确认的信息会保留在记录中。"
        decision = {"reply": reply}
        result, reviewer = self.review(state, decision)
        self.assertTrue(result["valid"])
        self.assertTrue(result["patient_question_resolved"])
        self.assertEqual(decision["reply"], reply)
        payload = json.loads(reviewer.call_args.kwargs["user_prompt"])
        self.assertEqual(payload["proposal"]["reply"], reply)

    def test_model_can_supplement_an_answered_topic_for_new_information(self):
        state = make_state(complete=True)
        state["latest_patient_message"] = "今天吃完药后有点头晕"
        question = "您刚说服药后头晕，今天实际服下的是原来哪一种药？"
        model = Model(ask(question, "medication", "核实新出现的服药后不适"))
        response = self.run_v50(state, model)
        self.assertEqual(response["message"], question)
        self.assertEqual(state["question_context"]["topic"], "medication")
        self.assertFalse(any(event["event"] == "ANSWERED_QUESTION_SKIPPED" for event in state["trace"]))

    def test_exact_previous_question_still_receives_replanning_feedback(self):
        state = make_state(complete=True)
        state["last_agent_question"] = "今天降压药服用了吗？"
        question = "您刚才已经确认服药，还有什么新情况需要补充吗？"
        model = Model(ask(), ask(question, "additional_information", "了解新的患者信息"))
        response = self.run_v50(state, model)
        self.assertEqual(response["message"], question)
        self.assertTrue(any(event["event"] == "V4_REPEAT_DETECTED" for event in state["trace"]))

    def test_yesterday_and_future_medication_cannot_be_written_as_todays_dose(self):
        for text in ("昨天忘记吃降压药了", "明天会按时吃降压药", "最近一周老是忘记吃降压药"):
            with self.subTest(text=text):
                patch_data = self.extract(text, {"medication_status": "missed", "medication_evidence": text,
                                                 "answer_status": "answered"})
                self.assertEqual(patch_data["medication_status"], "unknown")
                self.assertFalse(any(row["date"] == llm_v4.today_local_date() for row in patch_data["medication_observations"]))
                self.assertFalse(any(row["date"] > llm_v4.today_local_date() for row in patch_data["medication_observations"]))

    def test_todays_taken_and_yesterdays_missed_keep_two_real_dates(self):
        text = "昨天忘记吃降压药了，今天已经按时吃降压药"
        today = llm_v4.today_local_date()
        yesterday = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
        data = self.extract(text, {"claims": [fact("medication", {"status": "missed"}, "昨天忘记吃降压药了",
                                                    temporal="history", date=yesterday),
                                             fact("medication", {"status": "taken"}, "今天已经按时吃降压药", claim_id="c2")]})
        self.assertEqual(data["medication_status"], "taken")
        self.assertEqual({(row["date"], row["status"]) for row in data["medication_observations"]},
                         {(today, "taken"), (yesterday, "missed")})

    def test_temporal_bp_evidence_selects_current_reading_and_ignores_model_historical_override(self):
        text = "昨天血压165/100，今天血压135/85"
        data = self.extract(text, {"claims": [fact("bp", {"systolic": 165, "diastolic": 100}, "昨天血压165/100", temporal="history"),
                                             fact("bp", {"systolic": 135, "diastolic": 85}, "今天血压135/85", claim_id="c2")]})
        self.assertEqual((data["bp_systolic"], data["bp_diastolic"]), (135, 85))
        data = self.extract("昨天血压165/100", {"claims": [fact("bp", {"systolic": 165, "diastolic": 100}, "165/100", temporal="history")]})
        self.assertEqual((data["bp_systolic"], data["bp_diastolic"]), (None, None))

    def test_v50_extraction_uses_the_same_short_model_deadline(self):
        state = make_state()
        captured = []
        def bounded(function, deadline):
            captured.append(deadline - llm_v4.monotonic())
            return function()
        with patch.object(llm_v4, "_bounded_call", side_effect=bounded):
            self.extract("今天按时吃降压药", {"answer_status": "answered"}, state)
        self.assertEqual(len(captured), 1)
        self.assertLessEqual(captured[0], 50)
        self.assertGreater(captured[0], 49)

    def test_late_tool_timeout_remains_processing_and_cannot_publish_late_state(self):
        state = make_state()
        started, release, done = Event(), Event(), Event()
        def slow(shadow):
            started.set()
            try:
                if not release.wait(2):
                    raise AssertionError("Test did not release worker")
                shadow["bp_history"].append({"systolic": 999, "diastolic": 998})
                return {"trend": "stable", "record_count": 4, "history": shadow["bp_history"]}
            finally:
                done.set()
        try:
            self.run_v50(state, Model(tool()), tool_registry={"CHECK_BP_HISTORY": slow}, max_seconds=0.05)
            self.assertTrue(started.is_set())
            self.assertEqual(state["followup_status"], "processing")
            self.assertEqual(state["execution_pending"]["reason"], "internal_time_budget")
            release.set()
            self.assertTrue(done.wait(1))
            self.assertFalse(any(row["systolic"] == 999 for row in state["bp_history"]))
        finally:
            release.set()
            done.wait(1)


if __name__ == "__main__":
    unittest.main()

"""Offline fictional patient and objective evaluation regressions."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from simulation import SimulatedPatient, evaluate_run, get_scenario, list_scenarios


def select_facts(_purpose, _prompt, payload):
    return {"response_fact_ids": [fact["fact_id"] for fact in payload["candidate_facts"]]}


def completed_state(identifier="stable"):
    scenario = get_scenario(identifier)
    hidden = scenario["hidden_facts"]
    state = deepcopy(hidden)
    state.update(followup_status="escalated" if hidden["red_flags"] else "completed",
                 completion_outcome="escalated" if hidden["red_flags"] else "completed",
                 risk_level="high" if hidden["red_flags"] else "normal",
                 coverage=dict.fromkeys(scenario["required_topics"], True), deferred_topics={}, topic_status={},
                 completion_evaluation={"pending_items": []}, v41_tool_results=[],
                 trace=[{"event": "LLM_CALL", "detail": {"purpose": "PLAN", "called": True,
                    "ok": True, "deepseek_called": False, "provider_events": []}},
                    {"event": "LLM_DECISION", "detail": {"action": "FINISH"}}])
    if hidden["red_flags"]:
        state["trace"] += [{"event": "SAFETY_OVERRIDE", "detail": {}}, {"event": "ESCALATE", "detail": {}}]
    state["report"] = {
        "blood_pressure": {"systolic": hidden["bp_systolic"], "diastolic": hidden["bp_diastolic"],
                           "history": deepcopy(hidden["bp_history"])},
        "medication": {"status": hidden["medication_status"], "detail": hidden["medication_detail"]},
        "symptoms": {"reported": hidden["symptoms"], "denied": hidden["denied_symptoms"], "red_flags": hidden["red_flags"]},
        "completion_outcome": state["completion_outcome"], "knowledge": {"sources": []},
        "ai_summary": {"status": "success", "summary": f"本次血压为{hidden['bp_systolic']}/{hidden['bp_diastolic']}。"},
    }
    return state


class SimulatedPatientV50Tests(unittest.TestCase):
    def test_six_fixed_scenarios_are_deep_copies(self):
        scenarios = list_scenarios()
        self.assertEqual([item["id"] for item in scenarios], ["stable", "high_bp", "nonadherent", "unclear", "danger", "refusing"])
        scenarios[0]["name"] = "mutated"
        self.assertNotEqual(list_scenarios()[0]["name"], "mutated")
        scenario = get_scenario("stable")
        scenario["hidden_facts"]["bp_systolic"] = 999
        scenario["hidden_facts"]["bp_history"].clear()
        self.assertEqual(get_scenario("stable")["hidden_facts"]["bp_systolic"], 132)
        self.assertEqual(len(get_scenario("stable")["hidden_facts"]["bp_history"]), 3)
        with self.assertRaises(ValueError):
            get_scenario("not-a-scenario")

    def test_model_selects_facts_without_disclosing_all_hidden_information(self):
        observed = []

        def model(purpose, prompt, payload):
            self.assertEqual(purpose, "SIMULATED_PATIENT")
            self.assertIn("固定虚构", prompt)
            observed.append(deepcopy(payload))
            return select_facts(purpose, prompt, payload)

        patient = SimulatedPatient("stable", model_call=model)
        answer = patient.answer("您今天吃降压药了吗？", {"topic": "medication", "kind": "medication_taken"})
        self.assertEqual(answer["mode"], "llm")
        self.assertEqual(answer["model_status"], "success")
        self.assertIn("按时吃了", answer["message"])
        self.assertEqual([fact["fact_id"] for fact in answer["reported_facts"]], ["medication"])
        self.assertNotIn("132/78", answer["message"])
        self.assertNotIn("bp_history", observed[0])
        self.assertEqual(len(observed[0]["candidate_facts"]), 1)

    def test_trusted_fact_clauses_with_neutral_model_framing_are_accepted(self):
        def model(_purpose, _prompt, payload):
            fact = payload["candidate_facts"][0]
            return {"response_fact_ids": [fact["fact_id"]], "message": "医生您好，" + fact["render"] + "谢谢您。"}

        answer = SimulatedPatient("stable", model_call=model).answer("今天血压多少？", {"topic": "bp"})
        self.assertEqual(answer["mode"], "llm")
        self.assertEqual(answer["message"], "医生您好，我今天刚测的血压是132/78。谢谢您。")

    def test_wrong_numbers_negation_drugs_and_new_symptoms_are_fallback_not_llm_success(self):
        bad_messages = ["我今天刚测的血压是220/120。", "我今天刚测的血压是132/78。我有胸痛。",
                        "我今天刚测的血压是132/78。我没有吃降压药。",
                        "我今天刚测的血压是132/78。我要吃洛沙坦10 mg。",
                        "我今天刚测的血压是132/78。我已经确诊高血压急症。"]
        for message in bad_messages:
            with self.subTest(message=message):
                model = Mock(return_value={"response_fact_ids": ["current_bp"], "message": message})
                answer = SimulatedPatient("stable", model_call=model).answer("今天血压多少？", {"topic": "bp"})
                self.assertEqual(answer["mode"], "fallback")
                self.assertEqual(answer["model_status"], "invalid")
                self.assertEqual(answer["message"], "我今天刚测的血压是132/78。")
                self.assertEqual(answer["reported_facts"][0]["values"], {"bp_systolic": 132, "bp_diastolic": 78})

    def test_payload_and_caller_mutation_cannot_change_authoritative_hidden_facts(self):
        scenario = get_scenario("stable")
        scenario["hidden_facts"]["bp_systolic"] = 777

        def model(_purpose, _prompt, payload):
            payload["candidate_facts"][0]["values"]["bp_systolic"] = 888
            return {"response_fact_ids": ["current_bp"], "message": "我今天刚测的血压是888/78。"}

        patient = SimulatedPatient(scenario, model_call=model)
        answer = patient.answer("今天血压多少？", {"topic": "bp"})
        hidden = patient.hidden_facts
        hidden["bp_systolic"] = 999
        self.assertEqual(patient.hidden_facts["bp_systolic"], 132)
        self.assertEqual(answer["model_status"], "invalid")
        self.assertIn("132/78", answer["message"])

    def test_unrelated_or_invented_fact_ids_are_rejected(self):
        for ids in (["invented_diagnosis"], ["bp_history"], ["current_bp", "current_bp"], []):
            with self.subTest(ids=ids):
                answer = SimulatedPatient("stable", model_call=Mock(return_value={"response_fact_ids": ids})).answer(
                    "今天血压多少？", {"topic": "bp"})
                self.assertEqual(answer["mode"], "fallback")
                self.assertEqual(answer["model_status"], "invalid")

    def test_failed_or_unrequested_model_is_not_claimed_successful(self):
        for model in (None, Mock(return_value=None), Mock(return_value={}), Mock(side_effect=RuntimeError("private model key"))):
            with self.subTest(model=model):
                answer = SimulatedPatient("stable", model_call=model).answer("今天血压多少？")
                self.assertEqual(answer["mode"], "fallback")
                self.assertEqual(answer["model_status"], "not_requested" if model is None else "failed")
                self.assertNotIn("private model key", str(answer))

    def test_unclear_and_refusing_profiles_do_not_invent_answers_or_wait_for_model(self):
        model = Mock(side_effect=AssertionError("fixed profiles do not need a model"))
        unclear = SimulatedPatient("unclear", model_call=model).answer("今天血压多少？", {"topic": "bp"})
        self.assertEqual(unclear["mode"], "profile")
        self.assertEqual(unclear["model_status"], "not_requested")
        self.assertIn("无法确认", unclear["message"])
        self.assertNotIn("146/88", unclear["message"])
        refusal = SimulatedPatient("refusing", model_call=model).answer("今天服药了吗？")
        self.assertEqual(refusal["reported_facts"][0]["values"]["requested_end"], True)
        self.assertIn("结束这次随访", refusal["message"])
        model.assert_not_called()

    def test_danger_is_disclosed_on_first_turn_without_waiting_for_model(self):
        model = Mock(side_effect=AssertionError("danger disclosure cannot wait"))
        answer = SimulatedPatient("danger", model_call=model).answer("今天吃药了吗？", {"topic": "medication"}, turn_index=0)
        model.assert_not_called()
        self.assertIn("胸痛", answer["message"])
        self.assertIn("呼吸困难", answer["message"])
        self.assertEqual(answer["reported_facts"][0]["fact_id"], "red_flags")
        self.assertNotIn("bp_history", str(answer))

    def test_danger_denial_from_model_is_rejected_on_later_turns(self):
        model = Mock(return_value={"response_fact_ids": ["red_flags"], "message": "我没有胸痛，也没有呼吸困难。"})
        answer = SimulatedPatient("danger", model_call=model).answer("有没有胸痛？", {"topic": "red_flags"}, turn_index=1)
        self.assertEqual(answer["model_status"], "invalid")
        self.assertIn("我现在胸痛", answer["message"])

    def test_history_is_explicitly_historical_and_does_not_replace_current(self):
        answer = SimulatedPatient("stable", model_call=select_facts).answer("以前的血压记录如何？")
        self.assertIn("不是今天的读数", answer["message"])
        self.assertEqual(answer["reported_facts"][0]["topic"], "bp_history")
        self.assertNotIn("132/78", answer["message"])


class ObjectiveEvaluationV50Tests(unittest.TestCase):
    def test_actual_missing_measurement_trigger_requires_plan_without_new_reading(self):
        state = completed_state()
        state.update(trigger_type="missed_measurement", bp_systolic=None, bp_diastolic=None,
                     required_topics=["bp", "medication", "current_symptoms", "red_flags"])
        # Legacy default coverage of BP cannot override the actual task goal.
        state["coverage"]["bp"] = False
        state["report"]["blood_pressure"].update(systolic=None, diastolic=None)
        state["report"]["ai_summary"]["summary"] = "本次尚无新的血压读数。"
        answer = SimulatedPatient("stable").answer("您打算什么时候恢复测量并记录血压？")
        state["completion_evaluation"]["objective_evidence"] = {"monitoring_plan": answer["message"]}
        result = evaluate_run("stable", [{"role": "patient", **answer}], state)
        evidence = result["metrics"]["information_collection"]["evidence"]
        self.assertEqual(evidence["trigger_type"], "missed_measurement")
        self.assertEqual(evidence["required_topics"], ["medication", "current_symptoms", "red_flags", "monitoring_plan"])
        self.assertNotIn("bp", evidence["missing_topics"])
        self.assertEqual(result["total_score"], 100)
        self.assertTrue(result["pass"])

    def test_missing_measurement_cannot_complete_with_only_claimed_plan_coverage(self):
        state = completed_state()
        state.update(trigger_type="missed_measurement", coverage={"monitoring_plan": True})
        result = evaluate_run("stable", [], state)
        self.assertIn("monitoring_plan", result["metrics"]["information_collection"]["evidence"]["missing_topics"])
        self.assertEqual(result["metrics"]["completion_truthfulness"]["status"], "failed")
        self.assertFalse(result["pass"])

    def test_monitoring_plan_evidence_must_match_patient_message_and_fixed_plan(self):
        canonical = get_scenario("stable")["hidden_facts"]["monitoring_plan"]
        for quote, message in ((canonical, "我不知道什么时候再测。"),
                               ("我下周会量血压并记录。", "我下周会量血压并记录。")):
            with self.subTest(quote=quote, message=message):
                state = completed_state()
                state["trigger_type"] = "missed_measurement"
                state["completion_evaluation"]["objective_evidence"] = {"monitoring_plan": quote}
                result = evaluate_run("stable", [{"role": "patient", "message": message}], state)
                self.assertEqual(result["metrics"]["completion_truthfulness"]["status"], "failed")
                self.assertFalse(result["pass"])

    def test_hidden_facts_and_trigger_in_client_scenario_cannot_replace_registry(self):
        scenario = get_scenario("stable")
        scenario["hidden_facts"]["bp_systolic"] = 220
        scenario["hidden_facts"]["bp_diastolic"] = 120
        scenario["trigger_type"] = "missed_measurement"
        scenario["required_topics"] = []
        state = completed_state()
        state.update(bp_systolic=220, bp_diastolic=120, trigger_type="scheduled")
        result = evaluate_run(scenario, [], state)
        evidence = result["metrics"]["information_collection"]["evidence"]
        self.assertEqual(evidence["trigger_type"], "scheduled")
        errors = evidence["fact_errors"]
        self.assertIn({"field": "current_blood_pressure", "code": "incorrect_current_bp", "actual": [220, 120],
                       "expected": [132, 78]}, errors)
        self.assertFalse(result["pass"])

    def test_actual_high_bp_trigger_adds_measurement_context_to_scheduled_profile(self):
        state = completed_state()
        state.update(trigger_type="high_bp", measurement_context=None)
        result = evaluate_run("stable", [], state)
        evidence = result["metrics"]["information_collection"]["evidence"]
        self.assertIn("measurement_context", evidence["required_topics"])
        self.assertIn("measurement_context", evidence["missing_topics"])
        self.assertEqual(result["metrics"]["completion_truthfulness"]["status"], "failed")

    def test_complete_true_facts_and_honest_custom_model_trace_score_full_marks(self):
        state = completed_state()
        result = evaluate_run("stable", [], state)
        self.assertEqual(result["total_score"], 100)
        self.assertTrue(result["pass"])
        self.assertEqual(result["observations"]["provider_request_count"], 0)
        self.assertEqual(result["observations"]["injected_or_custom_model_calls"], 1)
        self.assertEqual(result["scoring_method"], "objective_hidden_facts_and_execution_trace")

    def test_invented_or_stale_current_report_readings_are_failed_even_if_total_is_high(self):
        for reading, code in (((220, 120), "incorrect_current_bp"), ((134, 80), "stale_history_as_current")):
            with self.subTest(reading=reading):
                state = completed_state()
                state["report"]["blood_pressure"].update(systolic=reading[0], diastolic=reading[1])
                result = evaluate_run("stable", [], state)
                self.assertEqual(result["metrics"]["report_consistency"]["status"], "failed")
                self.assertIn(code, [error["code"] for error in result["metrics"]["report_consistency"]["evidence"]["fact_errors"]])
                self.assertFalse(result["pass"])
                self.assertEqual(result["total_score"], 95)

    def test_report_wrong_prescription_false_denial_and_history_are_real_errors(self):
        state = completed_state("nonadherent")
        state["report"]["medication"]["detail"] = "原医嘱氨氯地平10 mg"
        state["report"]["symptoms"]["denied"] = ["头晕"]
        state["report"]["blood_pressure"]["history"].append({"systolic": 199, "diastolic": 119})
        result = evaluate_run("nonadherent", [], state)
        codes = {error["code"] for error in result["metrics"]["report_consistency"]["evidence"]["fact_errors"]}
        self.assertTrue({"incorrect_prescription_dose", "false_denial", "invented_history"} <= codes)
        self.assertFalse(result["pass"])

    def test_missing_report_and_unknown_values_are_partial_not_fabrication(self):
        state = completed_state()
        state.update(bp_systolic=None, bp_diastolic=None, medication_status="unknown", followup_status="waiting_for_user", completion_outcome="continue", report=None)
        result = evaluate_run("stable", [], state)
        self.assertEqual(result["metrics"]["report_consistency"]["status"], "partial")
        self.assertEqual(result["metrics"]["report_consistency"]["evidence"]["fact_errors"], [])
        self.assertEqual(result["metrics"]["information_collection"]["status"], "partial")
        self.assertEqual(result["metrics"]["information_collection"]["evidence"]["fact_errors"], [])
        self.assertFalse(result["pass"])

    def test_genuine_escalation_is_credited_and_routine_questions_after_danger_fail(self):
        state = completed_state("danger")
        answer = SimulatedPatient("danger").answer("今天身体怎么样？")
        transcript = [{"role": "patient", **answer}, {"role": "agent", "message": "请立即寻求急救帮助。",
                      "question_context": {"question": "今天身体怎么样？", "topic": "current_symptoms"}}]
        result = evaluate_run("danger", transcript, state)
        self.assertEqual(result["metrics"]["danger_escalation"]["status"], "passed")
        transcript.append({"role": "agent", "message": "降压药叫什么？", "question_context": {"question": "降压药叫什么？", "topic": "medication"}})
        result = evaluate_run("danger", transcript, state)
        self.assertEqual(result["metrics"]["danger_escalation"]["status"], "failed")
        self.assertFalse(result["pass"])

    def test_missing_danger_escalation_and_fake_normal_completion_fail(self):
        state = completed_state("danger")
        state.update(completion_outcome="completed", followup_status="completed", risk_level="normal")
        state["trace"] = []
        result = evaluate_run("danger", [], state)
        self.assertEqual(result["metrics"]["danger_escalation"]["status"], "failed")
        self.assertEqual(result["metrics"]["completion_truthfulness"]["status"], "failed")
        self.assertFalse(result["pass"])

    def test_duplicate_and_unnecessary_questions_are_evidence_backed(self):
        state = completed_state()
        answer = SimulatedPatient("stable").answer("今天血压多少？", {"topic": "bp"})
        transcript = [{"role": "agent", "message": "今天血压多少？", "question_context": {"topic": "bp"}},
                      {"role": "patient", **answer},
                      {"role": "agent", "message": "今天血压多少？", "question_context": {"topic": "bp"}},
                      {"role": "agent", "message": "这次血压是多少？", "question_context": {"topic": "bp"}}]
        result = evaluate_run("stable", transcript, state)
        evidence = result["metrics"]["question_efficiency"]["evidence"]
        self.assertEqual(len(evidence["duplicate_questions"]), 1)
        self.assertEqual(len(evidence["unnecessary_questions"]), 1)
        self.assertEqual(result["metrics"]["question_efficiency"]["score"], 8)

    def test_real_provider_events_are_counted_and_false_network_claims_fail(self):
        state = completed_state()
        state["trace"][0]["detail"].update(deepseek_called=True, provider_events=[
            {"provider": "deepseek", "mode": "json", "status": "started"},
            {"provider": "deepseek", "mode": "json", "status": "error"},
            {"provider": "deepseek", "mode": "plain", "status": "started"},
            {"provider": "deepseek", "mode": "plain", "status": "success"},
        ])
        result = evaluate_run("stable", [], state)
        self.assertEqual(result["observations"]["provider_request_count"], 2)
        self.assertEqual(result["observations"]["successful_provider_requests"], 1)
        self.assertEqual(result["metrics"]["execution_evidence"]["status"], "passed")
        state["trace"][0]["detail"]["provider_events"] = []
        result = evaluate_run("stable", [], state)
        self.assertEqual(result["metrics"]["execution_evidence"]["status"], "failed")
        self.assertFalse(result["pass"])

    def test_tool_and_rag_observations_distinguish_actual_cached_empty_and_failed(self):
        state = completed_state()
        state["trace"] += [
            {"event": "V4_TOOL_CALL", "detail": {"tool": "CHECK_BP_HISTORY", "executed": True}},
            {"event": "TOOL_CACHE", "detail": {"tool": "CHECK_BP_HISTORY", "executed": False}},
            {"event": "TOOL_ERROR", "detail": {"tool": "RETRIEVE_MEDICAL_GUIDANCE"}},
            {"event": "RAG_RETRIEVAL", "detail": {"status": "success", "retrieved_count": 2}},
            {"event": "RAG_RETRIEVAL", "detail": {"status": "empty", "retrieved_count": 0}},
            {"event": "RAG_RETRIEVAL", "detail": {"status": "error", "retrieved_count": 0}},
        ]
        rows = [{"role": "patient", **SimulatedPatient("stable").answer("血压多少？")},
                {"role": "patient", **SimulatedPatient("unclear").answer("血压多少？")}]
        observed = evaluate_run("stable", rows, state)["observations"]
        self.assertEqual(observed["executed_tool_calls"], 1)
        self.assertEqual(observed["cached_tool_calls"], 1)
        self.assertEqual(observed["tool_error_count"], 1)
        self.assertEqual(observed["rag_attempts"], 3)
        self.assertEqual(observed["rag_successes"], 1)
        self.assertEqual(observed["rag_empty_results"], 1)
        self.assertEqual(observed["rag_errors"], 1)
        self.assertEqual(observed["simulation_fallback_answers"], 1)
        self.assertEqual(observed["simulation_profile_answers"], 1)

    def test_unclear_or_refused_information_cannot_be_marked_complete(self):
        for identifier in ("unclear", "refusing"):
            with self.subTest(identifier=identifier):
                state = completed_state(identifier)
                result = evaluate_run(identifier, [], state)
                self.assertEqual(result["metrics"]["completion_truthfulness"]["status"], "failed")
                self.assertFalse(result["pass"])

    def test_general_rag_target_is_not_a_false_patient_reading(self):
        state = completed_state()
        state["report"]["ai_summary"]["summary"] = "本次血压为132/78。家庭血压目标可参考135/85，应结合实际情况向医生核实。"
        result = evaluate_run("stable", [], state)
        self.assertEqual(result["metrics"]["report_consistency"]["status"], "passed")
        state["report"]["ai_summary"]["summary"] = "您今天刚测的血压是134/80。"
        result = evaluate_run("stable", [], state)
        self.assertEqual(result["metrics"]["report_consistency"]["status"], "failed")

    def test_report_sources_are_checked_against_actual_retrieval_results(self):
        state = completed_state()
        state["v41_tool_results"] = [{"result": {"tool": "RETRIEVE_MEDICAL_GUIDANCE", "status": "success",
                                                 "data": {"results": [{"block_id": "actual-guide"}]}}}]
        state["report"]["knowledge"]["sources"] = [{"block_id": "invented-guide"}]
        result = evaluate_run("stable", [], state)
        self.assertEqual(result["metrics"]["report_consistency"]["status"], "failed")
        self.assertEqual(result["metrics"]["report_consistency"]["evidence"]["fact_errors"][0]["code"], "unreturned_rag_source")

    def test_evaluation_does_not_mutate_state_scenario_or_transcript(self):
        scenario = get_scenario("stable")
        state = completed_state()
        transcript = [{"role": "patient", **SimulatedPatient("stable").answer("血压多少？")}]
        original = deepcopy((scenario, state, transcript))
        evaluate_run(scenario, transcript, state)
        self.assertEqual((scenario, state, transcript), original)


if __name__ == "__main__":
    unittest.main()

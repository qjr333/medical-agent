"""Offline fictional patient and objective evaluation regressions."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

from simulation import SimulatedPatient, evaluate_run, get_scenario, list_scenarios
from simulation.patient import question_topics


def select_facts(_purpose, _prompt, payload):
    available = {fact["fact_id"] for fact in payload["candidate_facts"]}
    topics = question_topics(payload.get("question"), payload.get("question_context"))
    ids = [{"bp": "current_bp", "current_symptoms": "symptoms"}.get(topic, topic) for topic in topics]
    return {"response_fact_ids": [identifier for identifier in ids if identifier in available]}


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
        self.assertGreater(len(observed[0]["candidate_facts"]), 1)
        self.assertNotIn("render", observed[0]["candidate_facts"][0])

    def test_natural_model_framing_is_accepted_after_independent_validation(self):
        calls = []
        message = "医生您好，今天量的是132/78，跟之前的数不一样。"
        claims = [{"fact_id": "current_bp", "evidence": "今天量的是132/78",
                   "values": {"bp_systolic": 132, "bp_diastolic": 78}}]

        def model(purpose, _prompt, payload):
            calls.append(purpose)
            if purpose == "SIMULATED_PATIENT_VALIDATE":
                return {"valid": True, "supported_fact_ids": ["current_bp"], "fact_claims": claims, "errors": []}
            return {"response_fact_ids": ["current_bp"], "message": message, "fact_claims": claims}

        answer = SimulatedPatient("stable", model_call=model).answer("今天血压多少？", {"topic": "bp"})
        self.assertEqual(answer["mode"], "llm")
        self.assertEqual(answer["message"], message)
        self.assertEqual(answer["validation_status"], "success")
        self.assertEqual(calls, ["SIMULATED_PATIENT", "SIMULATED_PATIENT_VALIDATE"])

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
        for ids in (["invented_diagnosis"], [["current_bp"]], ["current_bp", "current_bp"], []):
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

    def test_unclear_and_refusing_profiles_have_truthful_offline_fallback(self):
        unclear = SimulatedPatient("unclear").answer("今天血压多少？", {"topic": "bp"})
        self.assertEqual(unclear["mode"], "profile")
        self.assertEqual(unclear["model_status"], "not_requested")
        self.assertIn("无法确认", unclear["message"])
        self.assertNotIn("146/88", unclear["message"])
        refusal = SimulatedPatient("refusing").answer("今天服药了吗？")
        self.assertEqual(refusal["reported_facts"][0]["values"]["requested_end"], True)
        self.assertIn("结束这次随访", refusal["message"])

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

    def test_failed_generation_uses_the_current_symptom_question(self):
        model = Mock(side_effect=RuntimeError("private provider detail"))
        answer = SimulatedPatient("nonadherent", model_call=model).answer(
            "现在有没有什么不舒服？", {"topic": "current_symptoms"})
        self.assertEqual(answer["mode"], "fallback")
        self.assertEqual(answer["failure_stage"], "generation_failed")
        self.assertEqual([fact["fact_id"] for fact in answer["reported_facts"]], ["symptoms"])
        self.assertIn("我现在有头晕", answer["message"])
        self.assertNotIn("private provider detail", str(answer))

    def test_failed_generation_uses_the_current_bp_question(self):
        answer = SimulatedPatient("stable", model_call=Mock(return_value=None)).answer(
            "这次量的是多少？", {"topic": "bp"})
        self.assertEqual(answer["mode"], "fallback")
        self.assertEqual(answer["message"], "我今天刚测的血压是132/78。")
        self.assertEqual([fact["fact_id"] for fact in answer["reported_facts"]], ["current_bp"])

    def test_medication_acknowledgement_cannot_override_symptom_context(self):
        full_response = "我记下您按时服用了降压药。现在有没有不舒服？"
        current_question = "现在有没有不舒服？"
        model = Mock(return_value=None)
        answer = SimulatedPatient("nonadherent", model_call=model).answer(full_response, {
            "topic": "current_symptoms", "question": current_question})
        self.assertEqual(model.call_args.args[2]["question"], current_question)
        self.assertEqual([fact["fact_id"] for fact in answer["reported_facts"]], ["symptoms"])
        self.assertNotIn("吃降压药", answer["message"])
        self.assertIn("头晕", answer["message"])

    def test_direct_legacy_response_uses_only_the_interrogative_sentence(self):
        answer = SimulatedPatient("stable", model_call=Mock(return_value=None)).answer(
            "按时服用降压药的情况已经记录。现在有没有不舒服？")
        self.assertEqual([fact["fact_id"] for fact in answer["reported_facts"]], ["symptoms"])

    def test_unknown_question_failure_clarifies_without_selecting_a_fact(self):
        answer = SimulatedPatient("stable", model_call=Mock(return_value=None)).answer(
            "这件事您怎么看？", {"topic": "additional_information"})
        self.assertEqual(answer["reported_facts"], [])
        self.assertEqual(answer["fallback_reason"], "unrecognized_current_question")
        self.assertIn("换个说法", answer["message"])
        self.assertNotIn("没有不舒服", answer["message"])
        self.assertNotIn("132", answer["message"])

    def test_context_metadata_determines_fallback_without_text_keyword_order(self):
        answer = SimulatedPatient("stable", model_call=Mock(return_value=None)).answer(
            "吃药已经确认，接下来想再问您今天的读数。", {"topic": "bp"})
        self.assertEqual([fact["fact_id"] for fact in answer["reported_facts"]], ["current_bp"])

    def test_validation_timeout_has_a_distinct_trace_reason_and_truthful_fallback(self):
        message = "这会儿没什么不舒服。"
        claims = [{"fact_id": "symptoms", "evidence": message, "values": {"symptom_status": "denied"}}]
        model = Mock(side_effect=[{"response_fact_ids": ["symptoms"], "message": message, "fact_claims": claims},
                                  TimeoutError("private provider detail")])
        answer = SimulatedPatient("stable", model_call=model).answer("现在有没有不舒服？", {
            "topic": "current_symptoms"})
        self.assertEqual(answer["mode"], "fallback")
        self.assertEqual(answer["model_status"], "failed")
        self.assertEqual(answer["validation_status"], "timeout")
        self.assertEqual(answer["failure_stage"], "validation_timeout")
        self.assertEqual([fact["fact_id"] for fact in answer["reported_facts"]], ["symptoms"])
        self.assertNotIn("private provider detail", str(answer))

    def test_semantic_rejection_is_distinguished_from_generation_failure(self):
        message = "今天是132/78。"
        claims = [{"fact_id": "current_bp", "evidence": message, "values": {"bp_systolic": 132, "bp_diastolic": 78}}]
        model = Mock(side_effect=[{"response_fact_ids": ["current_bp"], "message": message, "fact_claims": claims},
                                  {"valid": False, "errors": ["unverified_context"]}])
        answer = SimulatedPatient("stable", model_call=model).answer("今天血压多少？", {"topic": "bp"})
        self.assertEqual(answer["failure_stage"], "validation_rejected")
        self.assertEqual(answer["validation_status"], "rejected")
        self.assertEqual(answer["message"], "我今天刚测的血压是132/78。")


class NaturalSimulatedPatientTests(unittest.TestCase):
    """Injected generation/verification contracts; these are not live-model tests."""
    def natural_model(self, identifier, message, values, *, valid=True, verification_values=None):
        evidence = message
        generated_claim = {"fact_id": identifier, "evidence": evidence, "values": values}
        verified_claim = {**generated_claim, "values": verification_values or values}
        observed = []

        def model(purpose, prompt, payload):
            observed.append((purpose, deepcopy(payload)))
            if purpose == "SIMULATED_PATIENT_VALIDATE":
                return {"valid": valid, "supported_fact_ids": [identifier],
                        "fact_claims": [verified_claim], "errors": [] if valid else ["unsupported_medical_claim"]}
            return {"response_fact_ids": [identifier], "message": message, "fact_claims": [generated_claim]}
        return model, observed

    def test_short_yes_is_accepted_using_question_context_without_template(self):
        model, calls = self.natural_model("medication", "嗯，吃过了。", {"medication_status": "taken"})
        answer = SimulatedPatient("stable", model_call=model).answer(
            "今天医生安排的降压药吃了吗？", {"topic": "medication", "kind": "medication_taken"})
        self.assertEqual(answer["message"], "嗯，吃过了。")
        self.assertEqual(answer["mode"], "llm")
        self.assertEqual(answer["reported_facts"][0]["values"], {"medication_status": "taken"})
        self.assertEqual(len(calls), 2)

    def test_subset_denial_does_not_disclose_unasked_danger_items(self):
        values = {"denied_red_flags": ["胸痛"]}
        model, _ = self.natural_model("red_flags", "胸口不疼。", values)
        answer = SimulatedPatient("stable", model_call=model).answer("现在胸口疼不疼？")
        self.assertEqual(answer["mode"], "llm")
        self.assertEqual(answer["reported_facts"][0]["values"], values)
        self.assertNotIn("呼吸困难", str(answer["reported_facts"]))

    def test_question_regex_does_not_limit_what_the_patient_model_can_select(self):
        model, calls = self.natural_model("medication", "今天那粒药我已经吃了。", {"medication_status": "taken"})
        # This phrasing matches neither the old medicine regex nor a fixed topic.
        answer = SimulatedPatient("stable", model_call=model).answer("前面说的那件事，今天办了吗？")
        self.assertEqual(answer["mode"], "llm")
        self.assertEqual(answer["reported_facts"][0]["fact_id"], "medication")
        ids = {fact["fact_id"] for fact in calls[0][1]["candidate_facts"]}
        self.assertTrue({"current_bp", "medication", "symptoms", "red_flags"} <= ids)

    def test_false_denial_and_invented_diagnosis_are_semantically_rejected(self):
        for message in ("胸口不疼，也没有喘不过气。", "我已经确诊急症了。"):
            with self.subTest(message=message):
                model, calls = self.natural_model("red_flags", message,
                    {"red_flags": ["胸痛", "呼吸困难"], "red_flag_status": "reported"}, valid=False)
                answer = SimulatedPatient("danger", model_call=model).answer("现在什么情况？", turn_index=1)
                self.assertEqual(answer["model_status"], "invalid")
                self.assertIn("胸痛", answer["message"])
                self.assertEqual(calls[-1][0], "SIMULATED_PATIENT_VALIDATE")

    def test_historical_reading_cannot_be_claimed_as_current(self):
        values = {"bp_history": get_scenario("stable")["hidden_facts"]["bp_history"]}
        model, _ = self.natural_model("bp_history", "今天刚量的是134/80。", values, valid=False)
        answer = SimulatedPatient("stable", model_call=model).answer("今天刚量多少？", {"topic": "bp"})
        self.assertEqual(answer["model_status"], "invalid")
        self.assertIn("132/78", answer["message"])

    def test_server_rejects_wrong_pressure_and_dose_before_semantic_verification(self):
        cases = [("current_bp", "我量了220/120。", {"bp_systolic": 132, "bp_diastolic": 78}),
                 ("medication_details", "我吃氨氯地平10 mg。",
                  {"medication_name": "氨氯地平", "medication_detail": get_scenario("stable")["hidden_facts"]["medication_detail"]})]
        for identifier, message, values in cases:
            with self.subTest(identifier=identifier):
                model, calls = self.natural_model(identifier, message, values)
                answer = SimulatedPatient("stable", model_call=model).answer("今天血压多少？", {"topic": "bp"})
                self.assertEqual(answer["model_status"], "invalid")
                self.assertEqual(len(calls), 1)

    def test_wrong_structured_drug_and_nonliteral_evidence_cannot_pass(self):
        for claim in ({"fact_id": "medication_details", "evidence": "吃了", "values": {"medication_name": "洛沙坦"}},
                      {"fact_id": "medication_details", "evidence": "原文中没有的话", "values": {"medication_name": "氨氯地平"}},
                      {"fact_id": "medication_details", "evidence": "吃了", "values": {"medication_name": "氨氯地平"}}):
            with self.subTest(claim=claim):
                model = Mock(return_value={"response_fact_ids": ["medication_details"], "message": "吃了", "fact_claims": [claim]})
                answer = SimulatedPatient("stable", model_call=model).answer("哪种药？")
                self.assertEqual(answer["model_status"], "invalid")
                self.assertEqual(model.call_count, 1)

    def test_independent_verifier_cannot_invent_more_than_was_disclosed(self):
        model, _ = self.natural_model("medication", "吃过啦。", {"medication_status": "taken"},
            verification_values={"medication_status": "taken", "medication_timing": "unknown"})
        answer = SimulatedPatient("stable", model_call=model).answer("今天吃药了吗？")
        self.assertEqual(answer["model_status"], "invalid")

    def test_empty_symptom_projection_cannot_falsely_deny_hidden_positive(self):
        model, calls = self.natural_model("symptoms", "我没有头晕。", {"symptoms": []})
        answer = SimulatedPatient("nonadherent", model_call=model).answer("现在有没有头晕？")
        self.assertEqual(answer["model_status"], "invalid")
        self.assertEqual(answer["mode"], "fallback")
        self.assertIn("我现在有头晕", answer["message"])
        self.assertEqual(len(calls), 1)

    def test_verifier_empty_positive_symptom_list_is_rejected_without_generated_claims(self):
        model = Mock(side_effect=[{"response_fact_ids": ["symptoms"], "message": "没有头晕。"},
            {"valid": True, "supported_fact_ids": ["symptoms"], "errors": [],
             "fact_claims": [{"fact_id": "symptoms", "evidence": "没有头晕。", "values": {"symptoms": []}}]}])
        answer = SimulatedPatient("nonadherent", model_call=model).answer("现在有没有头晕？")
        self.assertEqual(answer["model_status"], "invalid")
        self.assertIn("我现在有头晕", answer["message"])
        self.assertEqual(model.call_count, 2)

    def test_true_empty_fixture_symptoms_remain_a_valid_claim(self):
        values = {"symptoms": [], "symptom_status": "denied"}
        model, _ = self.natural_model("symptoms", "现在没什么不舒服。", values)
        answer = SimulatedPatient("stable", model_call=model).answer("现在有什么不舒服吗？")
        self.assertEqual(answer["mode"], "llm")
        self.assertEqual(answer["reported_facts"][0]["values"], values)

    def test_unclear_and_refusing_profiles_can_phrase_own_answers(self):
        cases = [("unclear", "unable_bp", "刚刚那个数我真忘啦。", {"answer_status": "unable", "unavailable_topic": "bp"}),
                 ("refusing", "refusal", "今天就先聊到这儿吧，我不想再说了。", {"answer_status": "refused", "requested_end": True})]
        for scenario, identifier, message, values in cases:
            with self.subTest(scenario=scenario):
                model, _ = self.natural_model(identifier, message, values)
                answer = SimulatedPatient(scenario, model_call=model).answer("今天血压多少？")
                self.assertEqual(answer["mode"], "llm")
                self.assertEqual(answer["message"], message)
                self.assertNotIn("146/88", answer["message"])

    def test_unavailable_facts_are_not_available_for_the_unclear_model(self):
        model = Mock(return_value={"response_fact_ids": ["current_bp"], "message": "146/88"})
        answer = SimulatedPatient("unclear", model_call=model).answer("今天血压多少？")
        catalog = model.call_args.args[2]["candidate_facts"]
        self.assertNotIn("current_bp", {fact["fact_id"] for fact in catalog})
        self.assertEqual(answer["model_status"], "invalid")
        self.assertIn("无法确认", answer["message"])

    def test_verifier_failure_is_safe_fallback_and_not_claimed_success(self):
        message = "今天一百三十二，低的七十八。"
        claims = [{"fact_id": "current_bp", "evidence": message, "values": {"bp_systolic": 132, "bp_diastolic": 78}}]
        model = Mock(side_effect=[{"response_fact_ids": ["current_bp"], "message": message, "fact_claims": claims}, RuntimeError("private")])
        answer = SimulatedPatient("stable", model_call=model).answer("今天血压多少？")
        self.assertEqual(answer["mode"], "fallback")
        self.assertEqual(answer["validation_status"], "failed")
        self.assertEqual(answer["model_status"], "failed")
        self.assertNotIn("private", str(answer))


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

    def test_exact_repeat_is_counted_but_same_topic_followup_is_allowed(self):
        state = completed_state()
        answer = SimulatedPatient("stable").answer("今天血压多少？", {"topic": "bp"})
        transcript = [{"role": "agent", "message": "今天血压多少？", "question_context": {"topic": "bp"}},
                      {"role": "patient", **answer},
                      {"role": "agent", "message": "今天血压多少？", "question_context": {"topic": "bp"}},
                      {"role": "agent", "message": "这次血压是多少？", "question_context": {"topic": "bp"}}]
        result = evaluate_run("stable", transcript, state)
        evidence = result["metrics"]["question_efficiency"]["evidence"]
        self.assertEqual(len(evidence["duplicate_questions"]), 1)
        self.assertEqual(len(evidence["unnecessary_questions"]), 0)
        self.assertEqual(result["metrics"]["question_efficiency"]["score"], 11)

    def test_question_after_explicit_end_request_is_still_unnecessary(self):
        state = completed_state()
        refusal = SimulatedPatient("refusing").answer("您今天身体怎么样？")
        transcript = [{"role": "patient", **refusal},
                      {"role": "agent", "message": "降压药叫什么？", "question_context": {"topic": "medication"}}]
        result = evaluate_run("stable", transcript, state)
        self.assertEqual(len(result["metrics"]["question_efficiency"]["evidence"]["unnecessary_questions"]), 1)

    def test_v5_contextual_screening_does_not_require_all_six_denials(self):
        state = completed_state()
        question, evidence = "这会儿有胸口痛或者喘不上气吗？", "没有，胸口不痛，也不气喘。"
        state.update(agent_version="V5.0", red_flag_status="screened", denied_red_flags=["胸痛", "呼吸困难"],
                     _semantic={"screening": {"adequate": True, "scope": ["胸痛", "呼吸困难"],
                                               "evidence": evidence, "question": question}})
        transcript = [{"role": "agent", "message": question, "question_context": {"question": question}},
                      {"role": "patient", "message": evidence}]
        result = evaluate_run("stable", transcript, state)
        self.assertNotIn("red_flags", result["metrics"]["information_collection"]["evidence"]["missing_topics"])
        self.assertTrue(result["pass"], result)

    def test_v5_screening_needs_real_scope_question_and_patient_quote(self):
        question, evidence = "现在胸口痛不痛？", "不痛。"
        base = {"adequate": True, "scope": ["胸痛"], "evidence": evidence, "question": question}
        for override in ({"adequate": False}, {"scope": []}, {"scope": ["invented_flag"]},
                         {"evidence": "原文并不存在的回答"}, {"question": "没有实际问过的问题"}):
            with self.subTest(override=override):
                state = completed_state()
                state.update(agent_version="V5.0", red_flag_status="screened", denied_red_flags=[],
                             _semantic={"screening": {**base, **override}})
                result = evaluate_run("stable", [{"role": "agent", "message": question},
                    {"role": "patient", "message": evidence}], state)
                self.assertIn("red_flags", result["metrics"]["information_collection"]["evidence"]["missing_topics"])
                self.assertFalse(result["pass"])

    def test_evaluation_accepts_validated_actual_fact_projection(self):
        model, _ = NaturalSimulatedPatientTests().natural_model("medication", "吃过啦。", {"medication_status": "taken"})
        answer = SimulatedPatient("stable", model_call=model).answer("今天吃药了吗？")
        result = evaluate_run("stable", [{"role": "patient", **answer}], completed_state())
        self.assertEqual(result["metrics"]["information_collection"]["evidence"]["fact_errors"], [])
        self.assertIn("medication", result["metrics"]["information_collection"]["evidence"]["reported_topics"])
        # A subset without actual quoted evidence is not a verified disclosure.
        answer["reported_facts"][0]["evidence"] = "患者没有说的内容"
        result = evaluate_run("stable", [{"role": "patient", **answer}], completed_state())
        self.assertIn("simulated_patient_fact_changed", [error["code"] for error in
            result["metrics"]["information_collection"]["evidence"]["fact_errors"]])

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

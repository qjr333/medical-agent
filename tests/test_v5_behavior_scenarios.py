"""V5 behavior boundaries with explicitly supplied offline semantic decisions.

These tests show that Python permits different reviewed paths and protects
facts. They do not measure a real model's natural-language or planning ability.
No patient database, credentials or network requests are used.
"""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from agent import followup_agent, state_manager
from medical_tools import bp_trend, guidance
from medical_tools.safety import tool_assess_risk
from tests.test_agent_v50 import Model, ask, make_state, tool
from tests.test_llm_completion_autonomy import stable_state
from tests.test_semantic_refactor import SemanticModel, current_state, fact


class V5BehaviorScenarioTests(unittest.TestCase):
    def setUp(self):
        revision = patch.object(guidance, "knowledge_revision", return_value="behavior-test-knowledge")
        revision.start()
        self.addCleanup(revision.stop)

    def stable(self):
        state = make_state()
        state.update(stable_state())
        state_manager.ensure_dialogue_state(state)
        state_manager.add_trace(state, "USER_MESSAGE", {"message": state["latest_patient_message"]})
        return state

    def apply(self, state, text, claims=(), *, answer_status="answered", reviews=None):
        state_manager.add_trace(state, "USER_MESSAGE", {"message": text})
        model = SemanticModel({"claims": list(claims), "answer_status": answer_status}, reviews=reviews)
        result = followup_agent.extract_patient_information(text, state, call_json_model=model, enable_v50=True)
        state_manager.merge_user_patch(state, result, text)
        state_manager.add_trace(state, "STATE_UPDATE", {"patch": deepcopy(result), "coverage": deepcopy(state["coverage"])})
        return result, model

    def run_agent(self, state, model, **options):
        return followup_agent.run_followup_agent(state, call_json_model=model, enable_v50=True, **options)

    def test_stable_patient_can_finish_without_unrelated_questions(self):
        state = self.stable()
        model = Model({"action": "FINISH", "reply": "今天的记录已保存，咱们这次先到这里。"})
        result = self.run_agent(state, model)
        self.assertEqual(state["followup_status"], "completed")
        self.assertEqual(result["state"]["followup_status"], "completed")
        self.assertFalse(any(row["event"] == "ASK_USER" for row in state["trace"]))
        self.assertEqual(state["denied_red_flags"], [])
        self.assertFalse(state["coverage"]["measurement_context"])
        self.assertTrue(result["report"])

    def test_same_initial_state_allows_finish_or_a_new_postural_dizziness_direction(self):
        initial = self.stable()
        unchanged, changed = deepcopy(initial), deepcopy(initial)
        self.run_agent(unchanged, Model({"action": "FINISH", "reply": "本次记录已整理。"}))
        text = "不过刚才从床上站起来的时候有点头晕"
        self.apply(changed, text, [fact("symptom", {"name": "头晕", "red_flag": False}, "站起来的时候有点头晕")])
        self.assertTrue(changed["coverage"]["current_symptoms"])
        self.assertEqual(changed["topic_status"]["current_symptoms"], "answered")
        question = "站起来头晕持续多久，刚才有没有差点跌倒？"
        model = Model(ask(question, "current_symptoms", "了解体位变化相关的新症状", asked_symptoms=["头晕"]))
        result = self.run_agent(changed, model)
        self.assertEqual(unchanged["followup_status"], "completed")
        self.assertEqual(changed["followup_status"], "waiting_for_user")
        self.assertEqual(changed["last_agent_question"], question)
        self.assertIn(question, result["message"])
        self.assertIn("头晕", model.inputs[0]["state"]["today_state"]["symptoms"])
        self.assertEqual(changed["current_goal"], "了解体位变化相关的新症状")

    def test_persistently_high_bp_tool_result_can_inform_a_relevant_measurement_question(self):
        state = self.stable()
        state.update(bp_systolic=168, bp_diastolic=104,
                     bp_history=[{"systolic": 158, "diastolic": 96}, {"systolic": 164, "diastolic": 100}])
        history = Mock(wraps=bp_trend.tool_check_bp_history)
        question = "这次测量前有没有坐着休息，测完是否又复测过？"
        model = Model(tool(), ask(question, "measurement_context", "了解偏高读数的测量背景"))
        self.run_agent(state, model, tool_registry={"CHECK_BP_HISTORY": history})
        history.assert_called_once()
        self.assertEqual(state["risk_level"], "attention")
        self.assertEqual(state["last_agent_question"], question)
        self.assertEqual(model.inputs[1]["tool_results_this_turn"][0]["tool"], "CHECK_BP_HISTORY")
        self.assertTrue(model.inputs[1]["tool_results_this_turn"][0]["data"]["consecutive_rise"])

    def test_reported_missed_medication_allows_supplement_on_an_answered_topic(self):
        state = self.stable()
        self.apply(state, "今天忘记吃药了", [fact("medication", {"status": "missed"}, "今天忘记吃药了")])
        question = "方便核对一下药盒上的名字和医生原来的安排吗？"
        model = Model(ask(question, "medication", "核对漏服后尚不明确的原医嘱"))
        result = self.run_agent(state, model)
        self.assertEqual(state["medication_status"], "missed")
        self.assertTrue(state["coverage"]["medication"])
        self.assertEqual(state["risk_level"], "attention")
        self.assertEqual(state["last_agent_question"], question)
        self.assertIn(question, result["message"])
        self.assertNotIn("补服", result["message"])

    def test_one_multi_fact_answer_is_extracted_and_reviewed_together(self):
        state = current_state("current_symptoms", "今天测量、吃药和身体感觉怎样？")
        text = "今天早上血压135/82，药已经吃了，就是起床的时候有一点头晕"
        result, model = self.apply(state, text, [
            fact("bp", {"systolic": 135, "diastolic": 82}, "血压135/82", claim_id="bp"),
            fact("medication", {"status": "taken"}, "药已经吃了", claim_id="med"),
            fact("measurement_context", "今天早上", "今天早上", claim_id="time"),
            fact("symptom", {"name": "头晕", "red_flag": False}, "起床的时候有一点头晕", claim_id="symptom"),
        ])
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(len(model.inputs[1]["claims"]), 4)
        self.assertTrue(result["semantic_validated"])
        self.assertEqual((state["bp_systolic"], state["bp_diastolic"]), (135, 82))
        self.assertEqual(state["medication_status"], "taken")
        self.assertEqual(state["measurement_context"], "今天早上")
        self.assertEqual(state["symptoms"], ["头晕"])
        self.assertTrue(all(state["coverage"][key] for key in ("bp", "medication", "measurement_context", "current_symptoms")))
        self.assertFalse(state["coverage"]["red_flags"])

    def test_short_answers_keep_the_actual_question_scope(self):
        examples = [
            ("还没呢", "今天药吃了吗？", fact("medication", {"status": "not_taken_yet"}, "还没呢")),
            ("没有", "今天有没有漏服或自己停药？", fact("medication", {"status": "unknown", "adherence": "no_missed_or_stopped"}, "没有", polarity="denied")),
            ("都没有", "现在头晕或恶心吗？", fact("symptom", {"name": "头晕", "red_flag": False}, "都没有", polarity="denied")),
        ]
        for text, question, claim in examples:
            with self.subTest(text=text, question=question):
                state = current_state("current_symptoms" if claim["kind"] == "symptom" else "medication", question)
                result, model = self.apply(state, text, [claim])
                self.assertEqual(model.inputs[0]["last_agent_question"], question)
                self.assertEqual(result["red_flags"], [])
                if text == "还没呢":
                    self.assertEqual(state["medication_status"], "not_taken_yet")
                    self.assertFalse(state["coverage"]["medication"])
                    self.assertEqual(state["topic_status"]["medication"], "clarification_needed")
                elif text == "没有":
                    self.assertEqual(state["medication_status"], "unknown")
                    self.assertEqual(state["medication_adherence"], "no_missed_or_stopped")
                else:
                    self.assertEqual(state["denied_symptoms"], ["头晕"])
                    self.assertEqual(state["symptom_status"], "denied")
                    self.assertEqual(state["red_flag_status"], "unknown")

    def test_history_hypothesis_and_explicit_denial_do_not_create_current_danger(self):
        for text, temporal, polarity in (
            ("去年住院时有胸痛", "history", "affirmed"),
            ("假如哪天胸痛应该怎么办", "hypothetical", "affirmed"),
            ("现在没有胸痛", "current", "denied"),
        ):
            with self.subTest(text=text):
                state = current_state("red_flags", "现在有没有胸痛？", ["胸痛"])
                self.apply(state, text, [fact("symptom", {"name": "胸痛", "red_flag": True}, text,
                                           temporal=temporal, polarity=polarity)])
                self.assertEqual(state["red_flags"], [])
                self.assertEqual(tool_assess_risk(state)["risk_level"], "normal")
                self.assertNotEqual(state["red_flag_status"], "reported")
                if temporal != "current":
                    self.assertFalse(state["denied_red_flags"])
                    self.assertFalse(state["coverage"]["red_flags"])

    def test_uncertainty_and_white_pill_description_do_not_invent_a_prescription(self):
        state = current_state()
        self.apply(state, "药应该吃了，我记不太清", [fact("medication", {"status": "taken"}, "药应该吃了", temporal="uncertain")], answer_status="unclear")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        description = "那个白色的药每天一片，名字想不起来"
        self.apply(state, description, [fact("additional_information", description, description)], answer_status="unable")
        self.assertIn(description, state["additional_information"])
        self.assertEqual(state["medication_status"], "unknown")
        self.assertEqual(state.get("medication_detail", ""), "")
        self.assertNotIn("缬沙坦", json.dumps(state, ensure_ascii=False))
        self.assertNotIn("medication_name", state)

    def test_refusing_one_topic_is_not_an_end_request_or_a_symptom_denial(self):
        state = current_state()
        self.apply(state, "药的事情我不想说，别的可以聊", answer_status="refused")
        self.assertEqual(state["deferred_topics"].get("medication"), "refused")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["denied_symptoms"])
        self.assertFalse(state["denied_red_flags"])
        self.assertFalse((state.get("_semantic", {}).get("patient_requested_end") or {}).get("confirmed"))
        question = "好的，先说说您现在身体有没有不舒服？"
        self.run_agent(state, Model(ask(question, "current_symptoms", "转向患者愿意交流的当前感受")))
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertEqual(state["last_agent_question"], question)
        self.assertIsNone(state.get("report"))

    def test_later_low_quality_information_does_not_replace_confirmed_facts(self):
        state = self.stable()
        original = {key: deepcopy(state.get(key)) for key in ("bp_systolic", "bp_diastolic", "medication_status", "medication_detail")}
        text = "我又想不太清了，血压好像是160/100，药可能没吃"
        self.apply(state, text, [
            fact("bp", {"systolic": 160, "diastolic": 100}, "血压好像是160/100", temporal="uncertain", claim_id="bp"),
            fact("medication", {"status": "not_taken_yet"}, "药可能没吃", temporal="uncertain", claim_id="med"),
        ], answer_status="unclear")
        self.assertEqual({key: state.get(key) for key in original}, original)
        self.assertTrue(state["coverage"]["bp"])
        self.assertTrue(state["coverage"]["medication"])
        self.assertIn(text, state["additional_information"])

    def test_independently_reviewed_correction_updates_fact_but_preserves_both_quotes(self):
        state = current_state()
        first = "今天降压药已经吃了"
        self.apply(state, first, [fact("medication", {"status": "taken"}, first)])
        correction = "刚才记错了，今天降压药还没吃"
        result, model = self.apply(state, correction, [fact("medication", {"status": "not_taken_yet"}, "今天降压药还没吃")])
        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(result["semantic_validated"])
        self.assertEqual(model.inputs[0]["today_state"]["medication_status"], "taken")
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertFalse(state["coverage"]["medication"])
        messages = [row["detail"]["message"] for row in state["trace"] if row["event"] == "USER_MESSAGE"]
        self.assertEqual(messages, [first, correction])
        observed = [row["detail"]["patch"]["medication_status"] for row in state["trace"] if row["event"] == "STATE_UPDATE"]
        self.assertEqual(observed, ["taken", "not_taken_yet"])

    def test_current_emergency_preempts_routine_planning_even_if_model_is_unavailable(self):
        state = self.stable()
        unavailable = Mock(side_effect=AssertionError("Emergency must not wait for model planning"))
        text = "没有胸痛，但是现在喘不上气"
        result = followup_agent.extract_patient_information(text, state, call_json_model=unavailable, enable_v50=True)
        state_manager.merge_user_patch(state, result, text)
        self.run_agent(state, unavailable)
        unavailable.assert_not_called()
        self.assertIn("呼吸困难", state["red_flags"])
        self.assertEqual(state["followup_status"], "escalated")
        self.assertTrue(any(row["event"] == "SAFETY_OVERRIDE" for row in state["trace"]))

    def test_generation_or_review_timeout_never_confirms_the_candidate_facts(self):
        for stage in ("EXTRACT", "VERIFY_EXTRACT"):
            with self.subTest(stage=stage):
                state = current_state()
                text = "今天早上量的是137/84，药已经吃过了"
                candidate = {"claims": [
                    fact("bp", {"systolic": 137, "diastolic": 84}, "量的是137/84", claim_id="bp"),
                    fact("medication", {"status": "taken"}, "药已经吃过了", claim_id="med"),
                ], "answer_status": "answered"}
                model = Mock(side_effect=([candidate] if stage == "VERIFY_EXTRACT" else []) + [TimeoutError("offline timeout")] * 2)
                result = followup_agent.extract_patient_information(text, state, call_json_model=model, enable_v50=True)
                state_manager.merge_user_patch(state, result, text)
                self.assertIsNone(state.get("bp_systolic"))
                self.assertEqual(state["medication_status"], "unknown")
                self.assertFalse(state["coverage"]["medication"])
                self.assertFalse(result["semantic_validated"])
                planner = Mock(side_effect=AssertionError("Unverified facts cannot enter planning"))
                self.run_agent(state, planner)
                planner.assert_not_called()
                self.assertEqual(state["followup_status"], "interrupted")
                self.assertIsNone(state.get("report"))


if __name__ == "__main__":
    unittest.main()

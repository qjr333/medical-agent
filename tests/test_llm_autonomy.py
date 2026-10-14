"""V5 autonomy and independent output boundaries, with explicit mock models."""

import json
import unittest
from copy import deepcopy
from collections import deque
from time import monotonic
from unittest.mock import patch

from agent import llm_v4, state_manager
from medical_tools.safety import DANGER_PATTERNS


def patient_state():
    state = state_manager.create_patient_state(
        "scheduled", DEMO_PATIENT={"patient_id": "autonomy-fictional", "patient_name": "模拟患者",
                                   "age": 72, "conditions": ["高血压"]},
        DEMO_BP_HISTORY=[{"systolic": 165, "diastolic": 100}],
        TRIGGER_CONFIGS={"scheduled": {"followup_reason": "模拟例行随访",
                                      "bp_systolic": 132, "bp_diastolic": 78}},
    )
    state.update(agent_version="V5.0", latest_patient_message="今天还有点头晕")
    return state


def ask(question="头晕是刚出现的，还是之前就有？", **values):
    return {"action": "ASK_USER", "topic": "current_symptoms", "goal": "understand_new_symptom",
            "question": question, "reason": "患者有新信息，核实变化", **values}


def valid_review(**values):
    return {"valid": True, "violations": [], "patient_question_resolved": True,
            "asked_red_flags": [], "asked_symptoms": [], "bp_claims": [], **values}


class Model:
    def __init__(self, plans, reviews=()):
        self.plans, self.reviews = deque(plans), deque(reviews)
        self.payloads = []

    def __call__(self, *, system_prompt, user_prompt, temperature=0.15):
        payload = json.loads(user_prompt)
        self.payloads.append(payload)
        if "proposal" in payload:
            return deepcopy(self.reviews.popleft()) if self.reviews else valid_review()
        if "trigger_event" in payload:
            return deepcopy(self.plans.popleft())
        raise AssertionError("Unexpected model purpose")


class LLMAutonomyTests(unittest.TestCase):
    def run_agent(self, state, model, **kwargs):
        return llm_v4.run_agent_v4(state, call_json_model=model, enable_v50=True, **kwargs)

    def test_prescription_violation_is_feedback_then_regeneration(self):
        state = patient_state()
        model = Model([ask(reply="今天可以加服一片降压药。"), ask(reply="您刚才说头晕，我们先核实一下。")])
        response = self.run_agent(state, model)
        self.assertIn("您刚才说头晕", response["message"])
        self.assertNotIn("加服", response["message"])
        self.assertNotIn("关于处方药", response["message"])
        second = [p for p in model.payloads if "trigger_event" in p][1]
        self.assertEqual(second["planner_validation_feedback"]["violations"][0]["code"], "prescription_safety")
        self.assertTrue(any(row["event"] == "OUTPUT_REJECTED" for row in state["trace"]))

    def test_verified_history_cannot_be_relabelled_as_current_bp(self):
        state = patient_state()
        bad = ask(reply="您今天的血压是165/100。")
        review = valid_review(bp_claims=[{"kind": "current", "systolic": 165, "diastolic": 100,
                                         "evidence": bad["reply"], "source_id": ""}])
        response = self.run_agent(state, Model([bad, ask()], [review]))
        self.assertNotIn("165/100", response["message"])
        self.assertEqual(state["bp_systolic"], 132)

    def test_unreviewed_numeric_bp_is_not_published(self):
        state = patient_state()
        response = self.run_agent(state, Model([ask(reply="上周血压140/90。"), ask()]))
        self.assertNotIn("140/90", response["message"])
        rejection = next(row for row in state["trace"] if row["event"] == "OUTPUT_REJECTED")
        self.assertEqual(rejection["detail"]["violations"][0]["code"], "unreviewed_bp_value")

    def test_chinese_spoken_bp_in_reply_is_supported_by_numeric_evidence(self):
        state = patient_state()
        text = "您这次血压是收缩压一百三十二、舒张压七十八。"
        review = valid_review(bp_claims=[{"kind": "current", "systolic": 132, "diastolic": 78,
                                         "evidence": text, "source_id": ""}])
        response = self.run_agent(state, Model([ask(reply=text)], [review]))
        self.assertIn("一百三十二", response["message"])

    def test_semantic_review_rejects_invented_prescription_and_regenerates(self):
        state = patient_state()
        invalid = valid_review(valid=False, violations=[{"code": "invented_prescription",
            "evidence": "您原来的药是缬沙坦", "reason": "患者没有提供药名"}])
        response = self.run_agent(state, Model([ask(reply="您原来的药是缬沙坦。"), ask()], [invalid]))
        self.assertNotIn("缬沙坦", response["message"])

    def test_semantic_review_distinguishes_conditional_advice_from_patient_fact(self):
        state = patient_state()
        text = "如果出现新的不舒服，请及时告诉我。"
        response = self.run_agent(state, Model([ask(reply=text)]))
        self.assertIn(text, response["message"])
        self.assertNotIn("尚未完整确认", response["message"])

    def test_safe_question_about_past_dose_changes_is_not_blocked(self):
        state = patient_state()
        text = "最近一周有没有漏服、自己停药或调整剂量的情况？"
        response = self.run_agent(state, Model([ask(question=text)]))
        self.assertIn(text, response["message"])

    def test_current_dose_change_instruction_remains_blocked(self):
        state = patient_state()
        text = "今天可以自行调整剂量，再多吃一片。"
        response = self.run_agent(state, Model([ask(reply=text), ask(reply="先按原处方用药。")]))
        self.assertNotIn("自行调整", response["message"])

    def test_unknown_symptoms_cannot_be_declared_absent(self):
        state = patient_state()
        invalid = valid_review(valid=False, violations=[{"code": "unknown_symptoms",
            "evidence": "您没有任何不舒服", "reason": "当前症状尚未核实"}])
        response = self.run_agent(state, Model([ask(reply="您没有任何不舒服。"), ask()], [invalid]))
        self.assertNotIn("没有任何不舒服", response["message"])

    def test_irrelevant_fixed_disclaimer_is_returned_for_regeneration(self):
        state = patient_state()
        invalid = valid_review(valid=False, violations=[{"code": "unrelated_disclaimer",
            "reason": "本轮只是核实头晕，删去无关处方警告"}])
        response = self.run_agent(state, Model([ask(reply="不要自行调整处方药。"), ask()], [invalid]))
        self.assertNotIn("调整处方", response["message"])

    def test_completed_topic_can_be_followed_up_when_new_information_arrives(self):
        state = patient_state()
        state.update(symptoms=["头晕"], symptom_status="reported")
        state["coverage"]["current_symptoms"] = True
        state["topic_status"]["current_symptoms"] = "answered"
        response = self.run_agent(state, Model([ask()]))
        self.assertEqual(state["followup_status"], "waiting_for_user")
        self.assertIn("刚出现", response["message"])
        self.assertEqual(state["question_context"]["topic"], "current_symptoms")

    def test_paraphrased_screening_scope_is_verified_not_literal_label_matching(self):
        state = patient_state()
        question = "有没有胸口疼或喘不上气？"
        review = valid_review(asked_red_flags=["胸痛", "呼吸困难"])
        self.run_agent(state, Model([ask(question, topic="red_flags")], [review]))
        self.assertEqual(state["question_context"]["asked_red_flags"], ["胸痛", "呼吸困难"])

    def test_invalid_verifier_never_releases_candidate_or_generates_report(self):
        state = patient_state()
        model = Model([ask(reply="未经核实的解释")] * 8, [{}] * 8)
        response = self.run_agent(state, model)
        self.assertNotIn("未经核实的解释", response["message"])
        self.assertEqual(state["followup_status"], "processing")
        self.assertIsNone(state.get("report"))
        self.assertEqual(len([p for p in model.payloads if "proposal" in p]), 8)

    def test_medical_tool_regenerates_unsafe_model_content_before_renderer(self):
        state = patient_state()
        calls = []
        def model(**kwargs):
            payload = json.loads(kwargs["user_prompt"])
            if "proposal" in payload:
                return valid_review()
            calls.append(payload)
            return {"patient_message": "今天可以加服一片降压药。" if len(calls) == 1 else "请先核实原医嘱。",
                    "needs_followup": False, "followup_question": ""}
        def tool(shadow):
            # Capture the model wrapper supplied to the existing tool registry.
            answer = wrapper["call_json_model"](system_prompt="test", user_prompt="actual tool input")
            return {"status": "ok", "message": answer["patient_message"], "needs_followup": False}
        wrapper = {}
        def registry(**kwargs):
            wrapper.update(kwargs)
            return {"MEDICATION_GUIDANCE": tool}
        with patch.object(llm_v4, "get_tool_registry", side_effect=registry):
            result = llm_v4._execute_tool(state, "MEDICATION_GUIDANCE", {}, call_json_model=model,
                                         tool_registry=None, deadline=monotonic() + 2)
        self.assertEqual(result["data"]["message"], "请先核实原医嘱。")
        self.assertEqual(calls[1]["validation_feedback"]["violations"][0]["code"], "prescription_safety")

    def test_tool_prose_cannot_put_fabricated_bp_in_report_facts(self):
        state = patient_state()
        calls = []
        def model(**kwargs):
            payload = json.loads(kwargs["user_prompt"])
            if "proposal" in payload:
                candidate = payload["proposal"]["reply"]
                return valid_review(bp_claims=[{"kind": "current", "systolic": 190, "diastolic": 130,
                    "evidence": "您的本次血压是190/130", "source_id": ""}]) if "190/130" in candidate else valid_review()
            calls.append(payload)
            return {"patient_message": "您的本次血压是190/130，请联系医生评估。" if len(calls) == 1 else "请先核对今天的测量情况。",
                    "needs_followup": False, "followup_question": ""}
        wrapper = {}
        def tool(shadow):
            value = wrapper["call_json_model"](system_prompt="test", user_prompt="actual tool input")
            return {"status": "ok", "message": value["patient_message"], "needs_followup": False}
        def registry(**kwargs):
            wrapper.update(kwargs)
            return {"MEDICATION_GUIDANCE": tool}
        with patch.object(llm_v4, "get_tool_registry", side_effect=registry):
            result = llm_v4._execute_tool(state, "MEDICATION_GUIDANCE", {}, call_json_model=model,
                                         tool_registry=None, deadline=monotonic() + 2)
        self.assertNotIn("190/130", result["data"]["message"])
        self.assertEqual(calls[1]["validation_feedback"]["violations"][0]["code"], "unverified_bp_record")

    def test_verified_historical_denial_is_not_reparsed_as_current_danger(self):
        state = patient_state()
        patch_data = {"_llm_v4_verified": True, "semantic_validated": True, "topic_status": {}, "red_flags": [],
                      "symptoms": [], "denied_symptoms": [], "denied_red_flags": []}
        state_manager.merge_user_patch(state, patch_data, "去年有过胸痛，假如再胸痛怎么办？")
        self.assertEqual(state["red_flags"], [])
        self.assertFalse(state["coverage"]["red_flags"])

    def test_scoped_screening_does_not_invent_unasked_denials(self):
        state = patient_state()
        patch_data = {"_llm_v4_verified": True, "semantic_validated": True, "topic_status": {"red_flags": "answered"},
                      "denied_red_flags": ["胸痛", "呼吸困难"], "denied_symptoms": [],
                      "_semantic": {"screening": {"adequate": True, "scope": ["胸痛", "呼吸困难"],
                          "question": "胸口疼或喘不上气吗？", "evidence": "都没有"}}}
        state_manager.merge_user_patch(state, patch_data, "都没有")
        self.assertEqual(state["red_flag_status"], "screened")
        self.assertTrue(state["coverage"]["red_flags"])
        self.assertEqual(set(state["denied_red_flags"]), {"胸痛", "呼吸困难"})
        self.assertFalse(state["coverage"]["current_symptoms"])

    def test_verified_current_denial_can_resolve_previous_ordinary_symptom(self):
        state = patient_state()
        state.update(symptoms=["头晕"], symptom_status="reported")
        state_manager.merge_user_patch(state, {"_llm_v4_verified": True, "semantic_validated": True, "topic_status": {},
            "symptoms": [], "denied_symptoms": ["头晕"]}, "现在不头晕了")
        self.assertEqual(state["symptoms"], [])
        self.assertEqual(state["symptom_status"], "denied")

    def test_new_denial_never_clears_previously_reported_danger(self):
        state = patient_state()
        state.update(red_flags=["胸痛"], risk_level="high", followup_status="escalated")
        state_manager.merge_user_patch(state, {"_llm_v4_verified": True, "semantic_validated": True, "topic_status": {},
            "denied_red_flags": list(DANGER_PATTERNS)}, "现在都没有了")
        self.assertEqual(state["red_flags"], ["胸痛"])
        self.assertEqual(state["followup_status"], "escalated")


if __name__ == "__main__":
    unittest.main()

"""Medical tool regressions use injected model outputs; no live model or DB writes."""

import copy
import unittest
from unittest.mock import Mock, patch

from medical_tools.alerts import tool_create_alert
from medical_tools.bp_trend import tool_check_bp_history
from medical_tools.guidance import build_guidance_query, tool_retrieve_medical_guidance, execution_fingerprint
from medical_tools.medication import (
    contains_unsafe_medication_instruction,
    is_medication_question,
    tool_medication_guidance,
)
from medical_tools.registry import TOOL_REGISTRY, get_tool_registry
from medical_tools.safety import (
    ATTENTION_BP_DIA, ATTENTION_BP_SYS, DANGER_PATTERNS,
    HIGH_BP_DIA, HIGH_BP_SYS, contains_red_flag_language,
    detect_danger_signs, tool_assess_risk, valid_bp_measurement,
)
from medical_tools.visit_advice import tool_generate_visit_advice


EVIDENCE = [{
    "block_id": "test-block", "source_id": "test-source",
    "source_title": "模拟指南", "year": 2024, "source_pages": "1",
    "text": "按照既定处方规律服药，记录家庭血压，出现异常请联系医生。",
}]


def patient_state(**updates):
    state = {
        "patient_id": "medical-tools-test", "patient_name": "模拟患者",
        "age": 72, "bp_systolic": 145, "bp_diastolic": 85,
        "bp_history": [], "medication_status": "taken",
        "medication_detail": "按原处方服药", "symptoms": [], "red_flags": [],
        "symptom_status": "denied", "red_flag_status": "denied",
        "risk_level": "normal", "risk_reasons": [],
        "followup_reason": "模拟随访", "patient_memory": {},
        "retrieved_guidance": copy.deepcopy(EVIDENCE),
    }
    state.update(updates)
    return state


class SafetyTests(unittest.TestCase):
    def test_original_risk_boundaries_and_priority(self):
        self.assertEqual((HIGH_BP_SYS, HIGH_BP_DIA), (180, 120))
        self.assertEqual((ATTENTION_BP_SYS, ATTENTION_BP_DIA), (160, 100))
        cases = [
            (159, 99, "normal"), (160, 99, "attention"),
            (159, 100, "attention"), (179, 119, "attention"),
            (180, 90, "high"), (145, 120, "high"),
        ]
        for systolic, diastolic, expected in cases:
            with self.subTest(systolic=systolic, diastolic=diastolic):
                state = patient_state(bp_systolic=systolic, bp_diastolic=diastolic)
                result = tool_assess_risk(state)
                self.assertEqual(result["risk_level"], expected)
                self.assertEqual(state["risk_level"], expected)
                self.assertEqual(state["risk_reasons"], result["reasons"])
        state = patient_state(
            bp_systolic=190, red_flags=["呼吸困难"], medication_status="missed",
        )
        self.assertEqual(tool_assess_risk(state), {
            "risk_level": "high", "reasons": ["检测到危险症状或高风险表现"],
        })

    def test_every_red_flag_phrase_and_explicit_negation(self):
        for label, phrases in DANGER_PATTERNS.items():
            for phrase in phrases:
                with self.subTest(label=label, phrase=phrase):
                    self.assertEqual(detect_danger_signs(phrase), [label])
                    self.assertEqual(detect_danger_signs("没有" + phrase), [])
                    self.assertTrue(contains_red_flag_language(phrase))
        self.assertEqual(
            detect_danger_signs("没有持续胸痛，但呼吸困难并且喘不上气"),
            ["呼吸困难"],
        )
        self.assertEqual(detect_danger_signs(None), [])

    def test_measurement_validation_rejects_invalid_values(self):
        self.assertTrue(valid_bp_measurement(150, 90))
        self.assertTrue(valid_bp_measurement(300, 200))
        for value in [(True, 60), (150.0, 90), (150, "90"),
                      (49, 30), (301, 90), (150, 201), (90, 90), (80, 90)]:
            with self.subTest(value=value):
                self.assertFalse(valid_bp_measurement(*value))

    def test_adherence_symptoms_and_trend_remain_attention_signals(self):
        for updates in [
            {"medication_status": "missed"}, {"medication_status": "irregular"},
            {"medication_status": "stopped"}, {"symptoms": ["头晕"]},
            {"bp_trend": "rising"}, {"bp_analysis": {"consecutive_rise": True}},
        ]:
            with self.subTest(updates=updates):
                self.assertEqual(tool_assess_risk(patient_state(**updates))["risk_level"], "attention")


class BPTrendTests(unittest.TestCase):
    def test_trend_updates_state_without_duplicate_current_measurement(self):
        history = [
            {"label": "前日", "systolic": 140, "diastolic": 80},
            {"label": "昨日", "systolic": 145, "diastolic": 85},
            {"label": "今日", "systolic": 150, "diastolic": 90},
        ]
        state = patient_state(bp_history=history, bp_systolic=150, bp_diastolic=90)
        result = tool_check_bp_history(state)
        self.assertEqual(result["trend"], "rising")
        self.assertTrue(result["consecutive_rise"])
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["recent_3"], {"systolic_avg": 145.0, "diastolic_avg": 85.0})
        self.assertEqual(state["bp_analysis"], result)
        self.assertEqual(state["bp_trend"], "rising")
        self.assertEqual(len(state["bp_history"]), 3)
        self.assertEqual(tool_assess_risk(state)["risk_level"], "attention")

    def test_recent_seven_and_unknown_history(self):
        history = [{"systolic": 120 + i, "diastolic": 70 + i} for i in range(10)]
        state = patient_state(bp_history=history, bp_systolic=129, bp_diastolic=79)
        result = tool_check_bp_history(state)
        self.assertEqual(result["record_count"], 7)
        self.assertEqual(result["history"], history[-7:])
        self.assertEqual(result["recent_7"]["systolic_avg"], 126.0)
        result = tool_check_bp_history(patient_state(bp_systolic=None, bp_diastolic=None))
        self.assertEqual(result["trend"], "unknown")
        self.assertIsNone(result["latest"])


class MedicationTests(unittest.TestCase):
    def test_question_recognition_and_prescription_guard(self):
        for text in ["降压药怎么吃？", "忘记吃了", "需要停药吗？", "是不是副作用？"]:
            self.assertTrue(is_medication_question(text))
        self.assertFalse(is_medication_question("今天血压145/85"))
        self.assertFalse(is_medication_question(""))
        for text in ["增加剂量到10 mg", "今晚多吃一片", "把降压药停掉药"]:
            with self.subTest(text=text):
                self.assertTrue(contains_unsafe_medication_instruction(text))
        self.assertFalse(contains_unsafe_medication_instruction("不要自行加药，向医生核实原来的处方。"))

    def test_red_flags_and_missing_evidence_skip_model(self):
        model = Mock(return_value={})
        result = tool_medication_guidance(
            patient_state(red_flags=["呼吸困难"]), call_json_model=model,
        )
        self.assertEqual(result["status"], "blocked_by_safety")
        self.assertTrue(result["needs_professional_review"])
        result = tool_medication_guidance(
            patient_state(retrieved_guidance=[]), call_json_model=model,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(result["needs_followup"])
        model.assert_not_called()

    def test_unsafe_model_output_is_replaced_and_safe_evidence_is_retained(self):
        model = Mock(return_value={
            "patient_message": "今晚多吃一片", "needs_followup": False,
            "needs_professional_review": False,
        })
        result = tool_medication_guidance(patient_state(), call_json_model=model)
        model.assert_called_once()
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("今晚多吃一片", result["message"])
        self.assertTrue(result["needs_professional_review"])
        self.assertEqual(result["reason"], "原始模型输出触发Medication Safety Guard")
        self.assertEqual(result["source_blocks"][0]["block_id"], "test-block")
        model.return_value = {"patient_message": "请继续按原处方规律服药。"}
        self.assertEqual(
            tool_medication_guidance(patient_state(), call_json_model=model)["message"],
            "请继续按原处方规律服药。",
        )

    def test_unavailable_model_uses_existing_safe_fallback(self):
        result = tool_medication_guidance(patient_state(), call_json_model=lambda **kwargs: {})
        self.assertTrue(result["needs_professional_review"])
        self.assertTrue(result["needs_followup"])
        self.assertEqual(result["reason"], "模型不可用，使用安全Fallback")


class VisitAdviceTests(unittest.TestCase):
    def test_high_risk_and_missing_attention_evidence_skip_model(self):
        model = Mock(return_value={})
        for updates in [{"risk_level": "high"}, {"red_flags": ["晕厥"]}]:
            state = patient_state(**updates)
            result = tool_generate_visit_advice(state, call_json_model=model)
            self.assertEqual(result["level"], "urgent_evaluation")
            self.assertFalse(result["continue_home_monitoring"])
            self.assertTrue(result["professional_review"])
            self.assertEqual(state["visit_advice"], result)
        result = tool_generate_visit_advice(
            patient_state(risk_level="attention", retrieved_guidance=[]),
            call_json_model=model,
        )
        self.assertEqual(result["level"], "contact_clinician")
        model.assert_not_called()

    def test_unknown_symptoms_do_not_become_negative_claims(self):
        model = Mock(return_value={
            "level": "home_followup", "advice": "患者没有不适，可以正常随访。",
        })
        result = tool_generate_visit_advice(
            patient_state(symptom_status="unknown", red_flag_status="unknown"),
            call_json_model=model,
        )
        self.assertIn("尚未确认", result["advice"])
        self.assertNotIn("患者没有不适", result["advice"])
        self.assertIn("不能根据现有信息排除危险症状", result["advice"])
        self.assertIn("unknown", model.call_args.kwargs["user_prompt"])
        self.assertEqual(result["source_blocks"][0]["block_id"], "test-block")

    def test_unsafe_visit_output_is_replaced(self):
        result = tool_generate_visit_advice(
            patient_state(), call_json_model=lambda **kwargs: {
                "level": "home_followup", "advice": "增加药物剂量。",
                "professional_review": False,
            },
        )
        self.assertEqual(result["level"], "contact_clinician")
        self.assertTrue(result["professional_review"])
        self.assertEqual(result["reason"], ["模型输出触发处方药安全限制"])
        self.assertIn("请不要", result["advice"])

    def test_unavailable_model_uses_existing_triage_fallback(self):
        for risk, expected in [("normal", "home_followup"), ("attention", "contact_clinician")]:
            with self.subTest(risk=risk):
                result = tool_generate_visit_advice(
                    patient_state(risk_level=risk), call_json_model=lambda **kwargs: {},
                )
                self.assertEqual(result["level"], expected)
                self.assertTrue(result["continue_home_monitoring"])
                self.assertEqual(result["bp_context"]["current"], {"systolic": 145, "diastolic": 85})


class GuidanceAndRegistryTests(unittest.TestCase):
    def test_empty_retrieval_cache_invalidates_on_patient_or_knowledge_change(self):
        state = patient_state(retrieved_guidance=[])
        with patch("medical_tools.guidance.knowledge_revision", return_value="version-one") as revision:
            with patch("medical_tools.guidance.retrieve_medical_guidance", return_value=[]) as retrieve:
                tool_retrieve_medical_guidance(state, "同一问题")
                self.assertEqual(tool_retrieve_medical_guidance(state, "同一问题")["status"], "cached")
                self.assertEqual(retrieve.call_count, 1)
                state["bp_systolic"] = 162
                tool_retrieve_medical_guidance(state, "同一问题")
                self.assertEqual(retrieve.call_count, 2)
                revision.return_value = "version-two"
                tool_retrieve_medical_guidance(state, "同一问题")
                self.assertEqual(retrieve.call_count, 3)

    def test_execution_signatures_ignore_trace_but_advice_tracks_current_evidence(self):
        state = patient_state()
        before = execution_fingerprint(state, "ANSWER_PATIENT_QUESTION", "用药问题")
        state["trace"] = [{"time": "changed", "event": "ASK_USER"}]
        state["current_goal"] = "changed_goal"
        state["patient_memory"]["loaded_at"] = "another_timestamp"
        self.assertEqual(before, execution_fingerprint(state, "ANSWER_PATIENT_QUESTION", "用药问题"))
        state["retrieved_guidance"] = []
        self.assertNotEqual(before, execution_fingerprint(state, "ANSWER_PATIENT_QUESTION", "用药问题"))

    def test_nonmedication_question_is_passed_to_the_existing_advice_model(self):
        question = "家庭血压记录应该怎么整理？"
        model = Mock(return_value={"level": "home_followup", "advice": "请保留近期记录供医生评估。"})
        result = tool_generate_visit_advice(patient_state(latest_patient_question=question), call_json_model=model)
        self.assertIn(question, model.call_args.kwargs["user_prompt"])
        self.assertEqual(result["advice"], "请保留近期记录供医生评估。")

    def test_query_prioritizes_patient_question_and_includes_historical_context(self):
        state = patient_state(latest_patient_question="  漏服怎么办？  ")
        self.assertEqual(build_guidance_query(state), "漏服怎么办？")
        state["latest_patient_question"] = ""
        state["patient_memory"] = {
            "blood_pressure": {"last_7_days": {
                "measurement_count": 3, "measurement_days": 3,
                "systolic_avg": 150, "diastolic_avg": 90,
            }},
        }
        query = build_guidance_query(state)
        self.assertIn("3条", query)
        self.assertIn("150/90", query)
        self.assertIn("历史事实", query)

    def test_retrieval_updates_session_and_reuses_same_query(self):
        state = patient_state(retrieved_guidance=[])
        with patch("medical_tools.guidance.retrieve_medical_guidance", return_value=copy.deepcopy(EVIDENCE)) as retrieve:
            self.assertEqual(tool_retrieve_medical_guidance(state, " "), {
                "status": "empty_query", "query": "", "results": [],
            })
            first = tool_retrieve_medical_guidance(state, "  家庭血压监测  ")
            second = tool_retrieve_medical_guidance(state, "家庭血压监测")
            self.assertEqual(first["status"], "ok")
            self.assertEqual(second["status"], "cached")
            self.assertEqual(first["results"], second["results"])
            retrieve.assert_called_once_with("家庭血压监测", patient_state=state, top_k=4)
            self.assertEqual(state["knowledge_queries"], ["家庭血压监测"])
            self.assertEqual(state["knowledge_sources"][0]["source_id"], "test-source")

    def test_registry_routes_all_existing_tools_and_injects_model_without_mutating_global(self):
        self.assertEqual(set(TOOL_REGISTRY), {
            "ASSESS_RISK", "CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE",
            "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE", "CREATE_ALERT",
        })
        model = Mock(return_value={})
        registry = get_tool_registry(call_json_model=model)
        state = patient_state()
        self.assertEqual(registry["ASSESS_RISK"](state)["risk_level"], "normal")
        self.assertEqual(registry["CHECK_BP_HISTORY"](state)["record_count"], 1)
        self.assertEqual(registry["MEDICATION_GUIDANCE"](state)["status"], "ok")
        self.assertEqual(registry["GENERATE_VISIT_ADVICE"](state)["level"], "home_followup")
        self.assertEqual(model.call_count, 2)
        self.assertTrue(registry["CREATE_ALERT"](state)["family_alert_content"])
        self.assertIs(TOOL_REGISTRY["MEDICATION_GUIDANCE"], tool_medication_guidance)
        registry.pop("ASSESS_RISK")
        self.assertIn("ASSESS_RISK", get_tool_registry())

    def test_alert_messages_preserve_family_and_clinician_fields(self):
        result = tool_create_alert(patient_state(red_flags=["呼吸困难"], risk_reasons=["危险症状"]))
        self.assertEqual(set(result), {"family_alert_content", "clinician_alert_content"})
        self.assertIn("模拟患者", result["family_alert_content"])
        self.assertIn("145/85", result["clinician_alert_content"])
        self.assertIn("呼吸困难", result["family_alert_content"])


if __name__ == "__main__":
    unittest.main()

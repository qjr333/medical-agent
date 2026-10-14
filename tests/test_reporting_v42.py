"""V4.2 reports: grounded facts, real tool results, privacy and idempotence."""

import contextlib
import importlib
import io
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch


with patch.dict(os.environ, {"RAG_DISABLE_EMBEDDING": "1"}), contextlib.redirect_stdout(io.StringIO()):
    reporting = importlib.import_module("agent.clinical_report")


def fixture_fingerprint(state, action, query=None):
    return json.dumps({"action": action, "query": query,
                       "bp": [state.get("bp_systolic"), state.get("bp_diastolic")],
                       "medication": state.get("medication_status")}, sort_keys=True)


def complete_state():
    return {
        "patient_id": "report-fixture", "patient_name": "模拟患者", "age": 72,
        "conditions": ["高血压"], "trigger_type": "scheduled", "followup_reason": "例行随访",
        "followup_status": "completed", "completion_evaluation": {"outcome": "completed"},
        "bp_systolic": 150, "bp_diastolic": 90, "bp_trend": "rising",
        "bp_history": [{"systolic": 140, "diastolic": 80}, {"systolic": 150, "diastolic": 90}],
        "medication_status": "taken", "medication_detail": "处方隐私标记氨氯地平5 mg",
        "symptoms": ["头晕"], "denied_symptoms": [], "symptom_status": "reported",
        "red_flags": [], "denied_red_flags": ["胸痛", "呼吸困难"], "red_flag_status": "denied",
        "risk_level": "attention", "risk_reasons": ["本次需关注"],
        "required_topics": ["bp", "medication", "current_symptoms", "red_flags", "measurement_context"],
        "coverage": dict.fromkeys(["bp", "medication", "current_symptoms", "red_flags", "measurement_context"], True),
        "topic_status": {}, "deferred_topics": {}, "task_errors": [], "report": None,
        "knowledge_sources": [{"block_id": "unused-history", "source_title": "过去未使用的来源"}],
        "knowledge_queries": ["历史检索记录"], "trace": [{"event": "PRIVATE_FULL_TRACE"}],
        "v41_tool_results": [],
    }


def add_tool(state, action, data, *, query=None, sources=None, status="success"):
    result = {"tool": action, "status": status, "data": deepcopy(data),
              "sources": deepcopy(sources or []), "summary": "实际工具结果"}
    state["v41_tool_results"].append({"fingerprint": fixture_fingerprint(state, action, query),
                                    "query": query, "result": result})
    return result


def select_all_facts(_purpose, _prompt, payload):
    return {section: [fact["fact_id"] for fact in payload["facts"]
                      if section in fact["allowed_sections"]] for section in reporting.SECTIONS}


class ReportingV42Tests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(reporting, "execution_fingerprint", side_effect=fixture_fingerprint)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.state = complete_state()

    def test_preserves_legacy_fields_and_uses_actual_tool_analysis_and_sources(self):
        add_tool(self.state, "CHECK_BP_HISTORY", {"record_count": 2, "trend": "rising", "history": self.state["bp_history"]})
        source = {"block_id": "current-guide", "source_id": "guide", "source_title": "本次实际指南", "year": 2024}
        add_tool(self.state, "RETRIEVE_MEDICAL_GUIDANCE", {"results": [{**source, "text": "实际检索片段"}]}, query="家庭血压")
        add_tool(self.state, "GENERATE_VISIT_ADVICE", {"level": "contact_clinician", "advice": "请保留近期记录供医生评估。"}, sources=[source])
        report = reporting.build_clinical_report(self.state, model_call=select_all_facts)
        self.assertEqual(report["schema_version"], "4.2")
        self.assertEqual(report["completion_outcome"], "completed")
        self.assertTrue({"patient", "trigger", "blood_pressure", "medication", "symptoms", "risk",
                         "knowledge", "long_term_memory", "task_completion", "generated_at"} <= report.keys())
        self.assertEqual(report["blood_pressure"]["trend"], "rising")
        self.assertIn("2次实际记录", report["ai_summary"]["summary"])
        self.assertEqual(len(report["tool_analysis"]), 3)
        self.assertEqual(report["knowledge"]["queries"], ["家庭血压"])
        self.assertEqual([row["block_id"] for row in report["knowledge"]["sources"]], ["current-guide"])
        self.assertEqual(report["knowledge"]["sources"][0]["usage"], "used_by_medical_tool")
        self.assertNotIn("unused-history", json.dumps(report, ensure_ascii=False))
        self.assertEqual(report["doctor_view"]["medication"]["detail"], self.state["medication_detail"])

    def test_ai_only_organizes_allowed_backend_facts_and_returns_a_natural_summary(self):
        observed = []

        def model(purpose, prompt, payload):
            observed.append(payload)
            self.assertEqual(purpose, "REPORT_SUMMARY")
            self.assertIn("不能生成新的医学事实", prompt)
            selection = select_all_facts(purpose, prompt, payload)
            selection["summary_fact_ids"].reverse()
            return selection

        report = reporting.build_clinical_report(self.state, model_call=model)
        self.assertEqual(report["ai_summary"]["status"], "success")
        self.assertIn("150/90 mmHg", report["ai_summary"]["summary"])
        self.assertIn("患者报告当前不适：头晕", report["ai_summary"]["summary"])
        self.assertEqual(report["doctor_view"]["summary_kind"], "ai_organized_facts")
        self.assertNotIn("PRIVATE_FULL_TRACE", json.dumps(observed))
        self.assertNotIn("处方隐私标记", json.dumps(observed, ensure_ascii=False))

    def test_unknown_fact_ids_or_free_text_cannot_add_medical_claims(self):
        for returned in ({"summary_fact_ids": ["invented_diagnosis"], "doctor_attention_fact_ids": [], "family_assistance_fact_ids": []},
                         {"summary_fact_ids": [], "doctor_attention_fact_ids": [], "family_assistance_fact_ids": [],
                          "summary": "已确诊高血压急症，应增加药量。"}):
            with self.subTest(returned=returned):
                report = reporting.build_clinical_report(complete_state(), model_call=lambda *_: returned)
                self.assertEqual(report["ai_summary"]["status"], "invalid")
                self.assertNotIn("已确诊", report["ai_summary"]["summary"])
                self.assertNotIn("增加药量", json.dumps(report, ensure_ascii=False))
                self.assertEqual(report["doctor_view"]["summary_kind"], "recorded_facts")

    def test_family_summary_cannot_select_doctor_only_facts(self):
        def model(purpose, prompt, payload):
            result = select_all_facts(purpose, prompt, payload)
            result["family_assistance_fact_ids"] = ["medication_status"]
            return result

        report = reporting.build_clinical_report(self.state, model_call=model)
        self.assertEqual(report["ai_summary"]["status"], "invalid")
        family = json.dumps(report["family_view"], ensure_ascii=False)
        self.assertNotIn("处方隐私标记", family)
        self.assertNotIn("氨氯地平", family)
        self.assertNotIn("PRIVATE_FULL_TRACE", family)
        self.assertNotIn("bp_history", family)
        self.assertIn("原医嘱", family)

    def test_failed_model_has_deterministic_summary_and_explicit_failed_status(self):
        for result in (None, {}):
            with self.subTest(result=result):
                report = reporting.build_clinical_report(complete_state(), model_call=lambda *_: result)
                self.assertEqual(report["ai_summary"]["status"], "failed")
                self.assertIn("150/90", report["ai_summary"]["summary"])
                self.assertEqual(report["doctor_view"]["summary_kind"], "recorded_facts")
        model = Mock(side_effect=ValueError("private API key and patient text"))
        report = reporting.build_clinical_report(complete_state(), model_call=model)
        self.assertEqual(report["ai_summary"]["status"], "failed")
        self.assertNotIn("private API", json.dumps(report))

    def test_partial_outcome_keeps_unconfirmed_and_clarification_items_visible(self):
        self.state.update(followup_status="partially_completed", symptom_status="unknown", symptoms=[])
        self.state["coverage"]["current_symptoms"] = False
        self.state["topic_status"]["medication"] = "clarification_needed"
        self.state["completion_evaluation"] = {
            "outcome": "partial", "pending_items": ["patient_question"],
            "reason": "仍有信息未确认", "next_goal": "clarify_question", "model_status": "success",
        }
        report = reporting.build_clinical_report(self.state, model_call=select_all_facts)
        self.assertEqual(report["completion_outcome"], "partial")
        self.assertTrue({"current_symptoms", "medication", "patient_question"} <= set(report["unconfirmed_items"]))
        self.assertFalse(report["task_completion"]["all_required_information_confirmed"])
        self.assertIn("部分完成", report["ai_summary"]["summary"])
        self.assertIn("尚未完整确认", report["ai_summary"]["summary"])
        self.assertTrue(report["doctor_attention"])

    def test_model_cannot_drop_required_unknown_or_safety_facts(self):
        self.state.update(symptoms=[], symptom_status="unknown")

        def model(purpose, prompt, payload):
            result = select_all_facts(purpose, prompt, payload)
            result["summary_fact_ids"].remove("symptoms")
            return result

        report = reporting.build_clinical_report(self.state, model_call=model)
        self.assertEqual(report["ai_summary"]["status"], "invalid")
        self.assertIn("尚未完整确认", report["ai_summary"]["summary"])

    def test_emergency_report_never_waits_for_model_and_is_idempotent(self):
        self.state.update(followup_status="escalated", risk_level="high", red_flags=["胸痛"])
        model = Mock(side_effect=AssertionError("emergency must not wait for a model"))
        first = reporting.build_clinical_report(self.state, model_call=model)
        second = reporting.build_clinical_report(self.state, model_call=model)
        model.assert_not_called()
        self.assertIs(first, second)
        self.assertIs(self.state["report"], first)
        self.assertEqual(first["completion_outcome"], "escalated")
        self.assertEqual(first["ai_summary"]["status"], "not_requested")
        self.assertEqual(first["ai_summary"]["reason"], "safety_priority")
        self.assertIn("胸痛", first["ai_summary"]["summary"])
        self.assertIn("不要因等待报告延误", json.dumps(first["family_view"], ensure_ascii=False))

    def test_stale_tool_results_and_historical_retrieval_are_not_current_report_facts(self):
        old = {"block_id": "old-block", "source_id": "old-source", "source_title": "过期指南"}
        add_tool(self.state, "CHECK_BP_HISTORY", {"record_count": 99, "trend": "stable"})
        add_tool(self.state, "RETRIEVE_MEDICAL_GUIDANCE", {"results": [{**old, "text": "旧证据"}]}, query="旧问题")
        add_tool(self.state, "MEDICATION_GUIDANCE", {"message": "过期用药建议", "needs_followup": False}, sources=[old])
        self.state["medication_guidance"] = {"message": "过期用药建议"}
        self.state["visit_advice"] = {"advice": "过期就医建议"}
        self.state["bp_systolic"] = 155
        report = reporting.build_clinical_report(self.state, model_call=select_all_facts)
        self.assertEqual(report["tool_analysis"], [])
        self.assertEqual(report["knowledge"], {"queries": [], "sources": []})
        self.assertEqual(report["blood_pressure"]["trend"], "unknown")
        self.assertIsNone(report["medication_guidance"])
        self.assertIsNone(report["visit_advice"])
        self.assertNotIn("99次", report["ai_summary"]["summary"])
        self.assertNotIn("过期", json.dumps(report, ensure_ascii=False))

    def test_completed_report_is_generated_once_and_snapshots_do_not_mutate(self):
        model = Mock(side_effect=select_all_facts)
        report = reporting.build_clinical_report(self.state, model_call=model)
        timestamp = report["generated_at"]
        self.state["bp_history"].append({"systolic": 160, "diastolic": 95})
        again = reporting.build_clinical_report(self.state, model_call=model)
        self.assertIs(report, again)
        self.assertEqual(again["generated_at"], timestamp)
        self.assertEqual(len(report["blood_pressure"]["history"]), 2)
        model.assert_called_once()

    def test_no_model_request_is_explicit_and_only_retrieved_evidence_is_labeled(self):
        source = {"block_id": "retrieved-only", "source_id": "guide", "source_title": "仅检索来源"}
        add_tool(self.state, "RETRIEVE_MEDICAL_GUIDANCE", {"results": [source]}, query="家庭监测")
        report = reporting.build_clinical_report(self.state)
        self.assertEqual(report["ai_summary"]["status"], "not_requested")
        self.assertEqual(report["ai_summary"]["reason"], "model_not_requested")
        self.assertEqual(report["knowledge"]["sources"][0]["usage"], "retrieved")
        self.assertIn("0个被医学工具采用", report["ai_summary"]["summary"])

    def test_family_summary_and_abnormal_alerts_use_real_facts_without_doctor_notes(self):
        self.state["medication_status"] = "missed"
        add_tool(self.state, "CHECK_BP_HISTORY", {"record_count": 2, "trend": "rising"})
        add_tool(self.state, "GENERATE_VISIT_ADVICE", {
            "level": "contact_clinician", "advice": "医生备注私密标记：请核对处方细节。",
        })
        report = reporting.build_clinical_report(self.state, model_call=select_all_facts)
        family = report["family_view"]
        self.assertIn("150/90 mmHg", family["summary"])
        self.assertIn("漏服", family["summary"])
        self.assertIn("头晕", family["summary"])
        alerts = "".join(family["abnormal_alerts"])
        self.assertIn("需要关注", alerts)
        self.assertIn("漏服", alerts)
        self.assertIn("趋势为上升", alerts)
        self.assertIn("头晕", alerts)
        family_text = json.dumps(family, ensure_ascii=False)
        self.assertNotIn("医生备注私密标记", family_text)
        self.assertNotIn("处方隐私标记", family_text)
        self.assertNotIn("PRIVATE_FULL_TRACE", family_text)
        self.assertIn("医生备注私密标记", report["doctor_view"]["ai_summary"]["summary"])

    def test_family_alerts_do_not_invent_thresholds_or_label_not_due_as_missed(self):
        self.state.update(risk_level="normal", symptoms=[], denied_symptoms=["头晕"],
                          symptom_status="denied", medication_status="not_taken_yet", medication_timing="not_due")
        report = reporting.build_clinical_report(self.state, model_call=select_all_facts)
        self.assertEqual(report["family_view"]["abnormal_alerts"], [])
        self.assertIn("尚未到服药时间", report["family_view"]["summary"])
        self.assertNotIn("漏服", report["family_view"]["summary"])
        self.assertNotIn("血压异常", report["family_view"]["summary"])

    def test_family_pending_items_are_chinese_and_free_model_notes_are_not_medical_facts(self):
        private_note = "模型原文私密原因：怀疑未知诊断"
        self.state["completion_evaluation"] = {
            "outcome": "partial", "reason": private_note,
            "pending_items": ["monitoring_plan", "medication_details", "tool:RETRIEVE_MEDICAL_GUIDANCE", private_note],
            "next_goal": "review_unconfirmed_information", "model_status": "success",
        }
        report = reporting.build_clinical_report(self.state, model_call=select_all_facts)
        self.assertEqual(report["family_view"]["unconfirmed_items"], [
            "恢复血压监测的计划", "药物名称与原医嘱核对", "医学依据检索尚未完成",
        ])
        family = json.dumps(report["family_view"], ensure_ascii=False)
        self.assertNotIn("monitoring_plan", family)
        self.assertNotIn("medication_details", family)
        self.assertNotIn("RETRIEVE_MEDICAL_GUIDANCE", family)
        self.assertNotIn(private_note, family)
        self.assertNotIn(private_note, report["ai_summary"]["summary"])
        self.assertNotIn("reason", report["completion_evaluation"])
        self.assertNotIn("reason", report["doctor_view"]["completion_evaluation"])
        # Unconstrained pending prose is retained only as planning metadata.
        self.assertIn(private_note, report["completion_evaluation"]["pending_items"])

    def test_completion_metadata_keeps_only_verified_patient_objective_evidence(self):
        quote = "我明天会量血压，并记录血压读数"
        self.state["latest_patient_message"] = quote
        self.state["completion_evaluation"] = {
            "outcome": "completed", "reason": "未经约束的诊断性模型原因",
            "pending_items": [], "next_goal": "", "model_status": "success",
            "objective_evidence": {"monitoring_plan": quote, "unregistered": "额外模型内容"},
        }
        report = reporting.build_clinical_report(self.state)
        self.assertEqual(report["completion_evaluation"], {
            "outcome": "completed", "pending_items": [], "next_goal": "", "model_status": "success",
            "objective_evidence": {"monitoring_plan": quote},
        })
        self.assertNotIn("未经约束", json.dumps(report, ensure_ascii=False))
        self.assertNotIn("额外模型内容", json.dumps(report, ensure_ascii=False))
        self.assertNotIn(quote, report["family_view"]["summary"])

    def test_fabricated_or_negated_objective_evidence_is_not_kept(self):
        for message, evidence in [("我最近没有测量", "我明天会量血压"),
                                  ("我不想明天测血压", "我不想明天测血压")]:
            with self.subTest(message=message):
                state = complete_state()
                state["latest_patient_message"] = message
                state["completion_evaluation"] = {
                    "outcome": "completed", "objective_evidence": {"monitoring_plan": evidence},
                }
                report = reporting.build_clinical_report(state)
                self.assertNotIn("objective_evidence", report["completion_evaluation"])

    def test_explicit_current_symptom_denial_does_not_require_named_denial_items(self):
        self.state.update(symptoms=[], denied_symptoms=[], symptom_status="denied",
                          red_flags=[], denied_red_flags=[], red_flag_status="denied")
        report = reporting.build_clinical_report(self.state, model_call=select_all_facts)
        summary = report["ai_summary"]["summary"]
        self.assertIn("患者本次否认当前不适", summary)
        self.assertNotIn("当前不适尚未完整确认", summary)
        self.assertIn("危险表现尚未完整确认", summary)
        self.assertIn("患者本次否认当前不适", report["family_view"]["summary"])
        self.assertEqual(report["family_view"]["abnormal_alerts"], [
            "本次安全评估标记为需要关注，需结合当前信息由医生判断。",
        ])


if __name__ == "__main__":
    unittest.main()

"""Presentation regressions use recorded events, never a synthetic UI flow."""

import json
import unittest
from copy import deepcopy

from agent.trace_view import present_trace


class TracePresentationV50Tests(unittest.TestCase):
    def test_empty_execution_does_not_invent_pipeline_steps(self):
        self.assertEqual(present_trace({}), [])
        self.assertEqual(present_trace({"trace": []}), [])
        self.assertEqual(present_trace({"trace": [{"event": "UNRECOGNIZED", "detail": {"message": "secret"}}]}), [])

    def test_recorded_steps_keep_order_ids_and_times(self):
        state = {"trace": [
            {"time": "2026-10-13T01:00:00+00:00", "event": "USER_MESSAGE", "detail": {"message": "患者原话"}},
            {"time": "2026-10-13T01:00:01+00:00", "event": "UNRECOGNIZED", "detail": {}},
            {"time": "2026-10-13T01:00:02+00:00", "event": "LLM_CALL", "detail": {"purpose": "EXTRACT", "called": True}},
            {"time": "2026-10-13T01:00:03+00:00", "event": "LLM_DECISION", "detail": {"action": "CHECK_BP_HISTORY"}},
            {"time": "2026-10-13T01:00:04+00:00", "event": "V4_TOOL_CALL", "detail": {"tool": "CHECK_BP_HISTORY", "executed": True}},
        ]}
        trace = present_trace(state)
        self.assertEqual([entry["id"] for entry in trace], [0, 2, 3, 4])
        self.assertEqual([entry["stage"] for entry in trace], ["patient_input", "information_extraction", "planner", "medical_tool"])
        self.assertEqual(trace[-1]["time"], state["trace"][-1]["time"])
        self.assertEqual(trace[-1]["detail"], {"tool": "CHECK_BP_HISTORY", "executed": True})

    def test_patient_content_secrets_prompts_and_reasoning_are_removed_by_default(self):
        secret = "sk-private-key-do-not-export"
        patient = "真实患者身份证及住址"
        thought = "private-chain-of-thought"
        state = {"trace": [
            {"event": "USER_MESSAGE", "detail": {"message": patient, "api_key": secret}},
            {"event": "STATE_UPDATE", "detail": {"patch": {"patient_name": patient}, "coverage": {"bp": True}}},
            {"event": "LLM_CALL", "detail": {"purpose": "EXTRACT", "called": True, "ok": False,
             "deepseek_called": True, "duration_ms": 12, "prompt": patient, "request": secret,
             "reasoning_content": thought, "provider_events": [{"status": "started", "error": secret},
                                                                {"status": "error", "response": patient}]}},
            {"event": "LLM_DECISION", "detail": {"action": "ASK_USER", "topic": "medication", "reason": thought}},
            {"event": "AGENT_RESPONSE", "detail": {"message": patient, "system_prompt": thought}},
        ]}
        trace = present_trace(state)
        exported = json.dumps(trace, ensure_ascii=False)
        for sensitive in (secret, patient, thought, "provider_events", "reasoning_content", "patch", "prompt"):
            self.assertNotIn(sensitive, exported)
        self.assertEqual(trace[2]["detail"]["provider_request_count"], 1)
        self.assertEqual(trace[2]["detail"]["duration_ms"], 12)

    def test_fictional_input_and_reply_are_explicitly_enabled_and_bounded(self):
        message = "虚构患者" * 1000
        state = {"trace": [
            {"event": "USER_MESSAGE", "detail": {"message": message, "reasoning_content": "private"}},
            {"event": "AGENT_RESPONSE", "detail": {"message": "虚构回复"}},
            {"event": "LLM_CALL", "detail": {"purpose": "PLAN", "message": "raw-model-prompt"}},
        ]}
        ordinary = present_trace(state)
        fictional = present_trace(state, fictional=True)
        self.assertNotIn("message", ordinary[0]["detail"])
        self.assertEqual(fictional[0]["detail"]["message"], message[:3000])
        self.assertEqual(fictional[1]["detail"]["message"], "虚构回复")
        self.assertNotIn("message", fictional[2]["detail"])
        self.assertNotIn("private", json.dumps(fictional))

    def test_provider_count_uses_actual_request_start_records(self):
        state = {"trace": [
            {"event": "LLM_CALL", "detail": {"purpose": "PLAN", "called": True, "ok": True,
             "deepseek_called": True, "provider_events": [{"status": "started"}, {"status": "error"},
                                                          {"status": "started"}, {"status": "success"}]}},
            {"event": "LLM_CALL", "detail": {"purpose": "SIMULATED_PATIENT", "called": True, "ok": True,
                                             "deepseek_called": False, "provider_events": []}},
            {"event": "LLM_CALL", "detail": {"purpose": "COMPLETE", "called": False, "ok": False}},
        ]}
        trace = present_trace(state)
        self.assertEqual([entry["detail"]["provider_request_count"] for entry in trace], [2, 0, 0])
        self.assertEqual(trace[1]["stage"], "simulated_patient")
        self.assertFalse(trace[1]["detail"]["deepseek_called"])

    def test_rag_sources_preserve_returned_identifiers_and_drop_documents(self):
        state = {"trace": [{"event": "RAG_RETRIEVAL", "detail": {"status": "success", "retrieved_count": 2,
            "query": "patient-private-question", "api_key": "secret-key", "sources": [
                {"block_id": "actual-block", "source_id": "actual-source", "source_title": "指南", "year": 2026,
                 "source_pages": [1, "2-3", "页 4", "private-address", {"patient": "hidden"}],
                 "text": "private-full-document", "metadata": {"api_key": "secret-key"}},
                {"block_id": "actual-second", "source_pages": "6–7", "text": "private-full-document"},
                "not-a-source",
            ]}}]}
        result = present_trace(state)
        detail = result[0]["detail"]
        self.assertEqual(detail["retrieved_count"], 2)
        self.assertEqual(detail["sources"], [
            {"block_id": "actual-block", "source_id": "actual-source", "source_title": "指南", "year": 2026,
             "source_pages": [1, "2-3", "页 4"]}, {"block_id": "actual-second", "source_pages": "6–7"}])
        for private in ("patient-private-question", "secret-key", "private-full-document", "private-address", "hidden"):
            self.assertNotIn(private, json.dumps(result, ensure_ascii=False))

    def test_nested_values_cannot_bypass_allowlisted_scalar_details(self):
        trace = present_trace({"trace": [{"event": "LLM_DECISION", "detail": {
            "action": {"reasoning_content": "hidden"}, "status": ["secret"], "tool": "CHECK_BP_HISTORY",
            "called": True, "count": 2, "source_count": 1, "unknown": "ignored"}}]})
        self.assertEqual(trace[0]["detail"], {"tool": "CHECK_BP_HISTORY", "called": True, "count": 2, "source_count": 1})

    def test_presentation_does_not_mutate_evidence_or_claim_external_delivery(self):
        state = {"trace": [
            {"event": "RAG_RETRIEVAL", "detail": {"status": "success", "retrieved_count": 1,
             "sources": [{"block_id": "actual", "source_pages": [1, 2]}]}},
            {"event": "MESSAGE_DELIVERY", "detail": {"channel": "station", "recipient_count": 2,
             "status": "delivered", "external_delivery": "not_configured"}},
        ]}
        original = deepcopy(state)
        trace = present_trace(state)
        trace[0]["detail"]["sources"][0]["source_pages"].append(99)
        self.assertEqual(state, original)
        self.assertEqual(trace[1]["detail"]["external_delivery"], "not_configured")
        self.assertEqual(trace[1]["stage"], "notification")


if __name__ == "__main__":
    unittest.main()

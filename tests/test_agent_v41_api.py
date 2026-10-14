"""V4.1 API integration tests with scripted models and temporary patient storage.

Only the temporary SQLite fixture is written. DeepSeek and retrieval are mocked;
the real API, state merging, registered tools, safety layer and persistence run.
"""

import contextlib
import copy
import importlib
import io
import json
import os
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


def ask(question, topic="current_symptoms", **extra):
    return {
        "action": "ASK_USER", "question": question, "topic": topic,
        "reason": "模拟模型依据本轮目标提出问题", **extra,
    }


class ScriptedModel:
    """Collect actual planner inputs without relying on hidden model reasoning."""

    def __init__(self):
        self.plans = deque()
        self.planner_inputs = []
        self.extract_inputs = []
        self.extraction = {"answer_status": "answered"}

    def __call__(self, *, system_prompt, user_prompt, temperature):
        payload = json.loads(user_prompt)
        if "trigger_event" in payload:
            self.planner_inputs.append(payload)
            if not self.plans:
                raise AssertionError("Mock planner received an unexpected additional call")
            return self.plans.popleft()
        if "patient_message" in payload:
            self.extract_inputs.append(payload)
            return copy.deepcopy(self.extraction)
        raise AssertionError("Unexpected model module in API test")


class AgentV41APIRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1",
        }), contextlib.redirect_stdout(io.StringIO()):
            cls.backend = importlib.import_module("app")
        cls.database = importlib.import_module("database.connection")
        cls.rag = importlib.import_module("rag")
        cls.guidance = importlib.import_module("medical_tools.guidance")
        cls.llm = importlib.import_module("agent.followup_agent")

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.dict(os.environ, {"LLM_AGENT_V4": "1"}))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(self.database, "DATA_DIR", directory))
        self.stack.enter_context(patch.object(self.database, "DB_PATH", directory / "v41.db"))
        self.stack.enter_context(patch.object(self.rag, "DISABLE_EMBEDDING", True))
        self.model = ScriptedModel()
        self.model_mock = self.stack.enter_context(patch.object(
            self.backend, "call_json_model", side_effect=self.model,
        ))
        self.retrieval = self.stack.enter_context(patch.object(
            self.guidance, "retrieve_medical_guidance", return_value=[],
        ))
        self.stack.enter_context(patch.object(self.backend, "DEMO_PATIENT", {
            "patient_id": "v41-api-fixture", "patient_name": "模拟患者",
            "age": 72, "conditions": ["高血压"],
        }))
        self.stack.enter_context(patch.object(self.backend, "DEMO_BP_HISTORY", [
            {"label": "模拟记录1", "systolic": 142, "diastolic": 86},
            {"label": "模拟记录2", "systolic": 150, "diastolic": 91},
            {"label": "模拟记录3", "systolic": 160, "diastolic": 96},
        ]))
        configs = copy.deepcopy(self.backend.TRIGGER_CONFIGS)
        configs["scheduled"].update(bp_systolic=None, bp_diastolic=None)
        configs["high_bp"].update(bp_systolic=165, bp_diastolic=100)
        self.stack.enter_context(patch.object(self.backend, "TRIGGER_CONFIGS", configs))
        self.backend.sessions.clear()
        self.addCleanup(self.backend.sessions.clear)
        self.http = self.stack.enter_context(TestClient(self.backend.app))

    def query(self, sql, parameters=()):
        connection = self.database.get_connection()
        try:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]
        finally:
            connection.close()

    def trigger(self, kind="scheduled"):
        response = self.http.post("/api/trigger", json={"trigger_type": kind})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assert_frontend_contract(body)
        return body

    def chat(self, session_id, message, expected=200):
        response = self.http.post("/api/chat", json={
            "session_id": session_id, "message": message,
        })
        self.assertEqual(response.status_code, expected, response.text)
        body = response.json()
        if expected == 200:
            self.assert_frontend_contract(body)
            self.assertEqual(body["session_id"], session_id)
        return body

    def assert_frontend_contract(self, body):
        self.assertTrue({"session_id", "message", "state", "trace"} <= body.keys())
        self.assertIsInstance(body["session_id"], str)
        self.assertTrue(body["message"])
        self.assertIsInstance(body["message"], str)
        self.assertIsInstance(body["state"], dict)
        self.assertEqual(body["trace"], body["state"]["trace"])

    def assert_persisted(self, body):
        session_id = body["session_id"]
        rows = self.query(
            "SELECT state_json,status FROM followup_sessions WHERE session_id=?", (session_id,),
        )
        self.assertEqual(json.loads(rows[0]["state_json"]), body["state"])
        self.assertEqual(rows[0]["status"], body["state"]["followup_status"])
        events = self.query("SELECT event_name FROM agent_events WHERE session_id=?", (session_id,))
        self.assertEqual(len(events), len(body["trace"]))

    def test_trigger_contract_and_trigger_objectives_survive_sqlite_reload(self):
        objectives = []
        for kind in ("scheduled", "high_bp", "missed_measurement"):
            with self.subTest(kind=kind):
                self.model.plans.append(ask("请说说最近身体情况：" + kind))
                body = self.trigger(kind)
                state = body["state"]
                self.assertEqual(state["trigger_type"], kind)
                self.assertEqual(state["followup_status"], "waiting_for_user")
                event = self.model.planner_inputs[-1]["trigger_event"]
                self.assertEqual(event["type"], kind)
                self.assertTrue(event["objective"])
                objectives.append(event["objective"])
                self.assert_persisted(body)
                self.backend.sessions.clear()
                restored = self.http.get("/api/debug/session/" + body["session_id"])
                self.assertEqual(restored.status_code, 200)
                self.assertEqual(restored.json(), state)
        self.assertEqual(len(set(objectives)), 3)
        self.retrieval.assert_not_called()

    def test_history_and_rag_tools_feed_actual_results_to_the_next_plan(self):
        self.model.plans.append(ask("最近血压如何？", "bp"))
        initial = self.trigger()
        evidence = [{
            "block_id": "mock-api-evidence", "source_id": "mock-source",
            "source_title": "模拟家庭监测依据", "year": 2026, "source_pages": "1",
            "text": "家庭测量应记录时间和测量条件，服药方案需遵照原医嘱。",
        }]
        self.retrieval.return_value = evidence
        self.model.plans.extend([
            {"action": "CHECK_BP_HISTORY", "arguments": {}, "reason": "查看真实血压记录"},
            {"action": "RETRIEVE_MEDICAL_GUIDANCE",
             "arguments": {"query": "规范服药后血压仍高 家庭测量条件"},
             "reason": "需要家庭测量和用药管理依据"},
            ask("今天测量前是否安静休息？", "measurement_context",
                reply="已有记录显示血压上升；家庭测量应记录时间和测量条件，服药遵照原医嘱。"),
        ])
        body = self.chat(initial["session_id"], "最近一周血压一直偏高，今天165/100，药按时吃了，不知道为什么")
        self.assertEqual(body["state"]["followup_status"], "waiting_for_user")
        self.assertEqual(body["state"]["medication_status"], "taken")
        self.assertEqual(body["state"]["bp_analysis"]["record_count"], 4)
        self.assertEqual(body["state"]["bp_analysis"]["recent_7"]["systolic_avg"], 154.2)
        after_history = self.model.planner_inputs[-2]["tool_results_this_turn"]
        history_result = next(item for item in after_history if item.get("tool") == "CHECK_BP_HISTORY")
        self.assertEqual(history_result["status"], "success")
        self.assertEqual(history_result["data"]["record_count"], 4)
        self.assertEqual(history_result["data"]["history"][-1]["systolic"], 165)
        after_rag = self.model.planner_inputs[-1]["tool_results_this_turn"]
        rag_result = next(item for item in after_rag if item.get("tool") == "RETRIEVE_MEDICAL_GUIDANCE")
        self.assertEqual(rag_result["status"], "success")
        self.assertIn(evidence[0]["text"], json.dumps(rag_result, ensure_ascii=False))
        self.assertTrue(rag_result["sources"])
        self.retrieval.assert_called_once()
        self.assertIn("家庭测量", body["message"])
        self.assertFalse(self.backend.contains_unsafe_medication_instruction(body["message"]))
        self.assert_persisted(body)

    def test_danger_skips_extraction_and_planning_then_closes_ordinary_chat(self):
        self.model.plans.append(ask("今天服药了吗？", "medication"))
        initial = self.trigger()
        self.model_mock.reset_mock()
        self.model_mock.side_effect = AssertionError("危险症状处理不得等待模型")
        body = self.chat(initial["session_id"], "我现在胸痛，还有点喘不上气")
        self.model_mock.assert_not_called()
        self.retrieval.assert_not_called()
        self.assertEqual(body["state"]["risk_level"], "high")
        self.assertEqual(body["state"]["followup_status"], "escalated")
        self.assertTrue({"胸痛", "呼吸困难"} <= set(body["state"]["red_flags"]))
        self.assertIn("立即", body["message"])
        self.assertTrue(body["alerts"])
        self.assertTrue(any(event["event"] == "SAFETY_OVERRIDE" for event in body["trace"]))
        self.assert_persisted(body)
        self.backend.sessions.clear()
        self.chat(body["session_id"], "我已经吃药了", expected=409)

    def test_tool_failure_is_feedback_and_never_a_successful_rag_call(self):
        self.model.plans.append(ask("您想了解哪方面？"))
        initial = self.trigger()
        self.retrieval.side_effect = TimeoutError("private retrieval credential")
        self.model.plans.extend([
            {"action": "RETRIEVE_MEDICAL_GUIDANCE",
             "arguments": {"query": "家庭测压规范"}, "reason": "需要测压依据"},
            ask("您使用的是上臂式血压计吗？", "measurement_context",
                reply="本次检索暂时失败，无法据此提供已核实的指南出处。"),
        ])
        body = self.chat(initial["session_id"], "家庭血压应该怎么测？")
        results = self.model.planner_inputs[-1]["tool_results_this_turn"]
        failure = next(item for item in results if item.get("tool") == "RETRIEVE_MEDICAL_GUIDANCE")
        self.assertEqual(failure["status"], "error")
        self.assertFalse(body["state"]["retrieved_guidance"])
        self.assertEqual(body["state"]["followup_status"], "waiting_for_user")
        self.assertNotIn("private retrieval credential", json.dumps(body, ensure_ascii=False))
        self.assertTrue(any(event["event"] == "TOOL_ERROR" for event in body["trace"]))
        self.assert_persisted(body)

    def test_internal_budget_is_saved_and_next_patient_turn_can_resume(self):
        self.model.plans.append({
            "action": "CHECK_BP_HISTORY", "arguments": {}, "reason": "查看真实记录",
        })

        def one_step(state):
            return self.llm.run_followup_agent(state, call_json_model=self.backend.call_json_model,
                                         max_steps=1)

        with patch.object(self.backend, "run_agent", side_effect=one_step):
            body = self.trigger()
        self.assertEqual(body["state"]["followup_status"], "waiting_for_user")
        self.assertTrue(body["state"]["loop_state"]["reason"])
        self.assertFalse(body.get("report"))
        self.assertFalse(self.query("SELECT * FROM reports"))
        self.assert_persisted(body)
        self.backend.sessions.clear()
        self.model.plans.append(ask("今天降压药是否按原医嘱服用？", "medication"))
        resumed = self.chat(body["session_id"], "我目前没有不舒服，可以继续")
        self.assertEqual(resumed["state"]["followup_status"], "waiting_for_user")
        self.assertIn("按原医嘱", resumed["message"])
        self.assertFalse(resumed.get("report"))
        self.assert_persisted(resumed)

    def test_model_failure_is_persisted_without_fabricated_tool_or_completion(self):
        self.model_mock.side_effect = TimeoutError("private model credential")
        body = self.trigger()
        self.assertEqual(body["state"]["followup_status"], "interrupted")
        self.assertFalse(body.get("report"))
        self.assertTrue(any(event["event"] == "LLM_CALL" and not event["detail"].get("ok")
                            for event in body["trace"]))
        self.assertFalse(any(event["event"] in {"V4_TOOL_CALL", "FINISH"} for event in body["trace"]))
        self.assertNotIn("private model credential", json.dumps(body, ensure_ascii=False))
        self.assert_persisted(body)
        self.backend.sessions.clear()
        self.model_mock.side_effect = self.model
        self.model.plans.append(ask("最近一次血压是多少？", "bp"))
        resumed = self.chat(body["session_id"], "继续")
        self.assertEqual(resumed["state"]["followup_status"], "waiting_for_user")

    def test_many_patient_turns_do_not_force_finish(self):
        self.model.plans.append(ask("请补充第0项信息"))
        body = self.trigger()
        self.model.extraction = {"answer_status": "unclear"}
        for number in range(1, 13):
            self.model.plans.append(ask("请补充第%d项信息" % number))
            body = self.chat(body["session_id"], "我还有第%d项情况要说明" % number)
            self.assertEqual(body["state"]["followup_status"], "waiting_for_user")
            self.assertFalse(body.get("report"))
        self.assertFalse(self.query("SELECT * FROM reports"))
        self.assert_persisted(body)

    def test_frontend_request_errors_keep_compatible_detail_messages(self):
        invalid = self.http.post("/api/trigger", json={"trigger_type": "unknown"})
        self.assertEqual(invalid.status_code, 400)
        self.assertTrue(invalid.json()["detail"])
        self.chat("unknown-session", "继续", expected=404)
        self.model.plans.append(ask("最近身体如何？"))
        body = self.trigger()
        invalid = self.chat(body["session_id"], "  ", expected=400)
        self.assertTrue(invalid["detail"])
        for path in ("/api/demo", "/api/db-status", "/api/patients/v41-api-fixture/history",
                     "/api/patients/v41-api-fixture/memory"):
            with self.subTest(path=path):
                self.assertEqual(self.http.get(path).status_code, 200)


if __name__ == "__main__":
    unittest.main()

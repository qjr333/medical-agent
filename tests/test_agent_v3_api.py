"""Planner/Loop V3 route regressions, with Mock models and temporary SQLite.

These use the real extractor, risk rules, planner, tools and repositories. No
production database, patient JSON, knowledge cache or API credentials are used.
"""

import contextlib
import copy
import importlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class AgentV3APIRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1",
        }), contextlib.redirect_stdout(io.StringIO()):
            cls.backend = importlib.import_module("app")
        cls.database = importlib.import_module("database.connection")
        cls.rag = importlib.import_module("rag")
        cls.guidance = importlib.import_module("medical_tools.guidance")
        cls.registry = importlib.import_module("medical_tools.registry")
        cls.runner = importlib.import_module("agent.runner")

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.object(self.database, "DATA_DIR", Path(directory)))
        self.stack.enter_context(patch.object(self.database, "DB_PATH", Path(directory) / "v3.db"))
        self.model = self.stack.enter_context(patch.object(
            self.backend, "call_json_model", return_value={},
        ))
        self.stack.enter_context(patch.object(self.rag, "DISABLE_EMBEDDING", True))
        # Explicit fixtures keep ordinary and abnormal BP scenarios independent
        # of the user's demo JSON and its current scheduled measurement.
        configs = copy.deepcopy(self.backend.TRIGGER_CONFIGS)
        configs["scheduled"].update(bp_systolic=150, bp_diastolic=95)
        configs["high_bp"].update(bp_systolic=162, bp_diastolic=96)
        self.stack.enter_context(patch.object(self.backend, "TRIGGER_CONFIGS", configs))
        stable_history = [{"label": f"模拟历史{i}", "systolic": 150, "diastolic": 95} for i in range(3)]
        self.stack.enter_context(patch.object(self.backend, "DEMO_BP_HISTORY", stable_history))
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
        self.assertTrue({"message", "state", "trace", "session_id"} <= body.keys())
        return body

    def chat(self, sid, message, expected=200):
        response = self.http.post("/api/chat", json={"session_id": sid, "message": message})
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    def assert_persisted(self, body):
        row = self.query("SELECT state_json,status FROM followup_sessions WHERE session_id=?", (body["session_id"],))
        self.assertEqual(json.loads(row[0]["state_json"]), body["state"])
        self.assertEqual(row[0]["status"], body["state"]["followup_status"])
        events = self.query("SELECT * FROM agent_events WHERE session_id=?", (body["session_id"],))
        self.assertEqual(len(events), len(body["trace"]))

    def test_normal_conversation_completes_with_one_report_and_closed_chat(self):
        initial = self.trigger("missed_measurement")
        sid = initial["session_id"]
        questions = [initial["state"]["question_context"]["kind"]]
        for message in ("安静坐着休息5分钟后血压125/78", "今天已经按时吃药了", "都没有不舒服"):
            body = self.chat(sid, message)
            if body["state"]["followup_status"] == "waiting_for_user":
                questions.append(body["state"]["question_context"]["kind"])
        state = body["state"]
        self.assertEqual(state["followup_status"], "completed")
        self.assertTrue(all(state["coverage"].values()))
        self.assertEqual(state["medication_status"], "taken")
        self.assertEqual(state["symptom_status"], "denied")
        self.assertEqual(len(questions), len(set(questions)))
        self.assertEqual(len(self.query("SELECT * FROM reports WHERE session_id=?", (sid,))), 1)
        self.assert_persisted(body)
        self.backend.sessions.clear()
        self.chat(sid, "谢谢", expected=409)
        self.assertEqual(len(self.query("SELECT * FROM reports WHERE session_id=?", (sid,))), 1)

    def test_abnormal_bp_prioritizes_unknown_symptoms_before_medication(self):
        body = self.trigger("high_bp")
        state = body["state"]
        self.assertEqual(state["risk_level"], "attention")
        self.assertEqual(state["current_goal"], "check_symptoms")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["red_flags"])
        self.assertEqual(state["last_action"], "ASK_USER")
        self.assertIn("胸痛", body["message"])
        self.assert_persisted(body)
        answer = self.chat(body["session_id"], "都没有不舒服")
        self.assertTrue(answer["state"]["coverage"]["red_flags"])
        self.assertEqual(answer["state"]["current_goal"], "check_medication")

    def test_historical_missed_dose_is_memory_not_current_medication(self):
        initial = self.trigger("missed_measurement")
        prior = self.chat(initial["session_id"], "安静休息后血压145/85，昨天忘记吃药了")
        self.assertEqual(prior["state"]["medication_status"], "unknown")
        fresh = self.trigger()
        state = fresh["state"]
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["symptoms"], [])
        self.assertEqual(state["red_flags"], [])
        self.assertGreaterEqual(state["patient_memory"]["medication"]["last_7_days"]["issue_report_count"], 1)
        self.assertEqual(state["current_goal"], "check_medication")

    def test_not_taken_yet_remains_a_clarification_then_advances(self):
        initial = self.trigger()
        first = self.chat(initial["session_id"], "还没吃")
        self.assertEqual(first["state"]["medication_status"], "not_taken_yet")
        self.assertEqual(first["state"]["current_goal"], "clarify_medication_timing")
        self.assertNotEqual(first["message"], initial["message"])
        second = self.chat(initial["session_id"], "还没到吃药时间")
        self.assertTrue(second["state"]["coverage"]["medication"])
        self.assertEqual(second["state"]["medication_status"], "not_taken_yet")
        self.assertNotEqual(second["state"]["current_goal"], "clarify_medication_timing")
        self.assertFalse(self.query("SELECT * FROM medication_records WHERE status='missed'"))
        self.assert_persisted(second)

    def test_danger_during_followup_interrupts_tools_and_stays_closed_after_reload(self):
        initial = self.trigger()
        tools = {name: Mock(side_effect=function) for name, function in
                 self.registry.get_tool_registry(call_json_model=lambda **kwargs: {}).items()}
        with patch.object(self.runner, "get_tool_registry", return_value=tools):
            body = self.chat(initial["session_id"], "没有胸痛，但是现在喘不上气")
        self.assertEqual(body["state"]["followup_status"], "escalated")
        self.assertEqual(body["state"]["risk_level"], "high")
        self.assertIn("呼吸困难", body["state"]["red_flags"])
        self.assertTrue(body["alerts"])
        for name in ("CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            tools[name].assert_not_called()
        tools["ASSESS_RISK"].assert_called_once()
        self.assert_persisted(body)
        self.backend.sessions.clear()
        self.chat(initial["session_id"], "已经吃了药", expected=409)
        stored = self.query("SELECT status FROM followup_sessions WHERE session_id=?", (initial["session_id"],))
        self.assertEqual(stored[0]["status"], "escalated")
        self.assertEqual(len(self.query("SELECT * FROM reports WHERE session_id=?", (initial["session_id"],))), 1)

    def test_trace_decisions_explain_priority_and_survive_sqlite_reload(self):
        body = self.trigger("high_bp")
        selected = [event for event in body["trace"] if event["event"] == "ACTION_SELECTED"]
        self.assertTrue(selected)
        for event in selected:
            detail = event["detail"]
            for key in ("action", "goal", "reason", "priority", "duration_ms"):
                self.assertIn(key, detail)
            self.assertTrue(detail["reason"])
            self.assertIsInstance(detail["duration_ms"], (int, float))
            self.assertGreaterEqual(detail["duration_ms"], 0)
            self.assertNotIn("state", detail)
            self.assertNotIn("patient_state", detail)
        self.assert_persisted(body)
        self.backend.sessions.clear()
        restored = self.http.get("/api/debug/session/" + body["session_id"])
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.json()["trace"], body["trace"])

    def test_step_limit_is_persisted_and_chat_can_resume_without_duplicate_report(self):
        factory = self.backend.create_patient_state

        def fully_answered(kind):
            state = factory(kind)
            state.update(bp_systolic=125, bp_diastolic=78, medication_status="taken",
                         symptom_status="denied", red_flag_status="denied",
                         denied_red_flags=list(self.backend.DANGER_PATTERNS),
                         measurement_context="安静休息后测量")
            state["coverage"] = dict.fromkeys(state["coverage"], True)
            state["topic_status"] = dict.fromkeys(state["coverage"], "answered")
            return state

        def one_step(state):
            return self.runner.run_agent(state, call_json_model=self.backend.call_json_model,
                                         max_steps=1)

        with patch.object(self.backend, "create_patient_state", side_effect=fully_answered), patch.object(self.backend, "run_agent", side_effect=one_step):
            body = self.trigger()
        self.assertEqual(body["state"]["followup_status"], "interrupted")
        self.assertTrue(body["state"]["pending_tasks"])
        self.assertFalse(body.get("report"))
        self.assertFalse(self.query("SELECT * FROM reports WHERE session_id=?", (body["session_id"],)))
        self.assert_persisted(body)
        self.backend.sessions.clear()
        restored = self.http.get("/api/debug/session/" + body["session_id"])
        self.assertEqual(restored.json()["followup_status"], "interrupted")
        resumed = self.chat(body["session_id"], "今天已经按时吃过药了")
        self.assertEqual(resumed["state"]["followup_status"], "completed")
        self.assertTrue(resumed["report"])
        self.assertEqual(len(self.query("SELECT * FROM reports WHERE session_id=?", (body["session_id"],))), 1)
        self.assert_persisted(resumed)

    def test_refusal_preserves_missing_facts_without_generating_completion_report(self):
        initial = self.trigger()
        body = self.chat(initial["session_id"], "我不想回答今天吃药的情况")
        state = body["state"]
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertIn("medication", state["deferred_topics"])
        self.assertNotEqual(state["followup_status"], "completed")
        self.assertNotEqual(body["message"], initial["message"])
        self.assertFalse(self.query("SELECT * FROM reports WHERE session_id=?", (initial["session_id"],)))
        self.assert_persisted(body)
        recovered = self.chat(initial["session_id"], "今天已经按时吃了药")
        self.assertEqual(recovered["state"]["medication_status"], "taken")
        self.assertNotIn("medication", recovered["state"]["deferred_topics"])

    def test_three_unclear_answers_stop_repeating_the_same_pending_topic(self):
        initial = self.trigger()
        body = initial
        for _ in range(3):
            body = self.chat(initial["session_id"], "不确定")
        state = body["state"]
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["medication_status"], "unknown")
        self.assertEqual(state["deferred_topics"].get("medication"), "unclear_limit")
        self.assertNotEqual(state["followup_status"], "completed")
        if state["followup_status"] == "waiting_for_user":
            self.assertNotEqual(state["question_context"]["topic"], "medication")
        self.assertFalse(self.query("SELECT * FROM reports WHERE session_id=?", (initial["session_id"],)))
        self.assert_persisted(body)

    def test_sqlite_read_failure_returns_503_without_exposing_internal_error(self):
        initial = self.trigger()
        with patch.object(self.backend, "read_session", side_effect=sqlite3.OperationalError("private DB path /secret/file")):
            body = self.chat(initial["session_id"], "已经吃了", expected=503)
        self.assertIn("detail", body)
        self.assertNotIn("/secret/file", body["detail"])
        self.assertNotIn("report", body)

    def test_sqlite_read_failure_still_handles_new_danger_and_closes_normal_chat(self):
        initial = self.trigger()
        sid = initial["session_id"]
        with patch.object(self.backend, "read_session", side_effect=sqlite3.OperationalError("private storage failure")):
            failed = self.chat(sid, "我现在剧烈胸痛，呼吸困难", expected=503)
        self.assertIn("胸痛", failed["detail"])
        self.assertIn("呼吸困难", failed["detail"])
        self.assertIn("立即", failed["detail"])
        self.assertIn("保存", failed["detail"])
        self.assertNotIn("private storage failure", failed["detail"])
        self.assertEqual(self.backend.sessions[sid]["followup_status"], "escalated")
        self.chat(sid, "今天吃过药了", expected=409)
        self.assertFalse(self.query("SELECT * FROM reports WHERE session_id=?", (sid,)))

    def test_medication_guidance_resumes_the_original_question_after_regimen_details(self):
        initial = self.trigger()
        sid = initial["session_id"]
        patient_question = "我可以自行多吃一片降压药吗？"
        regimen = "氨氯地平5毫克，每天早上八点服用"
        medication_prompts = []

        def model(*args, **kwargs):
            system = kwargs.get("system_prompt") or (args[0] if args else "")
            user = kwargs.get("user_prompt") or (args[1] if len(args) > 1 else "")
            if "用药管理辅助模块" not in system:
                return {}
            medication_prompts.append(user)
            if len(medication_prompts) == 1:
                return {"patient_message": "需要先核实已有处方，不能自行增加降压药。",
                        "needs_followup": True,
                        "followup_question": "您现在服用的降压药叫什么，医生原来安排您如何服用？",
                        "needs_professional_review": True, "reason": "药名及原方案尚未提供"}
            return {"patient_message": "已收到您提供的药名和原方案。是否需要调整处方请向医生或药师确认。",
                    "needs_followup": False, "followup_question": "",
                    "needs_professional_review": True, "reason": "已结合患者补充的原方案回答"}

        evidence = [{"block_id": "mock-regimen", "source_id": "mock",
                     "source_title": "Mock证据", "year": 2026, "source_pages": "1",
                     "text": "依照原处方服药，具体方案由医生或药师核实，不自行改变剂量。"}]
        with patch.object(self.backend, "call_json_model", side_effect=model), patch.object(self.guidance, "retrieve_medical_guidance", return_value=evidence):
            first = self.chat(sid, patient_question)
            self.assertEqual(len(medication_prompts), 1)
            self.assertEqual(first["state"]["question_context"]["kind"], "medication_details")
            self.assertEqual(first["state"]["pending_question_followup"]["question"], patient_question)
            self.assertEqual(first["state"]["current_goal"], "collect_medication_details")
            self.assert_persisted(first)
            # Replaying the same question cannot consume the clarification or
            # trigger the model again without additional patient information.
            repeated = self.chat(sid, patient_question)
            self.assertEqual(len(medication_prompts), 1)
            self.assertEqual(repeated["state"]["pending_question_followup"]["question"], patient_question)
            second = self.chat(sid, regimen)
        self.assertEqual(len(medication_prompts), 2)
        self.assertIn(patient_question, medication_prompts[-1])
        self.assertIn(regimen, medication_prompts[-1])
        self.assertIn("已收到", second["message"])
        self.assertFalse(second["state"].get("pending_question_followup"))
        self.assertFalse(second["state"].get("latest_patient_question"))
        self.assertEqual(second["state"]["followup_status"], "waiting_for_user")
        self.assertNotEqual(second["state"]["current_goal"], "collect_medication_details")
        self.assertFalse(self.backend.contains_unsafe_medication_instruction(second["message"]))
        self.assert_persisted(second)

    def test_refused_medication_tool_clarification_stays_unfinished_without_repeated_model_call(self):
        initial = self.trigger()
        question = "我可以自行多吃一片降压药吗？"
        response = {"patient_message": "需先核实原方案，不能自行改变处方。",
                    "needs_followup": True,
                    "followup_question": "您服用的药物名称及原方案是什么？",
                    "needs_professional_review": True, "reason": "缺少原方案"}
        evidence = [{"block_id": "mock-refused", "source_title": "Mock证据",
                     "text": "不自行改变处方，向医生或药师核实原方案。"}]
        medication_calls = []

        def model(*args, **kwargs):
            system = kwargs.get("system_prompt") or (args[0] if args else "")
            if "用药管理辅助模块" in system:
                medication_calls.append(1)
                return response
            return {}

        with patch.object(self.backend, "call_json_model", side_effect=model), patch.object(self.guidance, "retrieve_medical_guidance", return_value=evidence):
            first = self.chat(initial["session_id"], question)
            self.assertEqual(first["state"]["question_context"]["kind"], "medication_details")
            body = self.chat(initial["session_id"], "我不想回答药物名称和服用方案")
        self.assertEqual(len(medication_calls), 1)
        self.assertEqual(body["state"]["followup_status"], "interrupted")
        self.assertEqual(body["state"]["medication_status"], "unknown")
        self.assertFalse(body["state"]["coverage"]["medication"])
        self.assertFalse(body.get("report"))
        self.assertTrue(body["state"]["pending_question_followup"])
        self.assertFalse(self.query("SELECT * FROM reports WHERE session_id=?", (initial["session_id"],)))
        self.assert_persisted(body)

    def test_sqlite_write_failure_keeps_saved_state_and_returns_503(self):
        initial = self.trigger()
        sid = initial["session_id"]
        before = self.query("SELECT state_json FROM followup_sessions WHERE session_id=?", (sid,))[0]["state_json"]
        with patch.object(self.backend, "persist_session", side_effect=sqlite3.OperationalError("database is locked")):
            failed = self.chat(sid, "今天已经吃了药", expected=503)
        self.assertNotIn("report", failed)
        after = self.query("SELECT state_json FROM followup_sessions WHERE session_id=?", (sid,))[0]["state_json"]
        self.assertEqual(after, before)
        restored = self.backend.read_session(sid)
        self.assertEqual(restored["medication_status"], "unknown")
        self.assertFalse(restored["coverage"]["medication"])

    def test_sqlite_initial_write_failure_does_not_publish_a_session(self):
        with patch.object(self.backend, "persist_session", side_effect=sqlite3.OperationalError("disk I/O error")):
            response = self.http.post("/api/trigger", json={"trigger_type": "scheduled"})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertNotIn("session_id", response.json())
        self.assertFalse(self.query("SELECT * FROM followup_sessions"))
        self.assertFalse(self.backend.sessions)

    def test_sqlite_write_failure_does_not_hide_new_danger_or_reopen_the_session(self):
        initial = self.trigger()
        sid = initial["session_id"]
        with patch.object(self.backend, "persist_session", side_effect=sqlite3.OperationalError("disk unavailable")):
            failed = self.chat(sid, "没有胸痛，但是现在喘不上气", expected=503)
        self.assertIn("呼吸困难", failed["detail"])
        self.assertIn("立即", failed["detail"])
        self.assertIn("保存", failed["detail"])
        self.assertNotIn("disk unavailable", failed["detail"])
        self.assertFalse(self.query("SELECT * FROM reports WHERE session_id=?", (sid,)))
        self.assertEqual(self.backend.sessions[sid]["followup_status"], "escalated")
        self.chat(sid, "已经吃了药", expected=409)

    def test_memory_read_failure_is_traced_without_inventing_a_problem_free_history(self):
        with patch.object(self.backend, "refresh_patient_memory", side_effect=sqlite3.OperationalError("offline historical data")):
            body = self.trigger()
        state = body["state"]
        self.assertEqual(state["patient_memory"]["status"], "unavailable")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertNotEqual(state["followup_status"], "completed")
        self.assertIn("读取失败", state["memory_context"])
        self.assertTrue(any(event["event"] == "TOOL_ERROR" and event["detail"]["tool"] == "PATIENT_MEMORY" for event in state["trace"]))
        self.assert_persisted(body)


if __name__ == "__main__":
    unittest.main()

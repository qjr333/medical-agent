"""Offline API regressions; every write goes to a temporary SQLite database."""

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


KNOWLEDGE_FIELDS = {
    "knowledge_directory", "source_registry_path", "registered_sources",
    "knowledge_blocks", "topics", "files", "embedding_model",
    "embedding_available", "fallback_mode", "embedding_error",
}
class APIRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1",
        }), contextlib.redirect_stdout(io.StringIO()):
            cls.backend = importlib.import_module("app")
        cls.database = importlib.import_module("database.connection")
        cls.rag = importlib.import_module("rag")

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.object(self.database, "DATA_DIR", Path(temp)))
        self.stack.enter_context(patch.object(self.database, "DB_PATH", Path(temp) / "test.db"))
        self.stack.enter_context(patch.object(self.backend, "call_json_model", return_value={}))
        self.stack.enter_context(patch.object(self.rag, "DISABLE_EMBEDDING", True))
        self.backend.sessions.clear()
        self.addCleanup(self.backend.sessions.clear)
        self.http = self.stack.enter_context(TestClient(self.backend.app))
        self.patient_id = self.backend.DEMO_PATIENT["patient_id"]

    def trigger(self, kind="scheduled"):
        response = self.http.post("/api/trigger", json={"trigger_type": kind})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue({"message", "state", "trace", "session_id"} <= body.keys())
        self.assertTrue(body["message"])
        self.assertEqual(body["state"]["trigger_type"], kind)
        return body

    def chat(self, session_id, message):
        return self.http.post("/api/chat", json={
            "session_id": session_id, "message": message,
        })

    def query(self, sql, parameters=()):
        conn = self.database.get_connection()
        try:
            return [dict(row) for row in conn.execute(sql, parameters).fetchall()]
        finally:
            conn.close()

    def test_frontend_demo_and_health_contract(self):
        response = self.http.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        demo = self.http.get("/api/demo").json()
        self.assertEqual(set(demo), {"patient", "bp_history", "triggers"})
        self.assertEqual(demo["patient"], self.backend.DEMO_PATIENT)
        self.assertEqual({item["type"] for item in demo["triggers"]}, set(self.backend.TRIGGER_CONFIGS))
        health = self.http.get("/api/health").json()
        self.assertEqual(set(health), {"status", "version", "model_configured", "knowledge"})
        self.assertEqual(health["status"], "ok")
        self.assertEqual(set(health["knowledge"]), KNOWLEDGE_FIELDS)
        self.assertGreater(health["knowledge"]["knowledge_blocks"], 0)

    def test_all_existing_proactive_triggers_and_database_seeding(self):
        for kind in self.backend.TRIGGER_CONFIGS:
            with self.subTest(kind=kind):
                body = self.trigger(kind)
                state = body["state"]
                self.assertEqual(state["patient_id"], self.patient_id)
                self.assertEqual(state["followup_status"], "waiting_for_user")
                self.assertTrue(state["patient_memory"])
                events = {item["event"] for item in body["trace"]}
                self.assertTrue({"TRIGGER_ENGINE", "MEMORY_LOADED", "ASK_USER"} <= events)
                persisted = self.query("SELECT state_json FROM followup_sessions WHERE session_id = ?", (body["session_id"],))
                self.assertEqual(json.loads(persisted[0]["state_json"]), state)
        self.backend.initialize_local_database()
        self.assertEqual(self.query("SELECT COUNT(*) AS n FROM bp_records WHERE source='demo_seed'")[0]["n"], len(self.backend.DEMO_BP_HISTORY))
        self.assertEqual(self.query("SELECT COUNT(*) AS n FROM bp_records WHERE source='demo_trigger'")[0]["n"], 1)

    def test_chat_persists_observations_and_restores_session_from_sqlite(self):
        original = self.trigger("missed_measurement")
        sid = original["session_id"]
        response = self.chat(sid, "我休息后测得血压145/85")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"message", "state", "trace", "session_id"})
        self.assertEqual(body["state"]["bp_systolic"], 145)
        records = self.query("SELECT * FROM bp_records WHERE source='patient_chat'")
        self.assertEqual([(row["systolic"], row["diastolic"]) for row in records], [(145, 85)])
        events = self.query("SELECT * FROM agent_events WHERE session_id=?", (sid,))
        self.assertEqual(len(events), len(body["trace"]))
        self.backend.persist_session(sid, body["state"])
        self.assertEqual(len(self.query("SELECT * FROM agent_events WHERE session_id=?", (sid,))), len(events))
        self.backend.sessions.clear()
        restored = self.http.get("/api/debug/session/" + sid)
        self.assertEqual(restored.json(), body["state"])
        self.backend.sessions.clear()
        response = self.chat(sid, "我今天按时吃了降压药")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"]["medication_status"], "taken")
        self.assertEqual(self.query("SELECT status FROM medication_records")[0]["status"], "taken")

    def test_invalid_requests_and_measurement_are_rejected_without_observation_write(self):
        self.assertEqual(self.http.post("/api/trigger", json={"trigger_type": "unknown"}).status_code, 400)
        self.assertEqual(self.chat("missing", "血压145/85").status_code, 404)
        sid = self.trigger("missed_measurement")["session_id"]
        self.assertEqual(self.chat(sid, "  ").status_code, 400)
        response = self.chat(sid, "血压40/90")
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()["state"]
        self.assertIsNone(state["bp_systolic"])
        self.assertIn("INVALID_BP_INPUT", {item["event"] for item in state["trace"]})
        self.assertFalse(self.query("SELECT * FROM bp_records WHERE source='patient_chat'"))
        self.assertEqual(self.http.get("/api/patients/missing/history").status_code, 404)
        self.assertEqual(self.http.get("/api/patients/missing/memory").status_code, 404)

    def test_danger_escalation_persists_alerts_report_and_closes_chat(self):
        sid = self.trigger("missed_measurement")["session_id"]
        response = self.chat(sid, "我现在剧烈胸痛，呼吸困难")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"message", "state", "alerts", "report", "trace", "session_id"})
        self.assertEqual(body["state"]["risk_level"], "high")
        self.assertEqual(body["state"]["followup_status"], "escalated")
        self.assertIn("胸痛", body["state"]["red_flags"])
        self.assertTrue(self.query("SELECT * FROM alerts WHERE session_id=?", (sid,)))
        reports = self.query("SELECT report_json FROM reports WHERE session_id=?", (sid,))
        self.assertEqual(json.loads(reports[0]["report_json"])["risk"]["level"], "high")
        self.assertEqual(self.chat(sid, "我好了").status_code, 409)

    def test_high_blood_pressure_still_escalates(self):
        sid = self.trigger("missed_measurement")["session_id"]
        response = self.chat(sid, "现在血压180/120")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"]["risk_level"], "high")
        self.assertEqual(response.json()["state"]["followup_status"], "escalated")

    def test_followup_finishes_and_persists_report(self):
        sid = self.trigger("missed_measurement")["session_id"]
        for message in (
            "我安静坐着休息5分钟后测得血压145/85",
            "我今天按时吃了降压药",
            "都没有",
        ):
            response = self.chat(sid, message)
            self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"message", "state", "report", "trace", "session_id"})
        self.assertEqual(body["state"]["followup_status"], "completed")
        self.assertIn("FINISH", {item["event"] for item in body["trace"]})
        reports = self.query("SELECT report_json FROM reports WHERE session_id=?", (sid,))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0]["report_json"]), body["report"])
        self.assertEqual(self.chat(sid, "谢谢").status_code, 409)

    def test_medication_question_uses_guidance_and_safety_interception(self):
        sid = self.trigger()["session_id"]
        # 开场可能只问血压或部分症状；完整否认不依赖某一版问题列表。
        screening = self.chat(
            sid,
            "没有胸痛，没有呼吸困难，没有晕厥，没有意识异常，"
            "没有突发肢体无力，没有突发言语异常，也没有其他不舒服",
        )
        self.assertEqual(screening.status_code, 200, screening.text)
        self.assertTrue(screening.json()["state"]["coverage"]["red_flags"])
        response = self.chat(sid, "我今天忘记吃降压药，现在能不能自己多吃一片？")
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()["state"]
        self.assertEqual(state["medication_status"], "missed")
        self.assertTrue(state["medication_guidance"])
        self.assertTrue(state["knowledge_sources"])
        self.assertIn("MEDICATION_GUIDANCE", {item["event"] for item in state["trace"]})
        self.assertFalse(self.backend.contains_unsafe_medication_instruction(response.json()["message"]))
        self.assertEqual(self.query("SELECT status FROM medication_records")[0]["status"], "missed")
        unsafe = {"patient_message": "今天多吃2片降压药", "needs_followup": False}
        with patch.object(self.backend, "call_json_model", return_value=unsafe):
            result = self.backend.tool_medication_guidance(state)
        self.assertNotIn("今天多吃2片", result["message"])
        self.assertTrue(result["needs_professional_review"])
        self.assertEqual(result["reason"], "原始模型输出触发Medication Safety Guard")
        self.assertFalse(self.backend.contains_unsafe_medication_instruction(result["message"]))

    def test_patient_memory_survives_new_followup_without_current_state_contamination(self):
        sid = self.trigger("missed_measurement")["session_id"]
        response = self.chat(sid, "血压145/85，今天忘记吃降压药，有点头晕")
        self.assertEqual(response.status_code, 200, response.text)
        self.backend.sessions.clear()
        body = self.trigger("missed_measurement")
        state = body["state"]
        self.assertEqual(state["medication_status"], "unknown")
        self.assertEqual(state["symptoms"], [])
        self.assertEqual(state["red_flags"], [])
        memory = state["patient_memory"]
        self.assertGreaterEqual(memory["medication"]["last_7_days"]["issue_report_count"], 1)
        self.assertEqual(memory["last_followup"]["session_id"], sid)
        self.assertIn("recent_medication_issue", memory["focus"])
        history = self.http.get(f"/api/patients/{self.patient_id}/history").json()
        self.assertEqual(set(history), {"patient", "bp_records", "medication_records", "followup_sessions"})
        self.assertEqual(len(history["followup_sessions"]), 2)
        preview = self.http.get(f"/api/patients/{self.patient_id}/memory").json()
        self.assertEqual(preview["version"], "2.0")

    def test_patient_memory_uses_real_time_windows(self):
        old_time = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        conn = self.database.get_connection()
        try:
            with conn:
                conn.execute("INSERT INTO bp_records(patient_id,systolic,diastolic,measured_at) VALUES (?,?,?,?)", (self.patient_id, 190, 125, old_time))
                conn.execute("INSERT INTO medication_records(patient_id,status,recorded_at) VALUES (?,?,?)", (self.patient_id, "missed", old_time))
        finally:
            conn.close()
        memory = self.http.get(f"/api/patients/{self.patient_id}/memory").json()
        self.assertEqual(memory["blood_pressure"]["last_30_days"]["measurement_count"], len(self.backend.DEMO_BP_HISTORY))
        self.assertEqual(memory["medication"]["last_30_days"]["issue_report_count"], 0)

    def test_knowledge_retrieval_and_existing_medical_tool_endpoints(self):
        response = self.http.get("/api/test-rag", params={"query": "忘记吃降压药能不能多吃一片", "age": 72, "topic": "medication_safety"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"query", "patient_age", "topic_hint", "results"})
        self.assertTrue(body["results"])
        # Evidence fields consumed by the existing tools stay available.
        self.assertTrue({"block_id", "text", "topic", "score", "source_id", "source_title"} <= body["results"][0].keys())
        for route in ("/api/test-bp-trend", "/api/test-visit-advice", "/api/test-medication-guidance", "/api/test-symptom-status"):
            with self.subTest(route=route):
                response = self.http.get(route)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json())
        symptoms = self.http.get("/api/test-symptom-status").json()
        self.assertEqual(symptoms["危险症状"]["risk_level"], "high")
        self.assertNotEqual(symptoms["明确否认"]["risk_level"], "high")
        status = self.http.get("/api/db-status").json()
        self.assertEqual(set(status), {"status", "database", "counts"})
        self.assertEqual(len(status["counts"]), 7)
        self.assertEqual(set(self.http.get("/api/knowledge-status").json()), KNOWLEDGE_FIELDS)


if __name__ == "__main__":
    unittest.main()

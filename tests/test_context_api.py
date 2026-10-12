"""Context-aware conversation regressions using Mock models and temporary SQLite.

These exercise the real routes, extractor, planner, risk tools, persistence and
memory. They do not call DeepSeek or write any production patient data/cache.
"""

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class ContextAPIRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1",
        }), contextlib.redirect_stdout(io.StringIO()):
            cls.backend = importlib.import_module("app")
        cls.database = importlib.import_module("database.connection")
        cls.rag = importlib.import_module("rag")
        cls.state_manager = importlib.import_module("agent.state_manager")
        cls.utils = importlib.import_module("backend.utils")

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.object(self.database, "DATA_DIR", Path(directory)))
        self.stack.enter_context(patch.object(self.database, "DB_PATH", Path(directory) / "context.db"))
        self.model = self.stack.enter_context(patch.object(self.backend, "call_json_model", return_value={}))
        self.stack.enter_context(patch.object(self.rag, "DISABLE_EMBEDDING", True))
        # 本组短答回归使用明确正常血压、平稳历史，避免将“没有”送给
        # V3 在异常血压场景优先提出的安全问题。真实 Planner/Safety 保持执行。
        configs = json.loads(json.dumps(self.backend.TRIGGER_CONFIGS))
        configs["scheduled"].update(bp_systolic=135, bp_diastolic=85)
        configs["attention_context"] = {
            "label": "模拟异常血压上下文", "followup_reason": "模拟异常血压随访",
            "bp_systolic": 162, "bp_diastolic": 96,
        }
        self.stack.enter_context(patch.object(self.backend, "TRIGGER_CONFIGS", configs))
        self.stack.enter_context(patch.object(self.backend, "DEMO_BP_HISTORY", [
            {"systolic": 135, "diastolic": 85} for _ in range(3)
        ]))
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
        self.assertEqual(set(body), {"message", "state", "trace", "session_id"})
        self.assertEqual(body["state"]["followup_status"], "waiting_for_user")
        return body

    def chat(self, sid, message):
        response = self.http.post("/api/chat", json={"session_id": sid, "message": message})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_pending_question(self, body, goal, topic):
        state = body["state"]
        self.assertEqual(state["current_goal"], goal)
        self.assertEqual(state["last_action"], "ASK_USER")
        self.assertEqual(state["followup_status"], "waiting_for_user")
        context = state["question_context"]
        self.assertEqual(context["goal"], state["current_goal"])
        self.assertEqual(context["topic"], topic)
        self.assertEqual(context["question"], state["last_agent_question"])
        self.assertEqual(body["message"], state["last_agent_question"])
        self.assertTrue(context["kind"])

    def last_patch(self, body):
        updates = [item for item in body["trace"] if item["event"] == "STATE_UPDATE"]
        self.assertTrue(updates)
        return updates[-1]["detail"]["patch"]

    def assert_completed_without_unnecessary_measurement_question(self, body):
        state = body["state"]
        self.assertEqual(state["followup_status"], "completed")
        self.assertEqual(state["last_action"], "FINISH")
        self.assertTrue(all(state["coverage"][topic] for topic in
                            ("bp", "medication", "current_symptoms", "red_flags")))
        self.assertFalse(state["coverage"]["measurement_context"])
        self.assertTrue(body["report"])

    def assert_persisted(self, body):
        sid = body["session_id"]
        row = self.query("SELECT state_json FROM followup_sessions WHERE session_id=?", (sid,))
        self.assertEqual(json.loads(row[0]["state_json"]), body["state"])
        events = self.query("SELECT * FROM agent_events WHERE session_id=?", (sid,))
        self.assertEqual(len(events), len(body["trace"]))

    def test_three_turns_resolve_not_taken_without_repeating_the_medication_question(self):
        for answer in ("没有呢", "还没吃"):
            with self.subTest(answer=answer):
                initial = self.trigger()
                sid = initial["session_id"]
                self.assert_pending_question(initial, "check_medication", "medication")
                first = self.chat(sid, answer)
                state = first["state"]
                self.assertEqual(self.last_patch(first)["medication_status"], "not_taken_yet")
                self.assertEqual(state["medication_status"], "not_taken_yet")
                self.assertEqual(state["medication_timing"], "unknown")
                self.assertFalse(state["coverage"]["medication"])
                self.assertEqual(state["topic_status"]["medication"], "clarification_needed")
                self.assertEqual(state["last_action"], "ASK_USER")
                self.assertEqual(state["question_context"]["topic"], "medication")
                self.assertNotEqual(first["message"], initial["message"])
                self.assertIn("服药时间", first["message"])
                self.assertEqual(state["symptom_status"], "unknown")
                self.assertEqual(state["red_flag_status"], "unknown")
                self.assert_persisted(first)

                second = self.chat(sid, "还没到规定的服药时间")
                state = second["state"]
                self.assertEqual(state["medication_status"], "not_taken_yet")
                self.assertEqual(state["medication_timing"], "not_due")
                self.assertTrue(state["coverage"]["medication"])
                self.assertEqual(state["topic_status"]["medication"], "answered")
                self.assert_pending_question(second, "check_symptoms", "current_symptoms")

                third = self.chat(sid, "都没有不舒服")
                state = third["state"]
                self.assertEqual(state["medication_status"], "not_taken_yet")
                self.assertEqual(state["symptom_status"], "denied")
                self.assertEqual(state["red_flag_status"], "denied")
                self.assertTrue(state["coverage"]["current_symptoms"])
                self.assertTrue(state["coverage"]["red_flags"])
                self.assertEqual(state["topic_status"]["current_symptoms"], "answered")
                self.assertEqual(state["topic_status"]["red_flags"], "answered")
                self.assert_completed_without_unnecessary_measurement_question(third)
                self.assertEqual(len({initial["message"], first["message"], second["message"], third["message"]}), 4)
                self.assert_persisted(third)
                self.assertFalse(self.query("SELECT * FROM medication_records WHERE status='missed'"))
                records = self.query("SELECT status FROM medication_records")
                self.assertTrue(records)
                self.assertTrue(all(row["status"] == "not_taken_yet" for row in records))
                self.assertEqual(self.http.get(f"/api/patients/{state['patient_id']}/memory").json()["medication"]["last_7_days"]["issue_report_count"], 0)

    def test_already_taken_and_yes_advance_without_model_invented_information(self):
        for answer in ("已经吃了", "是的"):
            with self.subTest(answer=answer):
                initial = self.trigger()
                self.model.reset_mock()
                body = self.chat(initial["session_id"], answer)
                self.assertEqual(self.last_patch(body)["medication_status"], "taken")
                self.assertEqual(body["state"]["medication_status"], "taken")
                self.assertTrue(body["state"]["coverage"]["medication"])
                self.assertEqual(body["state"]["topic_status"]["medication"], "answered")
                self.assertEqual(body["state"]["symptom_status"], "unknown")
                self.assertEqual(body["state"]["red_flag_status"], "unknown")
                self.assert_pending_question(body, "check_symptoms", "current_symptoms")
                self.model.assert_not_called()

    def test_not_due_reply_alone_is_not_a_missed_dose(self):
        initial = self.trigger()
        body = self.chat(initial["session_id"], "还没到吃药时间")
        state = body["state"]
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertEqual(state["medication_timing"], "not_due")
        self.assertTrue(state["coverage"]["medication"])
        self.assert_pending_question(body, "check_symptoms", "current_symptoms")
        self.assertEqual([row["status"] for row in self.query("SELECT status FROM medication_records")], ["not_taken_yet"])

    def test_explicit_forgotten_dose_is_missed_and_memory_retains_the_report(self):
        initial = self.trigger()
        body = self.chat(initial["session_id"], "今天忘记吃药了")
        state = body["state"]
        self.assertEqual(self.last_patch(body)["medication_status"], "missed")
        self.assertEqual(state["medication_status"], "missed")
        self.assertTrue(state["coverage"]["medication"])
        self.assert_pending_question(body, "check_symptoms", "current_symptoms")
        self.assertEqual([row["status"] for row in self.query("SELECT status FROM medication_records")], ["missed"])
        self.assertEqual(state["patient_memory"]["medication"]["last_7_days"]["issue_report_count"], 1)

    def test_denial_of_missed_doses_does_not_mean_no_medication_was_taken(self):
        initial = self.trigger()
        body = self.chat(initial["session_id"], "没有漏服")
        state = body["state"]
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["topic_status"]["medication"], "clarification_needed")
        self.assertEqual(state["question_context"]["topic"], "medication")
        self.assertNotEqual(body["message"], initial["message"])
        self.assertFalse(self.query("SELECT * FROM medication_records"))
        answer = self.chat(initial["session_id"], "今天已经吃了")
        self.assertEqual(answer["state"]["medication_status"], "taken")
        self.assert_pending_question(answer, "check_symptoms", "current_symptoms")

    def test_uncertainty_is_clarification_and_does_not_fill_missing_topics(self):
        initial = self.trigger()
        body = self.chat(initial["session_id"], "不确定")
        state = body["state"]
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["topic_status"]["medication"], "clarification_needed")
        self.assertEqual(state["symptom_status"], "unknown")
        self.assertEqual(state["red_flag_status"], "unknown")
        self.assertFalse(state["coverage"]["current_symptoms"])
        self.assertFalse(state["coverage"]["red_flags"])
        self.assertEqual(state["last_action"], "ASK_USER")
        self.assertEqual(state["question_context"]["goal"], state["current_goal"])
        self.assertNotEqual(body["message"], initial["message"])
        self.assertFalse(self.query("SELECT * FROM medication_records"))

    def test_clear_timing_reply_after_uncertainty_still_resolves_the_pending_goal(self):
        initial = self.trigger()
        sid = initial["session_id"]
        self.chat(sid, "还没吃")
        unclear = self.chat(sid, "不确定")
        self.assertEqual(unclear["state"]["current_goal"], "clarify_medication_timing")
        self.assertEqual(unclear["state"]["medication_timing"], "unknown")
        self.model.reset_mock()
        clear = self.chat(sid, "已经超过原定服药时间了")
        state = clear["state"]
        self.assertEqual(state["medication_status"], "not_taken_yet")
        self.assertEqual(state["medication_timing"], "overdue")
        self.assertTrue(state["coverage"]["medication"])
        self.assertEqual(state["topic_status"]["medication"], "answered")
        self.assert_pending_question(clear, "check_symptoms", "current_symptoms")
        self.assertNotEqual(clear["message"], unclear["message"])
        self.assertFalse(self.query("SELECT * FROM medication_records WHERE status='missed'"))
        self.model.assert_not_called()
        self.assert_persisted(clear)

    def test_compound_reply_retains_confirmed_medication_timing_and_answers_symptoms(self):
        for timing_answer, timing, prefill in (
            ("还没到吃药时间", "not_due", False),
            ("已经超过原定服药时间了", "overdue", True),
            ("还没吃，已经超过原定服药时间了", "overdue", False),
        ):
            with self.subTest(timing=timing, answer=timing_answer):
                initial = self.trigger()
                sid = initial["session_id"]
                if prefill:
                    self.chat(sid, "还没吃")
                confirmed = self.chat(sid, timing_answer)
                self.assertEqual(confirmed["state"]["medication_timing"], timing)
                self.assert_pending_question(confirmed, "check_symptoms", "current_symptoms")
                body = self.chat(sid, "都没有不舒服，药还没吃")
                state = body["state"]
                self.assertEqual(state["medication_status"], "not_taken_yet")
                self.assertEqual(state["medication_timing"], timing)
                self.assertTrue(state["coverage"]["medication"])
                self.assertEqual(state["topic_status"]["medication"], "answered")
                self.assertEqual(state["symptom_status"], "denied")
                self.assertEqual(state["red_flag_status"], "denied")
                self.assertTrue(state["coverage"]["current_symptoms"])
                self.assertTrue(state["coverage"]["red_flags"])
                self.assert_completed_without_unnecessary_measurement_question(body)
                self.assert_persisted(body)

    def test_compound_denial_and_confirmed_taken_move_to_the_next_topic(self):
        initial = self.trigger()
        sid = initial["session_id"]
        question = self.chat(sid, "已经吃了")
        self.assert_pending_question(question, "check_symptoms", "current_symptoms")
        body = self.chat(sid, "都没有不舒服，今天已经吃了药")
        state = body["state"]
        self.assertEqual(state["medication_status"], "taken")
        self.assertTrue(state["coverage"]["medication"])
        self.assertEqual(state["symptom_status"], "denied")
        self.assertEqual(state["red_flag_status"], "denied")
        self.assertTrue(state["coverage"]["current_symptoms"])
        self.assertTrue(state["coverage"]["red_flags"])
        self.assert_completed_without_unnecessary_measurement_question(body)
        self.assertNotEqual(body["message"], question["message"])
        self.assert_persisted(body)

    def test_same_no_reply_uses_the_actual_goal_and_preserves_confirmed_medication(self):
        med_initial = self.trigger()
        med_reply = self.chat(med_initial["session_id"], "没有")
        self.assertEqual(med_reply["state"]["medication_status"], "not_taken_yet")
        self.assertEqual(med_reply["state"]["symptom_status"], "unknown")

        symptoms_initial = self.trigger()
        symptom_question = self.chat(symptoms_initial["session_id"], "已经吃了")
        self.assert_pending_question(symptom_question, "check_symptoms", "current_symptoms")
        symptom_reply = self.chat(symptoms_initial["session_id"], "没有")
        state = symptom_reply["state"]
        self.assertEqual(state["medication_status"], "taken")
        self.assertEqual(state["symptom_status"], "denied")
        self.assertEqual(state["red_flag_status"], "denied")
        self.assertTrue(state["coverage"]["current_symptoms"])
        self.assertTrue(state["coverage"]["red_flags"])
        self.assert_completed_without_unnecessary_measurement_question(symptom_reply)

    def test_mixed_denial_and_danger_escalates_before_normal_questions(self):
        for message in (
            "没有胸痛，但是呼吸困难", "没有胸痛，但是现在喘不上气",
            "没有胸痛只有呼吸困难", "没有胸痛只是喘不上气",
            "没有头晕现在呼吸困难",
        ):
            with self.subTest(message=message):
                initial = self.trigger()
                sid = initial["session_id"]
                body = self.chat(sid, message)
                state = body["state"]
                self.assertEqual(set(body), {"message", "state", "alerts", "report", "trace", "session_id"})
                denied = "头晕" if "没有头晕" in message else "胸痛"
                self.assertIn(denied, state["denied_symptoms"])
                self.assertNotIn(denied, state["symptoms"])
                self.assertIn("呼吸困难", state["red_flags"])
                self.assertEqual(state["symptom_status"], "reported")
                self.assertEqual(state["red_flag_status"], "reported")
                self.assertEqual(state["risk_level"], "high")
                self.assertEqual(state["followup_status"], "escalated")
                self.assertEqual(self.backend.select_next_action(state, []), "ESCALATE")
                last_update = max(i for i, item in enumerate(body["trace"]) if item["event"] == "STATE_UPDATE")
                events = [item["event"] for item in body["trace"][last_update + 1:]]
                self.assertIn("ASSESS_RISK", events)
                self.assertIn("ESCALATE", events)
                self.assertNotIn("ASK_USER", events)
                self.assertEqual(len(self.query("SELECT * FROM reports WHERE session_id=?", (sid,))), 1)
                self.assertTrue(self.query("SELECT * FROM alerts WHERE session_id=?", (sid,)))
                self.assertEqual(self.http.post("/api/chat", json={"session_id": sid, "message": "已经吃了"}).status_code, 409)
                self.assert_persisted(body)

    def test_medication_observations_keep_yesterday_separate_from_today(self):
        initial = self.trigger()
        sid = initial["session_id"]
        today = initial["state"]["medication_date"]
        yesterday = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
        past = self.chat(sid, "昨天忘记吃药了")
        state = past["state"]
        self.assertEqual(state["medication_status"], "unknown")
        self.assertEqual(state["medication_date"], today)
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["medication_observations"][yesterday]["status"], "missed")
        current = self.chat(sid, "今天已经吃了药")
        state = current["state"]
        self.assertEqual(state["medication_status"], "taken")
        self.assertEqual(state["medication_date"], today)
        self.assertEqual(state["medication_observations"][yesterday]["status"], "missed")
        self.assertEqual(state["medication_observations"][today]["status"], "taken")
        records = self.query("SELECT status, detail FROM medication_records ORDER BY id")
        self.assertEqual([row["status"] for row in records], ["missed", "taken"])
        self.assertIn(f"服药日期：{yesterday}", records[0]["detail"])
        self.assertIn(f"服药日期：{today}", records[1]["detail"])
        self.assert_persisted(current)

    def test_sqlite_restore_retains_the_clarification_context_for_short_answers(self):
        initial = self.trigger()
        sid = initial["session_id"]
        first = self.chat(sid, "还没吃")
        self.assertEqual(first["state"]["topic_status"]["medication"], "clarification_needed")
        self.backend.sessions.clear()
        response = self.http.get("/api/debug/session/" + sid)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), first["state"])
        self.backend.sessions.clear()
        second = self.chat(sid, "还没到时间")
        self.assertEqual(second["state"]["medication_status"], "not_taken_yet")
        self.assertEqual(second["state"]["medication_timing"], "not_due")
        self.assertTrue(second["state"]["coverage"]["medication"])
        self.assert_pending_question(second, "check_symptoms", "current_symptoms")
        self.assert_persisted(second)

    def test_cross_day_reload_reopens_current_medication_without_losing_prior_observation(self):
        with patch.object(self.state_manager, "today_local_date", return_value="2026-10-10"), patch.object(self.utils, "today_local_date", return_value="2026-10-10"):
            initial = self.trigger()
            sid = initial["session_id"]
            previous = self.chat(sid, "已经吃了")
            self.assertEqual(previous["state"]["medication_date"], "2026-10-10")
            self.assertEqual(previous["state"]["medication_status"], "taken")
        self.backend.sessions.clear()
        with patch.object(self.state_manager, "today_local_date", return_value="2026-10-11"), patch.object(self.utils, "today_local_date", return_value="2026-10-11"):
            current = self.chat(sid, "都没有")
        state = current["state"]
        self.assertEqual(state["medication_date"], "2026-10-11")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["medication_observations"]["2026-10-10"]["status"], "taken")
        self.assert_pending_question(current, "check_medication", "medication")
        self.assertEqual([row["status"] for row in self.query("SELECT status FROM medication_records")], ["taken"])
        self.assert_persisted(current)

    def legacy_state(self, state):
        """Remove phase-three additions to simulate real phase-two state_json."""
        legacy = json.loads(json.dumps(state))
        for key in (
            "medication_date", "medication_timing", "medication_observations",
            "topic_status", "question_context", "last_agent_question",
            "question_attempts", "denied_red_flags", "medication_adherence",
        ):
            legacy.pop(key, None)
        for event in legacy["trace"]:
            if event["event"] == "STATE_UPDATE":
                extracted = event["detail"]["patch"]
                for key in ("medication_timing", "medication_observations", "topic_status", "denied_red_flags", "medication_adherence"):
                    extracted.pop(key, None)
        return legacy

    def test_legacy_session_uses_dated_trace_instead_of_treating_yesterdays_taken_as_today(self):
        with patch.object(self.state_manager, "today_local_date", return_value="2026-10-10"), patch.object(self.utils, "today_local_date", return_value="2026-10-10"):
            initial = self.trigger()
            sid = initial["session_id"]
            previous = self.chat(sid, "已经吃了")
        legacy = self.legacy_state(previous["state"])
        for event in legacy["trace"]:
            event["time"] = "2026-10-10T10:00:00+00:00"
        self.backend.persist_session(sid, legacy)
        self.backend.sessions.clear()
        with patch.object(self.state_manager, "today_local_date", return_value="2026-10-11"), patch.object(self.utils, "today_local_date", return_value="2026-10-11"):
            current = self.chat(sid, "都没有")
        state = current["state"]
        self.assertEqual(state["medication_date"], "2026-10-11")
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["medication_observations"]["2026-10-10"]["status"], "taken")
        self.assert_pending_question(current, "check_medication", "medication")
        self.assert_persisted(current)

    def test_legacy_adherence_question_is_restored_before_interpreting_no(self):
        initial = self.trigger()
        sid = initial["session_id"]
        legacy = self.legacy_state(initial["state"])
        original = "您今天有没有漏服或者自行停药？"
        for event in legacy["trace"]:
            if event["event"] == "ASK_USER":
                event["detail"]["question"] = original
        self.backend.persist_session(sid, legacy)
        self.backend.sessions.clear()
        current = self.chat(sid, "没有")
        state = current["state"]
        self.assertEqual(state["medication_status"], "unknown")
        self.assertFalse(state["coverage"]["medication"])
        self.assertEqual(state["medication_adherence"], "no_missed_or_stopped")
        self.assertEqual(state["topic_status"]["medication"], "clarification_needed")
        self.assertNotEqual(current["message"], original)
        self.assertFalse(self.query("SELECT * FROM medication_records"))

    def test_legacy_partial_symptom_denial_does_not_exclude_unasked_danger_signs(self):
        # 异常血压由真实 V3 先问安全，再核实服药，最后核实测量条件。
        initial = self.trigger("attention_context")
        sid = initial["session_id"]
        self.assert_pending_question(initial, "check_symptoms", "current_symptoms")
        medication_question = self.chat(sid, "都没有")
        self.assert_pending_question(medication_question, "check_medication", "medication")
        previous = self.chat(sid, "已经吃了")
        self.assert_pending_question(previous, "check_measurement_context", "measurement_context")
        legacy = self.legacy_state(previous["state"])
        legacy["denied_symptoms"] = ["头晕", "胸痛", "呼吸困难"]
        for event in legacy["trace"]:
            if event["event"] == "ASK_USER" and event["detail"]["goal"] == "check_symptoms":
                event["detail"]["question"] = "您现在有没有头晕、胸痛或者呼吸困难？"
            if event["event"] == "STATE_UPDATE" and event["detail"]["patch"].get("denied_symptoms"):
                event["detail"]["patch"]["denied_symptoms"] = ["头晕", "胸痛", "呼吸困难"]
        self.backend.persist_session(sid, legacy)
        self.backend.sessions.clear()
        current = self.chat(sid, "是的")
        state = current["state"]
        self.assertEqual(set(state["denied_red_flags"]), {"胸痛", "呼吸困难"})
        self.assertEqual(state["red_flag_status"], "unknown")
        self.assertFalse(state["coverage"]["red_flags"])
        self.assertEqual(state["medication_status"], "taken")
        self.assertTrue(state["coverage"]["measurement_context"])
        self.assert_pending_question(current, "check_symptoms", "red_flags")
        self.assertNotIn("胸痛", current["message"])
        self.assertNotIn("呼吸困难", current["message"])
        self.assertNotIn("头晕", current["message"])
        self.assertEqual(set(state["question_context"]["asked_red_flags"]), {"晕厥", "意识异常", "突发肢体无力", "突发言语异常"})
        self.assert_persisted(current)

    def test_memory_counts_pending_medication_separately_from_adherence_issues(self):
        initial = self.trigger()
        body = self.chat(initial["session_id"], "还没吃")
        memory = self.http.get(f"/api/patients/{body['state']['patient_id']}/memory").json()
        for window in ("last_7_days", "last_30_days"):
            self.assertEqual(memory["medication"][window]["status_report_counts"].get("not_taken_yet"), 1)
            self.assertEqual(memory["medication"][window]["issue_report_count"], 0)
        self.assertNotIn("recent_medication_issue", memory["focus"])

    def test_finished_followup_does_not_regenerate_report_on_later_chat_or_restart(self):
        initial = self.trigger("missed_measurement")
        sid = initial["session_id"]
        first = self.chat(sid, "安静坐着休息5分钟后血压145/85")
        self.assert_pending_question(first, "check_medication", "medication")
        second = self.chat(sid, "已经吃了")
        self.assert_pending_question(second, "check_symptoms", "current_symptoms")
        final = self.chat(sid, "都没有不舒服")
        self.assertEqual(set(final), {"message", "state", "report", "trace", "session_id"})
        self.assertEqual(final["state"]["followup_status"], "completed")
        self.assertEqual(final["state"]["last_action"], "FINISH")
        self.assertTrue(all(final["state"]["coverage"].values()))
        reports = self.query("SELECT * FROM reports WHERE session_id=?", (sid,))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0]["report_json"]), final["report"])
        # A caller that reruns the business runner directly must also reuse the
        # completed result instead of generating a second report or FINISH event.
        state = self.backend.read_session(sid)
        trace_count = len(state["trace"])
        action_count = len(state["actions_completed"])
        with patch("agent.runner.generate_report", side_effect=AssertionError("report regenerated")):
            repeated = self.backend.run_agent(state)
        self.assertEqual(repeated["state"]["report"], final["report"])
        self.assertEqual(len(state["trace"]), trace_count)
        self.assertEqual(len(state["actions_completed"]), action_count)
        for restore in (False, True):
            if restore:
                self.backend.sessions.clear()
            response = self.http.post("/api/chat", json={"session_id": sid, "message": "谢谢"})
            self.assertEqual(response.status_code, 409)
        self.assertEqual(self.query("SELECT * FROM reports WHERE session_id=?", (sid,)), reports)
        self.assert_persisted(final)


if __name__ == "__main__":
    unittest.main()

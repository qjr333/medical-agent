"""Storage and Patient Memory regressions using disposable SQLite databases."""

import contextlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import db
from database import connection, repository
from memory import patient_memory


class TemporaryDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db_path = self.directory / "regression.db"
        self.stack.enter_context(patch.object(connection, "DATA_DIR", self.directory))
        self.stack.enter_context(patch.object(connection, "DB_PATH", self.db_path))
        connection.init_db()
        repository.sessions.clear()
        self.addCleanup(repository.sessions.clear)

    def query(self, sql, parameters=()):
        conn = connection.get_connection()
        try:
            return [dict(row) for row in conn.execute(sql, parameters).fetchall()]
        finally:
            conn.close()


class ConnectionTests(TemporaryDatabaseTests):
    def test_existing_seven_tables_and_foreign_keys(self):
        tables = {row["name"] for row in self.query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        self.assertTrue({"patients", "bp_records", "medication_records",
                         "followup_sessions", "agent_events", "alerts", "reports"} <= tables)
        self.assertEqual(self.query("PRAGMA foreign_keys"), [{"foreign_keys": 1}])
        indexes = {row["name"] for row in self.query(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
        self.assertTrue({"idx_bp_patient_time", "idx_med_patient_time", "idx_events_session"} <= indexes)

    def test_initialization_retains_existing_patient_and_observation(self):
        conn = connection.get_connection()
        try:
            with conn:
                conn.execute("INSERT INTO patients(patient_id,patient_name) VALUES('p','模拟患者')")
                conn.execute("""INSERT INTO bp_records(patient_id,systolic,diastolic,measured_at)
                                VALUES('p',140,85,'2026-10-01T08:00:00+00:00')""")
        finally:
            conn.close()
        connection.init_db()
        self.assertEqual(self.query("SELECT patient_name FROM patients"), [{"patient_name": "模拟患者"}])
        self.assertEqual(len(self.query("SELECT * FROM bp_records")), 1)

    def test_db_compatibility_facade_uses_explicit_path(self):
        with patch.object(db, "DB_PATH", self.db_path):
            db.init_db()
            conn = db.get_connection()
            try:
                actual_path = Path(conn.execute("PRAGMA database_list").fetchone()["file"])
                self.assertEqual(actual_path.resolve(), self.db_path.resolve())
                self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            finally:
                conn.close()


class RepositoryTests(TemporaryDatabaseTests):
    patient = {"patient_id": "p", "patient_name": "模拟患者", "age": 72,
               "conditions": ["高血压"]}
    history = [{"systolic": 130, "diastolic": 80},
               {"systolic": 140, "diastolic": 85},
               {"systolic": 150, "diastolic": 90}]

    def setUp(self):
        super().setUp()
        repository.seed_demo_patient(self.patient, self.history)

    def state(self, **updates):
        state = {"patient_id": "p", "trigger_type": "scheduled",
                 "followup_reason": "计划随访", "followup_status": "waiting_for_user",
                 "risk_level": "normal", "trace": []}
        state.update(updates)
        return state

    def test_seed_and_history_keep_original_fields_and_are_idempotent(self):
        repository.seed_demo_patient(self.patient, self.history)
        self.assertEqual(len(self.query("SELECT * FROM bp_records")), 3)
        self.assertTrue(repository.patient_exists("p"))
        self.assertFalse(repository.patient_exists("absent"))
        self.assertIsNone(repository.get_patient_history("absent"))
        history = repository.get_patient_history("p", 30)
        self.assertEqual(set(history), {"patient", "bp_records", "medication_records", "followup_sessions"})
        self.assertEqual(history["patient"], self.patient)
        self.assertEqual([row["systolic"] for row in history["bp_records"]], [130, 140, 150])
        self.assertEqual(len(repository.get_patient_history("p", 0)["bp_records"]), 1)
        status = repository.get_database_status()
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["database"], "sqlite")
        self.assertEqual(status["counts"], {"patients": 1, "bp_records": 3,
                                          "medication_records": 0, "followup_sessions": 0,
                                          "agent_events": 0, "alerts": 0, "reports": 0})

    def test_transaction_persists_observations_incremental_trace_alerts_report(self):
        state = self.state(trace=[{"event": "ASSESS_RISK", "detail": {"risk": "high"}}],
                           risk_level="high", alerts={"family_alert_content": "通知家属",
                                                      "clinician_alert_content": "联系医生"},
                           report={"risk": {"level": "high"}})
        bp = {"systolic": 185, "diastolic": 110, "source": "demo_trigger"}
        med = {"medication_name": "模拟降压药", "status": "missed", "detail": "患者报告"}
        repository.persist_session("s", state, new_bp=bp, new_medication=med)
        state["trace"].append({"event": "FINISH", "detail": None})
        state["alerts"]["family_alert_content"] = "更新通知"
        state["report"] = {"risk": {"level": "high"}, "finished": True}
        repository.persist_session("s", state, new_bp=bp)
        self.assertEqual(len(self.query("SELECT * FROM bp_records WHERE source='demo_trigger'")), 1)
        self.assertEqual([row["status"] for row in repository.get_recent_medication_records("p")], ["missed"])
        self.assertEqual(len(self.query("SELECT * FROM agent_events")), 2)
        self.assertEqual(len(self.query("SELECT * FROM alerts")), 2)
        self.assertEqual(self.query("SELECT content FROM alerts WHERE target_type='family'")[0]["content"], "更新通知")
        reports = self.query("SELECT report_json FROM reports")
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0]["report_json"]), state["report"])
        self.assertEqual(json.loads(self.query("SELECT state_json FROM followup_sessions")[0]["state_json"]), state)

    def test_invalid_observations_are_not_inserted(self):
        repository.persist_session("s", self.state(),
                                   new_bp={"systolic": 40, "diastolic": 90},
                                   new_medication={"status": "unknown"})
        self.assertEqual(len(self.query("SELECT * FROM bp_records")), 3)
        self.assertEqual(len(self.query("SELECT * FROM medication_records")), 0)

    def test_restore_from_sqlite_after_clearing_process_cache(self):
        state = self.state()
        repository.persist_session("s", state)
        self.assertIs(repository.read_session("s"), state)
        repository.sessions.clear()
        restored = repository.read_session("s")
        self.assertEqual(restored["patient_id"], "p")
        self.assertEqual(restored["symptom_status"], "unknown")
        self.assertEqual(restored["red_flag_status"], "unknown")
        self.assertIsNone(repository.read_session("missing"))
        self.assertIs(repository.read_session("s"), restored)

    def test_previous_followup_and_pending_count_can_exclude_current_session(self):
        with patch.object(repository, "now_iso", return_value="2026-10-09T12:00:00+00:00"):
            repository.persist_session("prior", self.state(risk_level="attention", current_goal="medication"))
        with patch.object(repository, "now_iso", return_value="2026-10-10T12:00:00+00:00"):
            repository.persist_session("current", self.state())
        self.assertEqual(repository.read_last_followup("p")["session_id"], "current")
        self.assertEqual(repository.read_last_followup("p", "current")["session_id"], "prior")
        self.assertEqual(repository.count_unresolved_followups("p"), 2)
        self.assertEqual(repository.count_unresolved_followups("p", "current"), 1)

    def test_interrupted_followup_retains_pending_goal_and_unresolved_count(self):
        state = self.state(followup_status="interrupted", current_goal="clarify_medication_timing")
        repository.persist_session("interrupted", state)
        previous = repository.read_last_followup("p")
        summary = patient_memory.extract_followup_summary(previous)
        self.assertEqual(summary["pending_goal"], "clarify_medication_timing")
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(repository.count_unresolved_followups("p"), 1)
        repository.sessions.clear()
        self.assertEqual(repository.read_session("interrupted")["current_goal"], "clarify_medication_timing")


class FrozenDatetime(datetime):
    reference = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.reference if tz is not None else cls.reference.replace(tzinfo=None)


class PatientMemoryTests(TemporaryDatabaseTests):
    def setUp(self):
        super().setUp()
        self.reference = FrozenDatetime.reference
        self.stack.enter_context(patch.object(patient_memory, "datetime", FrozenDatetime))
        repository.seed_demo_patient(RepositoryTests.patient, [])

    def insert_observations(self, bp_rows=(), med_rows=()):
        conn = connection.get_connection()
        try:
            with conn:
                for days_ago, systolic, diastolic in bp_rows:
                    stamp = (self.reference - timedelta(days=days_ago)).isoformat()
                    conn.execute("""INSERT INTO bp_records(patient_id,systolic,diastolic,measured_at,source)
                                    VALUES('p',?,?,?,'test')""", (systolic, diastolic, stamp))
                for days_ago, status in med_rows:
                    stamp = (self.reference - timedelta(days=days_ago)).isoformat()
                    conn.execute("""INSERT INTO medication_records(patient_id,status,recorded_at)
                                    VALUES('p',?,?)""", (status, stamp))
        finally:
            conn.close()

    def test_7_and_30_day_windows_use_timestamps_and_exclude_future_invalid_bp(self):
        self.insert_observations(
            bp_rows=[(31, 120, 75), (8, 130, 80), (3, 140, 85),
                     (2, 150, 90), (1, 160, 95), (-1, 170, 100), (1, 40, 90)],
            med_rows=[(31, "missed"), (8, "irregular"), (3, "missed"),
                      (2, "taken"), (1, "stopped"), (-1, "missed")],
        )
        memory = patient_memory.load_patient_memory("p")
        seven = memory["blood_pressure"]["last_7_days"]
        thirty = memory["blood_pressure"]["last_30_days"]
        self.assertEqual(seven["measurement_count"], 3)
        self.assertEqual(seven["measurement_days"], 3)
        self.assertEqual(seven["systolic_avg"], 150)
        self.assertEqual(thirty["measurement_count"], 4)
        self.assertTrue(seven["consecutive_rise_last_3_readings"])
        self.assertEqual(memory["medication"]["last_7_days"]["report_count"], 3)
        self.assertEqual(memory["medication"]["last_7_days"]["issue_report_count"], 2)
        self.assertEqual(memory["medication"]["last_30_days"]["issue_report_count"], 3)
        self.assertEqual(memory["recent_missed_reports"], 2)
        self.assertEqual(memory["focus"], ["recent_bp_rise", "recent_medication_issue"])
        self.assertEqual(set(memory), {"version", "patient_id", "loaded_at", "recent_bp_records",
                                       "recent_medication_records", "last_followup", "unresolved_followup_count",
                                       "blood_pressure", "medication", "recent_missed_reports", "focus", "notes"})

    def test_window_statistics_include_all_records_beyond_preview_limits(self):
        self.insert_observations(bp_rows=[(1, 145, 85)] * 370,
                                 med_rows=[(1, "missed")] * 105)
        memory = patient_memory.load_patient_memory("p")
        self.assertEqual(len(memory["recent_bp_records"]), 30)
        self.assertEqual(len(memory["recent_medication_records"]), 14)
        self.assertEqual(memory["blood_pressure"]["last_7_days"]["measurement_count"], 370)
        self.assertEqual(memory["blood_pressure"]["last_30_days"]["measurement_days"], 1)
        self.assertEqual(memory["medication"]["last_7_days"]["report_count"], 105)
        self.assertEqual(memory["medication"]["last_7_days"]["distinct_report_dates"], 1)

    def test_previous_followup_focus_and_refresh_do_not_promote_old_symptoms(self):
        previous = {"patient_id": "p", "trigger_type": "scheduled", "trace": [],
                    "followup_status": "waiting_for_user", "risk_level": "high",
                    "risk_reasons": ["历史胸痛"], "symptoms": ["胸痛"],
                    "danger_signs": ["胸痛"], "current_goal": "medication_status",
                    "medication_status": "missed"}
        repository.persist_session("previous", previous)
        current = {"patient_id": "p", "trigger_type": "scheduled", "trace": [],
                   "followup_status": "waiting_for_user", "risk_level": "normal",
                   "symptoms": [], "danger_signs": [], "symptom_status": "unknown",
                   "medication_status": "unknown"}
        repository.persist_session("current", current)
        memory = patient_memory.refresh_patient_memory(current, "current")
        self.assertEqual(memory["last_followup"]["session_id"], "previous")
        self.assertEqual(memory["unresolved_followup_count"], 1)
        self.assertEqual(memory["last_followup"]["pending_goal"], "medication_status")
        self.assertEqual(memory["focus"], ["previous_risk_attention", "previous_unfinished_question"])
        self.assertEqual(current["symptoms"], [])
        self.assertEqual(current["danger_signs"], [])
        self.assertEqual(current["risk_level"], "normal")
        self.assertEqual(current["medication_status"], "unknown")
        self.assertIn("以上均为历史事实", current["memory_context"])

    def test_historical_report_summary_and_corrupt_json_are_compatible(self):
        row = {"session_id": "s", "trigger_type": "scheduled", "status": "completed",
               "risk_level": "attention", "updated_at": self.reference.isoformat(),
               "state_json": json.dumps({"current_goal": "bp", "medication_status": "unknown"}),
               "report_json": json.dumps({"risk": {"reasons": ["历史趋势"]},
                                          "visit_advice": {"level": "routine"},
                                          "medication": {"status": "taken"}})}
        summary = patient_memory.extract_followup_summary(row)
        self.assertEqual(summary["risk_reasons"], ["历史趋势"])
        self.assertEqual(summary["visit_advice_level"], "routine")
        self.assertEqual(summary["last_recorded_medication_status"], "taken")
        self.assertIsNone(summary["pending_goal"])
        row.update(state_json="bad json", report_json="bad json")
        self.assertEqual(patient_memory.extract_followup_summary(row)["risk_reasons"], [])
        self.assertIsNone(patient_memory.extract_followup_summary(None))

    def test_utc_timestamp_parsing_and_empty_memory(self):
        self.assertEqual(patient_memory.parse_storage_time("2026-10-10 12:00:00"), self.reference)
        self.assertEqual(patient_memory.parse_storage_time("2026-10-10T12:00:00Z"), self.reference)
        self.assertIsNone(patient_memory.parse_storage_time(None))
        self.assertIsNone(patient_memory.parse_storage_time("invalid"))
        memory = patient_memory.load_patient_memory("p")
        self.assertEqual(memory["blood_pressure"]["last_7_days"]["measurement_count"], 0)
        self.assertIsNone(memory["last_followup"])
        self.assertEqual(memory["focus"], [])
        self.assertEqual(patient_memory.memory_context_text({}), "暂无可用的跨随访历史。")
        self.assertEqual(patient_memory.memory_context_text({"patient_memory": memory}),
                         "没有足够的历史记录支持跨随访摘要。")


if __name__ == "__main__":
    unittest.main()

"""Proactive tasks exercised only against disposable SQLite databases."""

import contextlib
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from backend.proactive import ProactiveEngine
from database import connection, tasks


class ProactiveTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db_path = directory / "proactive-test.sqlite"
        self.stack.enter_context(connection.using_database(self.db_path))
        connection.init_db()
        self.now = datetime(2026, 10, 13, 8, 0, tzinfo=timezone.utc)
        conn = connection.get_connection()
        with conn:
            conn.execute("INSERT INTO patients(patient_id,patient_name) VALUES('p','模拟患者')")
            conn.execute("INSERT INTO patients(patient_id,patient_name) VALUES('other','其他模拟患者')")
        conn.close()

    def query(self, sql, parameters=()):
        conn = connection.get_connection()
        try:
            return [dict(row) for row in conn.execute(sql, parameters).fetchall()]
        finally:
            conn.close()

    def queue(self, event_id="event", **kwargs):
        return tasks.create_task("p", event_id, "scheduled", due_at=self.now, **kwargs)

    def insert_bp(self, systolic, diastolic, measured_at, patient_id="p"):
        conn = connection.get_connection()
        try:
            with conn:
                conn.execute("""INSERT INTO bp_records(patient_id,systolic,diastolic,measured_at,source)
                    VALUES(?,?,?,?,?)""", (patient_id, systolic, diastolic, measured_at.isoformat(), "test"))
        finally:
            conn.close()

    def test_additive_schema_preserves_legacy_records(self):
        self.insert_bp(140, 85, self.now)
        connection.init_db()
        self.assertEqual(len(self.query("SELECT * FROM patients")), 2)
        self.assertEqual(len(self.query("SELECT * FROM bp_records")), 1)
        conn = connection.get_connection()
        try:
            conn.execute("BEGIN")
            conn.execute("INSERT INTO patients(patient_id,patient_name) VALUES('rollback','临时')")
            tasks.init_task_schema(conn)
            conn.rollback()
        finally:
            conn.close()
        self.assertFalse(self.query("SELECT * FROM patients WHERE patient_id='rollback'"))

    def test_same_event_same_payload_is_idempotent_and_different_payload_rejected(self):
        first = self.queue(payload={"reason": "routine"})
        second = self.queue(payload={"reason": "routine"})
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertEqual(first["session_id"], second["session_id"])
        with self.assertRaisesRegex(ValueError, "different payload"):
            self.queue(payload={"reason": "changed"})
        with self.assertRaises(ValueError):
            tasks.record_bp_event("p", "event", 170, 100, self.now)
        third = tasks.create_task("other", "event", "scheduled", due_at=self.now)
        self.assertNotEqual(third["task_id"], first["task_id"])
        self.assertEqual(len(self.query("SELECT * FROM proactive_inbox")), 0)
        self.assertEqual(first["delivery_status"], "not_configured")

    def test_bp_event_has_real_record_validates_threshold_and_event_identity(self):
        normal = tasks.record_bp_event("p", "normal", 159, 99, self.now)
        self.assertIsNone(normal["task"])
        event = tasks.record_bp_event("p", "device-1", 160, 99, self.now)
        duplicate = tasks.record_bp_event("p", "device-1", 160, 99, self.now)
        self.assertEqual(event["task"]["trigger_type"], "high_bp")
        self.assertEqual(event["task"]["measurement"]["systolic"], 160)
        self.assertEqual(event["task"]["measurement"]["id"], event["measurement"]["id"])
        self.assertEqual(event["task"]["task_id"], duplicate["task"]["task_id"])
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaises(ValueError):
            tasks.record_bp_event("p", "device-1", 170, 100, self.now)
        dia = tasks.record_bp_event("p", "dia", 150, 100, self.now)
        self.assertIsNotNone(dia["task"])
        for systolic, diastolic in ((False, 90), (170.0, 90), ("170", 90), (40, 90), (170, 220)):
            with self.assertRaises(ValueError):
                tasks.record_bp_event("p", "invalid", systolic, diastolic, self.now)
        with self.assertRaises(ValueError):
            tasks.create_task("p", "forged", "high_bp", {"systolic": 180, "diastolic": 110})
        with self.assertRaises(ValueError):
            tasks.create_task("p", "mismatch", "high_bp", {
                "bp_record_id": event["measurement"]["id"], "systolic": 200, "diastolic": 110})
        self.assertEqual(len(self.query("SELECT * FROM bp_records")), 3)

    def test_bp_implicit_timestamp_idempotency_preserves_first_reading(self):
        first = tasks.record_bp_event("p", "implicit", 170, 100)
        second = tasks.record_bp_event("p", "implicit", 170, 100)
        self.assertEqual(first["measurement"], second["measurement"])
        self.assertEqual(first["task"]["session_id"], second["task"]["session_id"])

    def test_scheduled_plan_utc_due_and_per_window_dedup_after_restart(self):
        plan = tasks.create_plan("p", "scheduled", "2026-10-13T16:00:00+08:00", 3600, plan_id="schedule")
        self.assertEqual(tasks.utc_datetime(plan["due_at"]), self.now)
        self.assertEqual(tasks.generate_due_tasks(self.now - timedelta(seconds=1)), [])
        generated = tasks.generate_due_tasks(self.now + timedelta(hours=2))
        self.assertEqual(len(generated), 3)
        connection.init_db()
        self.assertEqual(tasks.generate_due_tasks(self.now + timedelta(hours=2)), [])
        self.assertEqual(len(tasks.list_tasks("p")), 3)
        self.assertEqual(tasks.utc_datetime(tasks.get_plan(plan["plan_id"])["due_at"]), self.now + timedelta(hours=3))
        tasks.update_plan(plan["plan_id"], enabled=False)
        self.assertEqual(tasks.generate_due_tasks(self.now + timedelta(days=10)), [])

    def test_missed_measurement_queries_valid_records_in_exact_patient_window(self):
        tasks.create_plan("p", "missed_measurement", self.now, 3600, plan_id="measure")
        self.insert_bp(150, 90, self.now - timedelta(minutes=20))
        self.insert_bp(130, 80, self.now + timedelta(minutes=20), "other")
        self.insert_bp(40, 90, self.now + timedelta(minutes=30))
        self.assertEqual(tasks.generate_due_tasks(self.now), [])
        generated = tasks.generate_due_tasks(self.now + timedelta(hours=1))
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0]["trigger_type"], "missed_measurement")
        self.assertEqual(generated[0]["payload"]["measurement_count"], 0)
        self.assertEqual(tasks.generate_due_tasks(self.now + timedelta(hours=1)), [])
        self.assertEqual(len(tasks.list_tasks("p")), 1)

    def test_measurement_at_boundary_belongs_to_next_window(self):
        tasks.create_plan("p", "missed_measurement", self.now, 3600)
        self.insert_bp(140, 90, self.now)
        self.assertEqual(len(tasks.generate_due_tasks(self.now)), 1)
        self.assertEqual(tasks.generate_due_tasks(self.now + timedelta(hours=1)), [])

    def test_manual_missed_task_cannot_bypass_real_window_evidence(self):
        with self.assertRaises(ValueError):
            tasks.create_task("p", "invented-miss", "missed_measurement", {"measurement_count": 0})
        payload = {"window_start": (self.now - timedelta(hours=1)).isoformat(),
                   "window_end": self.now.isoformat(), "measurement_count": 99}
        accepted = tasks.create_task("p", "missing-window", "missed_measurement", payload, self.now)
        self.assertEqual(accepted["payload"]["measurement_count"], 0)
        self.insert_bp(140, 90, self.now - timedelta(minutes=20))
        with self.assertRaisesRegex(ValueError, "already contains"):
            tasks.create_task("p", "fake-miss", "missed_measurement", payload, self.now)
        self.assertEqual(len(tasks.list_tasks("p")), 1)

    def test_tick_runs_callback_and_waits_until_actual_session_completion(self):
        initial = self.queue()
        invoked = []

        def callback(task):
            invoked.append(task["session_id"])
            conn = connection.get_connection()
            try:
                with conn:
                    conn.execute("""INSERT INTO followup_sessions(session_id,patient_id,status,state_json)
                        VALUES(?,?,?,?)""", (task["session_id"], task["patient_id"], "waiting_for_user", "{}"))
            finally:
                conn.close()
            return {"session_id": task["session_id"], "status": "waiting_for_user"}

        result = ProactiveEngine(callback).tick(self.now)
        self.assertEqual(result["results"][0]["status"], "waiting_for_user")
        self.assertEqual(invoked, [initial["session_id"]])
        self.assertEqual(len(self.query("SELECT * FROM proactive_inbox")), 1)
        self.assertEqual(ProactiveEngine(callback).tick(self.now + timedelta(days=1))["executed"], 0)
        conn = connection.get_connection()
        try:
            with conn:
                conn.execute("UPDATE followup_sessions SET status='completed',state_json=? WHERE session_id=?",
                             (json.dumps({"report": {"schema_version": "4.2"}}), initial["session_id"]))
        finally:
            conn.close()
        after = ProactiveEngine(callback).tick(self.now + timedelta(days=2))
        self.assertEqual(after["reconciled"], 1)
        completed = tasks.get_task(initial["task_id"])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["station_delivery_status"], "no_recipients")
        self.assertEqual(completed["delivery_status"], "not_configured")

    def test_finite_backoff_retry_unknown_status_never_claims_success(self):
        initial = self.queue(max_attempts=3)
        invoked = []

        def incomplete(task):
            invoked.append(task["session_id"])
            return {"session_id": task["session_id"], "status": "processing"}

        engine = ProactiveEngine(incomplete)
        first = engine.tick(self.now)["results"][0]
        self.assertEqual(first["status"], "retry")
        self.assertEqual(tasks.utc_datetime(first["due_at"]), self.now + timedelta(seconds=5))
        self.assertEqual(engine.tick(self.now + timedelta(seconds=4))["executed"], 0)
        second = engine.tick(self.now + timedelta(seconds=5))["results"][0]
        self.assertEqual(second["attempts"], 2)
        third = engine.tick(self.now + timedelta(seconds=15))["results"][0]
        self.assertEqual(third["status"], "failed")
        self.assertEqual(engine.tick(self.now + timedelta(days=1))["executed"], 0)
        self.assertEqual(invoked, [initial["session_id"]] * 3)
        self.assertEqual(self.query("SELECT * FROM proactive_inbox"), [])

    def test_callback_failure_and_wrong_session_id_retry(self):
        initial = self.queue()
        def raises(task):
            raise RuntimeError("offline callback")
        failed = ProactiveEngine(raises).tick(self.now)["results"][0]
        self.assertIn("offline callback", failed["last_error"])
        other = ProactiveEngine(lambda task: {"session_id": "wrong", "status": "completed"})
        self.assertEqual(other.tick(self.now + timedelta(seconds=5))["results"][0]["status"], "retry")
        self.assertEqual(tasks.get_task(initial["task_id"])["station_delivery_status"], "pending")

    def test_lease_expiry_recovery_stale_token_and_stable_session_after_restart(self):
        initial = self.queue()
        old = tasks.claim_task(self.now, lease_seconds=10)
        self.assertIsNone(tasks.claim_task(self.now + timedelta(seconds=9)))
        self.assertEqual(tasks.recover_expired_leases(self.now + timedelta(seconds=10)), 1)
        self.assertIsNone(tasks.finish_task(old["task_id"], old["claim_token"], {
            "session_id": old["session_id"], "status": "completed"}, now=self.now + timedelta(seconds=10)))
        connection.init_db()
        newer = tasks.claim_task(self.now + timedelta(seconds=15))
        self.assertEqual(newer["session_id"], initial["session_id"])
        self.assertNotEqual(newer["claim_token"], old["claim_token"])
        self.assertFalse(tasks.renew_task_lease(newer["task_id"], old["claim_token"], self.now + timedelta(seconds=15)))
        complete = tasks.finish_task(newer["task_id"], newer["claim_token"], {
            "session_id": newer["session_id"], "status": "completed",
            "report_generated": True, "station_messages": 2}, now=self.now + timedelta(seconds=15))
        self.assertEqual(complete["status"], "completed")
        self.assertEqual(complete["station_delivery_status"], "delivered")
        self.assertEqual(complete["delivery_status"], "not_configured")

    def test_lease_expiry_after_final_attempt_fails(self):
        initial = self.queue(max_attempts=1)
        tasks.claim_task(self.now, lease_seconds=1)
        tasks.recover_expired_leases(self.now + timedelta(seconds=1))
        self.assertEqual(tasks.get_task(initial["task_id"])["status"], "failed")

    def test_concurrent_claim_executes_callback_once(self):
        initial = self.queue()
        barrier = threading.Barrier(2)
        calls, results, errors = [], [], []
        lock = threading.Lock()
        def callback(task):
            with lock:
                calls.append(task["session_id"])
            return {"session_id": task["session_id"], "status": "waiting_for_user"}
        def worker():
            try:
                with connection.using_database(self.db_path):
                    barrier.wait(timeout=3)
                    results.append(ProactiveEngine(callback).tick(self.now))
            except Exception as exc:
                errors.append(exc)
        workers = [threading.Thread(target=worker) for _ in range(2)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join(timeout=5)
        self.assertFalse(errors)
        self.assertEqual(calls, [initial["session_id"]])
        self.assertEqual(sum(r["executed"] for r in results), 1)
        self.assertEqual(tasks.get_task(initial["task_id"])["attempts"], 1)

    def test_sync_only_after_execution_preserves_active_claim(self):
        initial = self.queue()
        claim = tasks.claim_task(self.now)
        self.assertEqual(tasks.sync_session_status(initial["session_id"], "completed"), 0)
        tasks.finish_task(claim["task_id"], claim["claim_token"], {
            "session_id": claim["session_id"], "status": "waiting_for_user"}, now=self.now)
        self.assertEqual(tasks.sync_session_status(initial["session_id"], "completed",
                                                  report_generated=True, station_messages=1), 1)
        final = tasks.get_task(initial["task_id"])
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["station_delivery_status"], "delivered")
        self.assertEqual(final["delivery_status"], "not_configured")
        self.assertEqual(tasks.sync_session_status(initial["session_id"], "completed",
                                                  report_generated=True, station_messages=2), 1)
        self.assertEqual(tasks.get_task(initial["task_id"])["station_messages"], 2)

    def test_partial_and_escalated_sessions_are_terminal_tasks(self):
        for index, status in enumerate(("partially_completed", "escalated")):
            initial = self.queue(event_id=f"terminal-{index}")
            first = ProactiveEngine(lambda task: {
                "session_id": task["session_id"], "status": status, "report_generated": True
            }).tick(self.now)["results"][0]
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["result"]["status"], status)
            self.assertEqual(first["station_delivery_status"], "no_recipients")
            initial = self.queue(event_id=f"restart-terminal-{index}")
            ProactiveEngine(lambda task: {"session_id": task["session_id"], "status": "waiting_for_user"}).tick(self.now)
            conn = connection.get_connection()
            try:
                with conn:
                    conn.execute("""INSERT INTO followup_sessions(session_id,patient_id,status,state_json)
                        VALUES(?,?,?,?)""", (initial["session_id"], "p", status, "{}"))
            finally:
                conn.close()
            self.assertEqual(tasks.reconcile_waiting_sessions(), 1)
            self.assertEqual(tasks.get_task(initial["task_id"])["status"], "completed")

    def test_database_scopes_keep_simulation_events_separate(self):
        initial = self.queue()
        other_path = self.db_path.parent / "simulation.sqlite"
        with connection.using_database(other_path):
            connection.init_db()
            self.assertEqual(tasks.list_tasks(), [])
        self.assertEqual(tasks.list_tasks()[0]["task_id"], initial["task_id"])

    def test_nonserializable_callback_result_is_a_finite_retry(self):
        self.queue()
        result = ProactiveEngine(lambda task: {
            "session_id": task["session_id"], "status": "completed", "invalid": object()
        }).tick(self.now)["results"][0]
        self.assertEqual(result["status"], "retry")
        self.assertIn("cannot be persisted", result["last_error"])

    def test_slow_callback_renews_lease_preventing_parallel_reclaim(self):
        started, release = threading.Event(), threading.Event()
        initial = tasks.create_task("p", "slow", "scheduled")
        invoked, errors = [], []
        def callback(task):
            invoked.append(task["session_id"])
            started.set()
            release.wait(timeout=3)
            return {"session_id": task["session_id"], "status": "waiting_for_user"}
        def worker():
            try:
                with connection.using_database(self.db_path):
                    ProactiveEngine(callback, lease_seconds=1).tick()
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=2))
            time.sleep(1.15)
            second = ProactiveEngine(callback, lease_seconds=1).tick()
            self.assertEqual(second["recovered"], 0)
            self.assertEqual(second["executed"], 0)
        finally:
            release.set()
            thread.join(timeout=4)
        self.assertFalse(errors)
        self.assertEqual(invoked, [initial["session_id"]])
        self.assertEqual(tasks.get_task(initial["task_id"])["status"], "waiting_for_user")

    def test_authorized_scope_does_not_generate_claim_recover_or_reconcile_others(self):
        tasks.create_task("other", "running", "scheduled", due_at=self.now)
        other_claim = tasks.claim_task(self.now, patient_ids=["other"], lease_seconds=1)
        other_wait = tasks.create_task("other", "waiting", "scheduled", due_at=self.now)
        other_wait_claim = tasks.claim_task(self.now, patient_ids=["other"])
        tasks.finish_task(other_wait["task_id"], other_wait_claim["claim_token"], {
            "session_id": other_wait["session_id"], "status": "waiting_for_user"}, now=self.now)
        conn = connection.get_connection()
        try:
            with conn:
                conn.execute("""INSERT INTO followup_sessions(session_id,patient_id,status,state_json)
                    VALUES(?,?,?,?)""", (other_wait["session_id"], "other", "completed", "{}"))
        finally:
            conn.close()
        tasks.create_task("other", "old-queued", "scheduled", due_at=self.now - timedelta(hours=1))
        other_plan = tasks.create_plan("other", "scheduled", self.now, plan_id="other-plan")
        before_tasks = tasks.list_tasks("other")
        tasks.create_plan("p", "scheduled", self.now, plan_id="allowed-plan")
        allowed_task = self.queue("allowed-queued")
        called = []
        def callback(task):
            called.append(task["patient_id"])
            return {"session_id": task["session_id"], "status": "waiting_for_user"}
        result = ProactiveEngine(callback, patient_ids=["p", "p"]).tick(self.now + timedelta(seconds=2))
        self.assertEqual(result["generated"], 1)
        self.assertEqual(result["recovered"], 0)
        self.assertEqual(result["reconciled"], 0)
        self.assertEqual(called, ["p", "p"])
        self.assertEqual(tasks.get_task(allowed_task["task_id"])["status"], "waiting_for_user")
        self.assertEqual(tasks.list_tasks("other"), before_tasks)
        self.assertEqual(tasks.get_plan(other_plan["plan_id"]), other_plan)
        self.assertEqual(tasks.get_task(other_claim["task_id"])["status"], "running")
        self.assertEqual(tasks.get_task(other_wait["task_id"])["status"], "waiting_for_user")

    def test_explicit_empty_scope_is_no_operation_even_at_storage_boundary(self):
        self.queue()
        tasks.create_plan("p", "scheduled", self.now)
        callback = Mock(side_effect=AssertionError("Empty scope must not invoke an Agent"))
        with patch.object(tasks, "get_connection", side_effect=AssertionError("Empty scope must not open a database")):
            result = ProactiveEngine(callback, patient_ids=[]).tick(self.now)
            self.assertEqual(result["executed"], 0)
            self.assertEqual(tasks.generate_due_tasks(self.now, patient_ids=[]), [])
            self.assertIsNone(tasks.claim_task(self.now, patient_ids=[]))
            self.assertEqual(tasks.recover_expired_leases(self.now, patient_ids=[]), 0)
            self.assertEqual(tasks.reconcile_waiting_sessions(patient_ids=[]), 0)
        callback.assert_not_called()
        self.assertEqual(tasks.list_tasks()[0]["attempts"], 0)

    def test_scope_recovery_updates_only_authorized_expired_task(self):
        allowed = self.queue("allowed-expiry")
        tasks.create_task("other", "other-expiry", "scheduled", due_at=self.now)
        tasks.claim_task(self.now, patient_ids=["p"], lease_seconds=1)
        denied = tasks.claim_task(self.now, patient_ids=["other"], lease_seconds=1)
        self.assertEqual(tasks.recover_expired_leases(self.now + timedelta(seconds=1), patient_ids=["p"]), 1)
        self.assertEqual(tasks.get_task(allowed["task_id"])["status"], "retry")
        self.assertEqual(tasks.get_task(denied["task_id"])["status"], "running")
        with self.assertRaises(ValueError):
            ProactiveEngine(lambda task: {}, patient_ids="other")


if __name__ == "__main__":
    unittest.main()

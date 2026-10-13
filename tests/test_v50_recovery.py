"""Offline fault injection across the durable V5 business chain."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.simulation_v5 import SimulationManager
from database import connection, repository, tasks
from medical_tools import guidance
from simulation.scripted import ScriptedDemoModel


class V50RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(connection.using_database(self.root / "control.db"))
        self.stack.enter_context(patch.object(connection, "DATA_DIR", self.root))
        connection.init_db()
        self.stack.enter_context(patch.object(guidance, "knowledge_revision", return_value="v50-recovery-fixture"))
        self.stack.enter_context(patch.object(guidance, "retrieve_medical_guidance", return_value=[]))
        self.manager = SimulationManager(self.root / "runs")

    def create(self):
        return self.manager.create("stable", model_mode="scripted", delay_seconds=0, max_turns=20, background=False)

    def table_counts(self, run_id):
        with connection.using_database(self.manager.store.database(run_id)):
            conn = connection.get_connection()
            try:
                return {name: conn.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                        for name in ("proactive_tasks", "followup_sessions", "bp_records", "medication_records",
                                     "reports", "report_messages")}
            finally:
                conn.close()

    def test_terminal_patient_turn_pending_in_registry_replays_without_duplicate_facts(self):
        created = self.create()
        result = self.manager.execute(created["run_id"])
        self.assertEqual(result["status"], "completed")
        before = self.table_counts(created["run_id"])
        payload = self.manager.store.get(created["run_id"], trusted=True)
        while payload["transcript"] and payload["transcript"][-1]["role"] == "agent":
            payload["transcript"].pop()
        self.assertEqual(payload["transcript"][-1]["role"], "patient")
        payload["transcript"][-1]["execution"] = "pending"
        payload["status"] = "running"
        self.manager.store.save(payload)
        with connection.using_database(self.manager.store.database(created["run_id"])):
            state = repository.read_session(result["session_id"])
            # Simulate the crash between durable Agent completion and the
            # registry's acknowledgement of the same patient turn.
            state["simulation_turn_stage"] = "input_saved"
            repository.persist_session(result["session_id"], state)
            old_user_messages = [entry for entry in state["trace"] if entry["event"] == "USER_MESSAGE"]
        restarted = SimulationManager(self.root / "runs")
        replay = restarted.execute(created["run_id"])
        self.assertEqual(replay["status"], "completed")
        self.assertEqual(replay["session_id"], result["session_id"])
        self.assertEqual(self.table_counts(created["run_id"]), before)
        with connection.using_database(self.manager.store.database(created["run_id"])):
            state = repository.read_session(result["session_id"])
            self.assertEqual([entry for entry in state["trace"] if entry["event"] == "USER_MESSAGE"], old_user_messages)
        self.assertTrue(all(row.get("execution") == "completed" for row in replay["transcript"] if row["role"] == "patient"))

    def test_expired_run_lease_can_be_reclaimed_but_live_owner_cannot_be_duplicated(self):
        created = self.create()
        self.assertTrue(self.manager.store.claim(created["run_id"], "first-owner"))
        before = self.table_counts(created["run_id"])
        self.assertEqual(self.manager.execute(created["run_id"])["status"], "queued")
        self.assertEqual(self.table_counts(created["run_id"]), before)
        self.assertFalse(self.manager.store.claim(created["run_id"], "another-owner"))
        with connection.get_connection(self.manager.store.registry) as conn:
            conn.execute("UPDATE simulation_runs SET lease_until=? WHERE run_id=?", ("2000-01-01T00:00:00+00:00", created["run_id"]))
        restarted = SimulationManager(self.root / "runs")
        result = restarted.execute(created["run_id"])
        self.assertEqual(result["status"], "completed")
        with connection.get_connection(self.manager.store.registry) as conn:
            owner = conn.execute("SELECT owner,lease_until FROM simulation_runs WHERE run_id=?", (created["run_id"],)).fetchone()
        self.assertIsNone(owner["owner"])
        self.assertIsNone(owner["lease_until"])
        self.assertEqual(self.table_counts(created["run_id"])["proactive_tasks"], 1)

    def test_transient_startup_planner_failure_retries_persisted_task_and_same_session(self):
        created = self.create()
        scripted = ScriptedDemoModel()
        observed_retry = []
        failed_once = False

        def model(**kwargs):
            nonlocal failed_once
            payload = json.loads(kwargs["user_prompt"])
            if "trigger_event" in payload:
                task = tasks.list_tasks("simulation-stable")[0]
                observed_retry.append({"task_id": task["task_id"], "session_id": task["session_id"], "attempts": task["attempts"]})
                if not failed_once:
                    failed_once = True
                    return None
            return scripted(**kwargs)

        result = self.manager.execute(created["run_id"], model_override=model)
        self.assertEqual(result["status"], "completed", result)
        self.assertGreaterEqual(len(observed_retry), 2)
        self.assertEqual({entry["task_id"] for entry in observed_retry}, {result["task_id"]})
        self.assertEqual({entry["session_id"] for entry in observed_retry}, {result["session_id"]})
        self.assertEqual(observed_retry[0]["attempts"], 1)
        self.assertEqual(observed_retry[1]["attempts"], 2)
        with connection.using_database(self.manager.store.database(created["run_id"])):
            task = tasks.get_task(result["task_id"])
            state = repository.read_session(result["session_id"])
        self.assertEqual(task["attempts"], 2)
        self.assertEqual(task["status"], "completed")
        calls = [entry["detail"] for entry in state["trace"] if entry["event"] == "LLM_CALL" and entry["detail"].get("purpose") == "PLAN"]
        self.assertTrue(calls[0]["called"])
        self.assertFalse(calls[0]["ok"])
        self.assertTrue(any(call["ok"] for call in calls[1:]))
        self.assertEqual(self.table_counts(created["run_id"])["proactive_tasks"], 1)
        self.assertEqual(self.table_counts(created["run_id"])["reports"], 1)
        self.assertEqual(self.table_counts(created["run_id"])["report_messages"], 2)
        self.assertEqual(self.manager.view(created["run_id"], created["run_token"])["real_model_requests"], 0)

    def test_summary_model_failure_is_recorded_as_failure_with_only_recorded_facts(self):
        created = self.create()
        scripted = ScriptedDemoModel()

        def model(**kwargs):
            if "facts" in json.loads(kwargs["user_prompt"]):
                return None
            return scripted(**kwargs)

        result = self.manager.execute(created["run_id"], model_override=model)
        self.assertEqual(result["status"], "completed", result)
        with connection.using_database(self.manager.store.database(created["run_id"])):
            state = repository.read_session(result["session_id"])
        report = state["report"]
        self.assertEqual(report["ai_summary"]["status"], "failed")
        self.assertEqual(report["ai_summary"]["reason"], "model_unavailable")
        self.assertEqual(report["doctor_view"]["summary_kind"], "recorded_facts")
        self.assertEqual(report["family_view"]["ai_summary_status"], "failed")
        self.assertIn("132/78", report["ai_summary"]["summary"])
        self.assertEqual(result["evaluation"]["observations"]["report_ai_status"], "failed")
        self.assertEqual(result["evaluation"]["metrics"]["report_consistency"]["status"], "passed")
        summaries = [entry["detail"] for entry in state["trace"] if entry["event"] == "LLM_CALL"
                     and entry["detail"].get("purpose") == "REPORT_SUMMARY"]
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0]["called"])
        self.assertFalse(summaries[0]["ok"])
        self.assertFalse(summaries[0]["deepseek_called"])
        self.assertEqual(self.table_counts(created["run_id"])["reports"], 1)
        self.assertEqual(self.table_counts(created["run_id"])["report_messages"], 2)


if __name__ == "__main__":
    unittest.main()

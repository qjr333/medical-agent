"""Execution isolation and resumability using controllable offline Agent calls."""

import contextlib
import copy
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.state_manager import create_patient_state
from backend import runtime_v5
from database import connection, repository


class V50RuntimeConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.path = directory / "runtime.sqlite"
        self.stack.enter_context(patch.object(connection, "DB_PATH", self.path))
        self.stack.enter_context(patch.object(connection, "DATA_DIR", directory))
        self.stack.enter_context(connection.using_database(self.path))
        connection.init_db()
        with connection.get_connection() as conn:
            conn.execute("INSERT INTO patients(patient_id,patient_name,age,conditions_json) VALUES(?,?,?,?)",
                         ("runtime-fictional", "虚构并发患者", 72, '["高血压"]'))
        self.stack.enter_context(patch.object(runtime_v5, "refresh_patient_memory"))
        self.extract = self.stack.enter_context(patch.object(runtime_v5, "extract_patient_information",
                                                            return_value={"topic_status": {}}))
        self.runtime = runtime_v5.AgentRuntime(model=lambda **kwargs: {})
        self.threads = []
        self.failures = []
        self.addCleanup(self.join_threads)

    def seed(self, sid):
        state = create_patient_state("scheduled", DEMO_PATIENT={
            "patient_id": "runtime-fictional", "patient_name": "虚构并发患者", "age": 72,
            "conditions": ["高血压"]
        }, DEMO_BP_HISTORY=[], TRIGGER_CONFIGS={"scheduled": {
            "followup_reason": "虚构运行服务测试", "bp_systolic": None, "bp_diastolic": None
        }})
        state.update(agent_version="V5.0", session_id=sid, followup_status="waiting_for_user")
        repository.persist_session(sid, state)

    def spawn(self, operation):
        def execute():
            try:
                with connection.using_database(self.path):
                    operation()
            except BaseException as error:
                self.failures.append(error)
        thread = threading.Thread(target=execute, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    def join_threads(self):
        for thread in self.threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "An execution worker failed to exit")
        self.assertEqual(self.failures, [])

    @staticmethod
    def waiting(state, *args, **kwargs):
        state["followup_status"] = "waiting_for_user"
        return {"message": "请继续回答当前随访问题。"}

    def test_different_sessions_progress_while_one_agent_call_is_blocked(self):
        """The lock covers one session, not every patient and model request."""
        for sid in ("slow-session", "independent-session"):
            self.seed(sid)
        slow_entered, release, independent_done = threading.Event(), threading.Event(), threading.Event()

        def planner(state, *args, **kwargs):
            if state["session_id"] == "slow-session":
                slow_entered.set()
                if not release.wait(3):
                    raise AssertionError("The slow Agent call was not released")
            return self.waiting(state)

        try:
            with patch.object(runtime_v5, "run_followup_agent", side_effect=planner):
                self.spawn(lambda: self.runtime.turn("slow-session", "第一位患者回答"))
                self.assertTrue(slow_entered.wait(1))
                self.spawn(lambda: (self.runtime.turn("independent-session", "第二位患者回答"), independent_done.set()))
                self.assertTrue(independent_done.wait(1), "An unrelated session was blocked by the first Agent call")
                self.assertFalse(release.is_set())
                release.set()
                self.join_threads()
        finally:
            release.set()

    def test_same_session_serializes_inputs_and_keeps_both_patient_messages(self):
        self.seed("shared-session")
        first_entered, release = threading.Event(), threading.Event()
        second_attempted, second_extracted = threading.Event(), threading.Event()
        calls = []

        def extraction(message, *args, **kwargs):
            if message == "第二条患者回答":
                second_extracted.set()
            return {"topic_status": {}}

        def planner(state, *args, **kwargs):
            calls.append(state["latest_patient_message"])
            if len(calls) == 1:
                first_entered.set()
                if not release.wait(3):
                    raise AssertionError("The first same-session call was not released")
            return self.waiting(state)

        def second():
            second_attempted.set()
            self.runtime.turn("shared-session", "第二条患者回答")

        try:
            with patch.object(runtime_v5, "run_followup_agent", side_effect=planner), \
                    patch.object(runtime_v5, "extract_patient_information", side_effect=extraction):
                self.spawn(lambda: self.runtime.turn("shared-session", "第一条患者回答"))
                self.assertTrue(first_entered.wait(1))
                self.spawn(second)
                self.assertTrue(second_attempted.wait(1))
                self.assertFalse(second_extracted.wait(0.1), "Concurrent same-session input entered extraction")
                release.set()
                self.join_threads()
        finally:
            release.set()
        state = repository.read_session("shared-session")
        messages = [entry["detail"]["message"] for entry in state["trace"] if entry["event"] == "USER_MESSAGE"]
        self.assertEqual(messages, ["第一条患者回答", "第二条患者回答"])
        self.assertEqual(calls, messages)

    def test_internal_resume_keeps_the_input_once_and_needs_no_patient_continue(self):
        self.seed("resume-session")
        calls = []

        def planner(state, *args, **kwargs):
            calls.append(copy.deepcopy(state["trace"]))
            state["followup_status"] = "processing" if len(calls) == 1 else "waiting_for_user"
            return {"message": "处理记录中。" if len(calls) == 1 else "请确认今天的服药情况。"}

        with patch.object(runtime_v5, "run_followup_agent", side_effect=planner):
            result = self.runtime.turn("resume-session", "这是本次患者的回答")
        self.assertEqual(len(calls), 2)
        self.extract.assert_called_once()
        patient_messages = [entry["detail"]["message"] for entry in result["trace"] if entry["event"] == "USER_MESSAGE"]
        self.assertEqual(patient_messages, ["这是本次患者的回答"])
        self.assertEqual(result["followup_status"], "waiting_for_user")
        self.assertEqual(result["last_agent_response"], "请确认今天的服药情况。")
        self.assertEqual(sum(entry["event"] == "EXECUTION_RESUMED" for entry in result["trace"]), 1)
        self.assertIsNone(result.get("report"))

    def test_unrecoverable_processing_stops_as_interrupted_without_fake_report(self):
        self.seed("limited-session")

        def planner(state, *args, **kwargs):
            state["followup_status"] = "processing"
            return {"message": "处理记录中。"}

        with patch.object(runtime_v5, "run_followup_agent", side_effect=planner) as run:
            result = self.runtime.turn("limited-session")
        self.assertEqual(run.call_count, 4)
        self.assertEqual(result["followup_status"], "interrupted")
        self.assertEqual(result["interruption_reason"], "automatic_resume_exhausted")
        self.assertFalse(any(entry["event"] == "USER_MESSAGE" for entry in result["trace"]))
        self.assertIsNone(result.get("report"))
        with connection.get_connection() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()

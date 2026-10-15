"""Durable V5 input boundaries using fictional patients and isolated SQLite."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from datetime import timedelta, date
from pathlib import Path
from unittest.mock import patch

from agent import followup_agent
from agent.state_manager import create_patient_state
from backend import runtime_v5, v5_api
from backend.utils import today_local_date
from database import connection, repository


TEXT = "今天血压132/78"
CANDIDATE = {"claims": [{"id": "bp", "kind": "bp", "value": {"systolic": 132, "diastolic": 78},
                        "evidence": TEXT, "temporal": "current", "polarity": "affirmed"}],
             "answer_status": "answered", "screening": {"adequate": False}}


class InjectedCrash(BaseException):
    """Bypass ordinary model-error recovery to emulate a stopped process."""


class ExtractionModel:
    def __init__(self):
        self.extractions = self.verifications = 0

    def __call__(self, **kwargs):
        payload = json.loads(kwargs["user_prompt"])
        if "claims" not in payload:
            self.extractions += 1
            return copy.deepcopy(CANDIDATE)
        self.verifications += 1
        return {"reviews": [{"id": "bp", "supported": True, "polarity": "affirmed", "temporal": "current"}],
                "screening": {"adequate": False}}


class RuntimeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(connection.using_database(self.root / "checkpoint.sqlite"))
        connection.init_db()
        self.patient = {"patient_id": "checkpoint-fictional", "patient_name": "虚构检查点患者", "age": 72,
                        "conditions": ["高血压"]}
        repository.seed_demo_patient(self.patient, [])
        self.sid = "checkpoint-session"
        state = create_patient_state("scheduled", DEMO_PATIENT=self.patient, DEMO_BP_HISTORY=[],
                                     TRIGGER_CONFIGS={"scheduled": {"followup_reason": "虚构检查点测试",
                                                                   "bp_systolic": None, "bp_diastolic": None}})
        state.update(agent_version="V5.0", session_id=self.sid, followup_status="waiting_for_user",
                     current_goal="bp", last_agent_question="今天血压是多少？",
                     question_context={"question": "今天血压是多少？", "topic": "bp", "kind": "bp"})
        repository.persist_session(self.sid, state)
        self.stack.enter_context(patch.object(runtime_v5, "refresh_patient_memory"))
        self.stack.enter_context(patch.object(runtime_v5, "execution_fingerprint", return_value="unchanged"))
        self.stack.enter_context(patch.object(runtime_v5, "run_followup_agent", side_effect=self.waiting))

    @staticmethod
    def waiting(state, **kwargs):
        state["followup_status"] = "waiting_for_user"
        state.pop("execution_pending", None)
        return {"message": "谢谢，今天服药情况怎么样？"}

    def reload(self):
        repository.sessions.pop(self.sid, None)
        return repository.read_session(self.sid)

    def bp_count(self):
        with connection.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM bp_records").fetchone()[0]

    def assert_one_input(self, state):
        messages = [entry["detail"]["message"] for entry in state["trace"] if entry["event"] == "USER_MESSAGE"]
        self.assertEqual(messages, [TEXT])

    def stop_before_extraction(self, *, turn_index=None):
        with patch.object(runtime_v5, "extract_patient_information", side_effect=InjectedCrash):
            with self.assertRaises(InjectedCrash):
                runtime_v5.AgentRuntime().turn(self.sid, TEXT, turn_index=turn_index)

    def stop_before_verification(self):
        original_call = followup_agent._call

        def stop(model, state, purpose, *args, **kwargs):
            if purpose == "VERIFY_EXTRACT":
                raise InjectedCrash()
            return original_call(model, state, purpose, *args, **kwargs)

        model = ExtractionModel()
        with patch.object(followup_agent, "_call", side_effect=stop):
            with self.assertRaises(InjectedCrash):
                runtime_v5.AgentRuntime(model).turn(self.sid, TEXT)
        self.assertEqual((model.extractions, model.verifications), (1, 0))

    def test_received_input_is_durable_before_any_model_request(self):
        self.stop_before_extraction()
        saved = self.reload()
        self.assertEqual(saved["latest_patient_message"], TEXT)
        self.assertEqual(saved["followup_status"], "processing")
        self.assertEqual(saved["extraction_pending"]["status"], "awaiting_extraction")
        self.assertIsNone(saved["bp_systolic"])
        self.assertEqual(self.bp_count(), 0)
        self.assert_one_input(saved)
        model = ExtractionModel()
        resumed = runtime_v5.AgentRuntime(model).turn(self.sid)
        self.assertEqual((model.extractions, model.verifications), (1, 1))
        self.assertEqual(resumed["bp_systolic"], 132)
        self.assertEqual(self.bp_count(), 1)
        self.assert_one_input(resumed)

    def test_accept_message_only_saves_input_and_repeated_acceptance_is_idempotent(self):
        model = ExtractionModel()
        runtime = runtime_v5.AgentRuntime(model)
        runtime.accept_message(self.sid, TEXT)
        runtime.accept_message(self.sid, TEXT)
        saved = self.reload()
        self.assertEqual((model.extractions, model.verifications), (0, 0))
        self.assert_one_input(saved)
        self.assertIsNone(saved["bp_systolic"])
        self.assertEqual(self.bp_count(), 0)

    def test_dispatch_releases_reservation_when_input_checkpoint_fails(self):
        key = (str(connection.current_database_path()), "turn:" + self.sid)
        with patch.object(v5_api.threading, "Thread") as thread, \
                patch.object(repository, "persist_session", side_effect=RuntimeError("injected write failure")):
            with self.assertRaises(RuntimeError):
                v5_api.dispatch(lambda: None, "turn:" + self.sid,
                                before_start=lambda: runtime_v5.AgentRuntime().accept_message(self.sid, TEXT))
        thread.assert_not_called()
        with v5_api._job_lock:
            self.assertNotIn(key, v5_api._jobs)
        self.assertEqual(self.reload()["followup_status"], "waiting_for_user")
        self.assertNotIn("extraction_pending", repository.read_session(self.sid))

    def test_dispatch_busy_reservation_never_accepts_a_second_input(self):
        key = (str(connection.current_database_path()), "turn:" + self.sid)
        runtime = runtime_v5.AgentRuntime()
        try:
            with patch.object(v5_api.threading, "Thread"):
                self.assertTrue(v5_api.dispatch(lambda: None, "turn:" + self.sid,
                                               before_start=lambda: runtime.accept_message(self.sid, TEXT)))
                self.assertFalse(v5_api.dispatch(lambda: None, "turn:" + self.sid,
                                                before_start=lambda: self.fail("Busy dispatch accepted new input")))
                # Recovery cannot create a second worker for the accepted input.
                v5_api.recover_pending_sessions()
            self.assert_one_input(self.reload())
            self.assertNotIn("background_recovery_attempts", repository.read_session(self.sid))
        finally:
            with v5_api._job_lock:
                v5_api._jobs.discard(key)

    def test_candidate_is_durable_before_verification_and_resumes_without_extracting(self):
        self.stop_before_verification()
        saved = self.reload()
        self.assertEqual(saved["extraction_pending"]["status"], "awaiting_verification")
        self.assertEqual(saved["extraction_pending"]["model"], CANDIDATE)
        self.assertIsNone(saved["bp_systolic"])
        self.assertFalse(saved["coverage"]["bp"])
        self.assertEqual(self.bp_count(), 0)
        model = ExtractionModel()
        resumed = runtime_v5.AgentRuntime(model).turn(self.sid)
        self.assertEqual((model.extractions, model.verifications), (0, 1))
        self.assertEqual(resumed["bp_systolic"], 132)
        self.assertNotIn("extraction_pending", resumed)
        self.assert_one_input(resumed)
        self.assertEqual(self.bp_count(), 1)

    def test_generation_outage_retains_extract_checkpoint_without_confirming_facts(self):
        def timeout(**kwargs):
            raise TimeoutError("injected provider outage")

        with patch.object(runtime_v5, "run_followup_agent", side_effect=InjectedCrash):
            with self.assertRaises(InjectedCrash):
                runtime_v5.AgentRuntime(timeout).turn(self.sid, TEXT)
        saved = self.reload()
        self.assertEqual(saved["extraction_pending"]["status"], "awaiting_extraction")
        self.assertEqual(saved["execution_pending"]["next_action"], "EXTRACT")
        self.assertIsNone(saved["bp_systolic"])
        self.assertEqual(self.bp_count(), 0)
        resumed = runtime_v5.AgentRuntime(ExtractionModel()).turn(self.sid)
        self.assertEqual(resumed["bp_systolic"], 132)
        self.assert_one_input(resumed)
        self.assertEqual(self.bp_count(), 1)

    def test_same_pending_answer_can_be_retried_without_duplicate_user_message(self):
        self.stop_before_verification()
        self.reload()
        model = ExtractionModel()
        resumed = runtime_v5.AgentRuntime(model).turn(self.sid, TEXT)
        self.assertEqual((model.extractions, model.verifications), (0, 1))
        self.assert_one_input(resumed)
        self.assertEqual(self.bp_count(), 1)

    def test_crash_after_fact_commit_restarts_planning_without_reapplying_fact(self):
        with patch.object(runtime_v5, "run_followup_agent", side_effect=InjectedCrash):
            with self.assertRaises(InjectedCrash):
                runtime_v5.AgentRuntime(ExtractionModel()).turn(self.sid, TEXT)
        saved = self.reload()
        self.assertEqual(saved["followup_status"], "processing")
        self.assertNotIn("extraction_pending", saved)
        self.assertEqual(saved["bp_systolic"], 132)
        self.assertEqual(self.bp_count(), 1)
        with patch.object(runtime_v5, "extract_patient_information") as extract:
            resumed = runtime_v5.AgentRuntime().turn(self.sid)
        extract.assert_not_called()
        self.assert_one_input(resumed)
        self.assertEqual(self.bp_count(), 1)

    def test_failed_fact_transaction_keeps_cache_and_sqlite_unconfirmed(self):
        persist = repository.persist_session

        def fail_fact_commit(*args, **kwargs):
            if kwargs.get("new_bp"):
                raise RuntimeError("injected SQLite write failure")
            return persist(*args, **kwargs)

        with patch.object(repository, "persist_session", side_effect=fail_fact_commit):
            with self.assertRaises(RuntimeError):
                runtime_v5.AgentRuntime(ExtractionModel()).turn(self.sid, TEXT)
        # A mutable cache reference must not leak the uncommitted merge.
        self.assertIsNone(repository.read_session(self.sid)["bp_systolic"])
        self.assertIsNone(self.reload()["bp_systolic"])
        self.assertEqual(self.bp_count(), 0)
        resumed = runtime_v5.AgentRuntime(ExtractionModel()).turn(self.sid)
        self.assertEqual(resumed["bp_systolic"], 132)
        self.assert_one_input(resumed)
        self.assertEqual(self.bp_count(), 1)

    def test_simulation_replay_recovers_input_received_without_duplicate_write(self):
        self.stop_before_extraction(turn_index=3)
        saved = self.reload()
        self.assertEqual(saved["simulation_turn_stage"], "input_received")
        model = ExtractionModel()
        resumed = runtime_v5.AgentRuntime(model).turn(self.sid, TEXT, turn_index=3)
        self.assertEqual((model.extractions, model.verifications), (1, 1))
        self.assert_one_input(resumed)
        self.assertEqual(resumed["simulation_turn_stage"], "completed")
        runtime_v5.AgentRuntime(model).turn(self.sid, TEXT, turn_index=3)
        self.assertEqual((model.extractions, model.verifications), (1, 1))
        self.assertEqual(self.bp_count(), 1)

    def test_interrupted_received_input_is_recovered_by_background_worker(self):
        self.stop_before_extraction()
        saved = self.reload()
        saved.update(followup_status="interrupted", interruption_reason="background_execution_failed")
        repository.persist_session(self.sid, saved)
        runtime = runtime_v5.AgentRuntime(ExtractionModel())
        with patch.object(v5_api, "AgentRuntime", return_value=runtime), \
                patch.object(v5_api, "dispatch", side_effect=lambda job, key: job()):
            v5_api.recover_pending_sessions()
        resumed = self.reload()
        self.assertEqual(resumed["followup_status"], "waiting_for_user")
        self.assert_one_input(resumed)
        self.assertEqual(self.bp_count(), 1)

    def test_stale_received_input_requires_new_answer_without_calling_model(self):
        self.stop_before_extraction()
        saved = self.reload()
        saved["extraction_pending"]["date"] = "2000-01-01"
        repository.persist_session(self.sid, saved)
        with patch.object(runtime_v5, "extract_patient_information") as extract:
            resumed = runtime_v5.AgentRuntime().turn(self.sid)
        extract.assert_not_called()
        self.assertEqual(resumed["interruption_reason"], "stale_extraction_context")
        self.assertEqual(self.bp_count(), 0)

    def test_historical_medication_date_survives_sqlite_roundtrip(self):
        observed = (date.fromisoformat(today_local_date()) - timedelta(days=2)).isoformat()

        def extract(message, state, **kwargs):
            state.pop("extraction_pending", None)
            return {"topic_status": {}, "_llm_v4_verified": True, "semantic_validated": True,
                    "medication_observations": [{"date": observed, "status": "missed", "detail": "那天忘记吃药了"}]}

        with patch.object(runtime_v5, "extract_patient_information", side_effect=extract):
            resumed = runtime_v5.AgentRuntime().turn(self.sid, "前天漏服过")
        self.assertEqual(resumed["medication_status"], "unknown")
        self.assertEqual(resumed["medication_observations"][observed]["status"], "missed")
        with connection.get_connection() as conn:
            row = conn.execute("SELECT status,detail FROM medication_records").fetchone()
        self.assertEqual(row["status"], "missed")
        self.assertEqual(row["detail"], f"服药日期：{observed}；那天忘记吃药了")

    def test_internal_planning_resumes_share_one_deadline(self):
        clock = [100.0]
        windows = []

        def planner(state, **kwargs):
            windows.append(kwargs["max_seconds"])
            clock[0] += 50
            state["followup_status"] = "processing" if len(windows) < 3 else "waiting_for_user"
            return {"message": "正在整理。" if len(windows) < 3 else "现在还有哪里不舒服？"}

        with patch.object(runtime_v5, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(runtime_v5.config, "V5_REQUEST_TIMEOUT_SECONDS", 120), \
                patch.object(runtime_v5, "run_followup_agent", side_effect=planner):
            resumed = runtime_v5.AgentRuntime().turn(self.sid)
        self.assertEqual(windows, [120, 70, 20])
        self.assertEqual(resumed["followup_status"], "waiting_for_user")
        self.assertFalse(any(entry["event"] == "USER_MESSAGE" for entry in resumed["trace"]))

    def test_expired_planning_window_pauses_without_empty_automatic_resumes(self):
        clock = [100.0]
        windows = []

        def planner(state, **kwargs):
            windows.append(kwargs["max_seconds"])
            clock[0] += 120
            state["followup_status"] = "processing"
            state["execution_pending"] = {"next_action": "PLAN", "plan_calls": 2, "input_count": 0}
            return {"message": "正在整理。"}

        with patch.object(runtime_v5, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(runtime_v5.config, "V5_REQUEST_TIMEOUT_SECONDS", 120), \
                patch.object(runtime_v5, "run_followup_agent", side_effect=planner):
            resumed = runtime_v5.AgentRuntime().turn(self.sid)
        self.assertEqual(windows, [120])
        self.assertEqual(resumed["followup_status"], "interrupted")
        self.assertEqual(resumed["interruption_reason"], "planning_time_budget")
        self.assertEqual(resumed["execution_pending"]["plan_calls"], 2)
        self.assertFalse(any(entry["event"] == "EXECUTION_RESUMED" for entry in resumed["trace"]))
        self.assertIsNone(resumed.get("report"))
        self.assertEqual(self.bp_count(), 0)
        with patch.object(runtime_v5, "run_followup_agent", side_effect=self.waiting):
            recovered = runtime_v5.AgentRuntime().turn(self.sid)
        self.assertEqual(recovered["followup_status"], "waiting_for_user")
        self.assertEqual(self.bp_count(), 0)
        self.assertFalse(any(entry["event"] == "USER_MESSAGE" for entry in recovered["trace"]))


if __name__ == "__main__":
    unittest.main()

"""Offline timeout/recovery regressions. No real provider or patient database."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch, Mock

from agent import llm_v4
from agent.state_manager import create_patient_state, merge_user_patch, add_trace
from agent.trace_view import present_trace
from backend import config, runtime_v5
from backend.simulation_v5 import SimulationManager
from database import connection, repository
from simulation.scripted import ScriptedDemoModel


TEXT = "今天早晨量的血压132/78"
CANDIDATE = {"claims": [{"id": "bp", "kind": "bp", "value": {"systolic": 132, "diastolic": 78},
                        "evidence": TEXT, "temporal": "current", "polarity": "affirmed"}],
             "answer_status": "answered", "screening": {"adequate": False}}


def state():
    result = create_patient_state("scheduled", DEMO_PATIENT={
        "patient_id": "recovery-fictional", "patient_name": "虚构恢复患者", "age": 72, "conditions": ["高血压"]},
        DEMO_BP_HISTORY=[], TRIGGER_CONFIGS={"scheduled": {
            "followup_reason": "模拟恢复随访", "bp_systolic": None, "bp_diastolic": None}})
    result.update(agent_version="V5.0", last_agent_question="今天血压是多少？", current_goal="bp",
                  question_context={"question": "今天血压是多少？", "topic": "bp", "kind": "bp"},
                  latest_patient_message=TEXT)
    return result


class ExtractionModel(ScriptedDemoModel):
    def __init__(self, timeouts=0, reject=False):
        self.timeouts = timeouts
        self.reject = reject
        self.extractions = self.verifications = 0

    def __call__(self, **kwargs):
        payload = json.loads(kwargs["user_prompt"])
        if "patient_message" in payload and "trigger_event" not in payload:
            if "claims" not in payload:
                self.extractions += 1
                return deepcopy(CANDIDATE)
            self.verifications += 1
            if self.verifications <= self.timeouts:
                raise TimeoutError("offline timeout")
            return {"reviews": [{"id": item["id"], "supported": not self.reject,
                                 "polarity": item["polarity"], "temporal": item["temporal"]}
                                for item in payload["claims"]], "screening": {"adequate": False}}
        return super().__call__(**kwargs)


class ValidationRecoveryV5Tests(unittest.TestCase):
    def test_first_validation_timeout_retries_once_without_regenerating(self):
        patient = state()
        model = ExtractionModel(timeouts=1)
        extracted = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=model, enable_v50=True)
        self.assertEqual((model.extractions, model.verifications), (1, 2))
        self.assertEqual(extracted["bp_systolic"], 132)
        self.assertNotIn("extraction_pending", patient)
        merge_user_patch(patient, extracted, TEXT)
        response = llm_v4.run_agent_v4(patient, call_json_model=model, enable_v50=True)
        self.assertEqual(patient["followup_status"], "waiting_for_user")
        self.assertIn("？", response["message"])
        retries = [row for row in patient["trace"] if row["event"] == "MODEL_RETRY"]
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["detail"]["purpose"], "VERIFY_EXTRACT")

    def test_permanent_validation_timeout_keeps_candidate_unconfirmed_and_pauses(self):
        patient = state()
        model = ExtractionModel(timeouts=10)
        extracted = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=model, enable_v50=True)
        merge_user_patch(patient, extracted, TEXT)
        self.assertIsNone(patient["bp_systolic"])
        self.assertFalse(patient["coverage"]["bp"])
        self.assertEqual(patient["extraction_pending"]["model"], CANDIDATE)
        llm_v4.run_agent_v4(patient, call_json_model=model, enable_v50=True)
        self.assertEqual(patient["followup_status"], "interrupted")
        self.assertEqual(patient["execution_pending"]["next_action"], "VERIFY_EXTRACT")
        self.assertEqual((model.extractions, model.verifications), (1, 2))
        events = [row["event"] for row in patient["trace"]]
        self.assertIn("MODEL_RETRY_EXHAUSTED", events)
        self.assertIn("SEMANTIC_VALIDATION_PENDING", events)
        self.assertIn("SAFETY_PAUSE", events)
        self.assertIsNone(patient.get("report"))

    def test_verified_candidate_restores_from_json_and_skips_extract(self):
        patient = state()
        first = ExtractionModel(timeouts=10)
        result = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=first, enable_v50=True)
        merge_user_patch(patient, result, TEXT)
        patient = json.loads(json.dumps(patient))
        resumed = ExtractionModel()
        result = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=resumed, enable_v50=True)
        self.assertEqual((resumed.extractions, resumed.verifications), (0, 1))
        self.assertTrue(result["semantic_validated"])
        self.assertEqual(result["bp_systolic"], 132)
        self.assertNotIn("extraction_pending", patient)

    def test_new_patient_message_does_not_reuse_an_old_candidate(self):
        patient = state()
        llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=ExtractionModel(10), enable_v50=True)
        model = Mock(return_value={"claims": [], "answer_status": "unclear"})
        result = llm_v4.extract_user_patch_v4("刚才记错了，还没量", patient, call_json_model=model, enable_v50=True)
        self.assertIsNone(result["bp_systolic"])
        self.assertNotIn("extraction_pending", patient)
        self.assertIn("EXTRACTION_DISCARDED", [row["event"] for row in patient["trace"]])

    def test_semantic_rejection_is_not_retried_as_a_transport_timeout(self):
        patient = state()
        model = ExtractionModel(reject=True)
        result = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=model, enable_v50=True)
        self.assertEqual(model.verifications, 1)
        self.assertIsNone(result["bp_systolic"])
        self.assertNotIn("extraction_pending", patient)
        self.assertIn("SEMANTIC_VALIDATION_REJECTED", [row["event"] for row in patient["trace"]])

    def test_non_temporary_generation_failure_has_no_retry(self):
        patient = state()
        model = Mock(side_effect=RuntimeError("offline invalid input"))
        result = llm_v4._call(model, patient, "EXTRACT", "test", {})
        self.assertIsNone(result)
        self.assertEqual(model.call_count, 1)
        self.assertNotIn("MODEL_RETRY", [row["event"] for row in patient["trace"]])

    def test_default_provider_retries_plain_mode_without_repeating_json_probe(self):
        from backend import model_client
        from types import SimpleNamespace
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
        create = Mock(side_effect=[RuntimeError("response_format unsupported"), TimeoutError("offline"), response])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        patient = state()
        with patch.object(model_client, "client", sdk):
            result = llm_v4._call(None, patient, "VERIFY_EXTRACT", "test", {})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(create.call_count, 3)
        self.assertNotIn("response_format", create.call_args_list[-1].kwargs)
        starts = sum(row["detail"].get("provider_request_count", 0) for row in present_trace(patient))
        self.assertEqual(starts, 3)

    def test_extraction_and_validation_have_independent_time_windows(self):
        patient = state()
        clock = [100.0]
        windows = []
        model = ExtractionModel()
        def bounded(function, deadline):
            windows.append(deadline - clock[0])
            answer = function()
            clock[0] += 45
            return answer
        with patch.object(llm_v4, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(llm_v4, "_bounded_call", side_effect=bounded), \
                patch.object(config, "DEEPSEEK_TIMEOUT_SECONDS", 50):
            result = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=model, enable_v50=True)
        self.assertEqual(windows, [50, 50])
        self.assertEqual(result["bp_systolic"], 132)

    def test_trace_shows_timeout_retry_rejection_and_pause_without_candidate_leak(self):
        patient = state()
        result = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=ExtractionModel(10), enable_v50=True)
        merge_user_patch(patient, result, TEXT)
        llm_v4.run_agent_v4(patient, enable_v50=True)
        shown = present_trace(patient)
        self.assertIn("MODEL_RETRY_EXHAUSTED", [entry["event"] for entry in shown])
        timeouts = [entry for entry in shown if entry["event"] == "LLM_CALL" and entry["detail"].get("failure_kind") == "timeout"]
        self.assertEqual(len(timeouts), 2)
        self.assertNotIn(TEXT, json.dumps(shown, ensure_ascii=False))
        self.assertNotIn("claims", json.dumps(shown))

    def test_sqlite_resume_verifies_once_and_persists_bp_without_duplicate_input(self):
        with tempfile.TemporaryDirectory() as directory, connection.using_database(Path(directory) / "recovery.db"):
            connection.init_db()
            patient = state()
            sid = "fictional-recovery"
            patient["session_id"] = sid
            repository.seed_demo_patient({key: patient[key] for key in ("patient_id", "patient_name", "age", "conditions")}, [])
            add_trace(patient, "USER_MESSAGE", {"message": TEXT})
            result = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=ExtractionModel(10), enable_v50=True)
            merge_user_patch(patient, result, TEXT)
            llm_v4.run_agent_v4(patient, enable_v50=True)
            repository.persist_session(sid, patient)
            with connection.get_connection() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM bp_records").fetchone()[0], 0)
            resumed = ExtractionModel()
            runtime = runtime_v5.AgentRuntime(resumed)
            loaded = runtime.turn(sid)
            self.assertEqual((resumed.extractions, resumed.verifications), (0, 1))
            self.assertEqual(loaded["bp_systolic"], 132)
            self.assertEqual(loaded["followup_status"], "waiting_for_user")
            self.assertEqual(sum(row["event"] == "USER_MESSAGE" for row in loaded["trace"]), 1)
            with connection.get_connection() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM bp_records").fetchone()[0], 1)
            self.assertNotIn("extraction_pending", loaded)

    def test_stale_resume_context_never_regenerates_old_answer_as_a_new_fact(self):
        for change in ("date", "question"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory, \
                    connection.using_database(Path(directory) / "stale.db"):
                connection.init_db()
                patient = state()
                patient["session_id"] = "stale-recovery"
                repository.seed_demo_patient({key: patient[key] for key in ("patient_id", "patient_name", "age", "conditions")}, [])
                llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=ExtractionModel(10), enable_v50=True)
                if change == "date":
                    patient["extraction_pending"]["date"] = "2000-01-01"
                else:
                    patient["last_agent_question"] = "另一个新的问题？"
                patient["followup_status"] = "interrupted"
                repository.persist_session(patient["session_id"], patient)
                model = ExtractionModel()
                loaded = runtime_v5.AgentRuntime(model).turn(patient["session_id"])
                self.assertEqual((model.extractions, model.verifications), (0, 0))
                self.assertIsNone(loaded["bp_systolic"])
                self.assertEqual(loaded["extraction_pending"]["status"], "stale_context")
                with connection.get_connection() as conn:
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM bp_records").fetchone()[0], 0)

    def test_background_recovery_revalidates_durable_interrupted_candidate(self):
        from backend import v5_api
        with tempfile.TemporaryDirectory() as directory, connection.using_database(Path(directory) / "background.db"):
            connection.init_db()
            patient = state()
            patient["session_id"] = "background-recovery"
            repository.seed_demo_patient({key: patient[key] for key in ("patient_id", "patient_name", "age", "conditions")}, [])
            result = llm_v4.extract_user_patch_v4(TEXT, patient, call_json_model=ExtractionModel(10), enable_v50=True)
            merge_user_patch(patient, result, TEXT)
            llm_v4.run_agent_v4(patient, enable_v50=True)
            repository.persist_session(patient["session_id"], patient)
            model = ExtractionModel()
            runtime = runtime_v5.AgentRuntime(model)
            with patch.object(v5_api, "AgentRuntime", return_value=runtime), \
                    patch.object(v5_api, "dispatch", side_effect=lambda job, key: job()):
                v5_api.recover_pending_sessions()
            loaded = repository.read_session(patient["session_id"])
            self.assertEqual(loaded["followup_status"], "waiting_for_user")
            self.assertEqual((model.extractions, model.verifications), (0, 1))
            self.assertEqual(loaded["background_recovery_attempts"], 1)
            with connection.get_connection() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM bp_records").fetchone()[0], 1)

    def test_simulation_resume_verifies_saved_candidate_before_next_patient_turn(self):
        class PersistedFailure(ScriptedDemoModel):
            def __call__(self, **kwargs):
                payload = json.loads(kwargs["user_prompt"])
                if "patient_message" in payload and "claims" in payload:
                    raise TimeoutError("offline verification outage")
                return super().__call__(**kwargs)
        class Recovered(ScriptedDemoModel):
            def __init__(self):
                self.first = None
            def __call__(self, **kwargs):
                payload = json.loads(kwargs["user_prompt"])
                if self.first is None:
                    self.first = deepcopy(payload)
                return super().__call__(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(directory)
            run = manager.create("stable", model_mode="scripted", background=False, delay_seconds=0)
            paused = manager.execute(run["run_id"], model_override=PersistedFailure())
            self.assertEqual(paused["followup_status"], "interrupted")
            model = Recovered()
            completed = manager.execute(run["run_id"], model_override=model)
            self.assertIn("claims", model.first)
            self.assertIn("patient_message", model.first)
            self.assertEqual(completed["followup_status"], "completed")
            with connection.using_database(manager.store.database(run["run_id"])):
                loaded = repository.read_session(completed["session_id"])
            self.assertIn("EXTRACTION_RESUMED", [row["event"] for row in loaded["trace"]])

    def test_stable_dual_agent_completes_after_one_verification_timeout(self):
        class OneTimeout(ScriptedDemoModel):
            count = 0
            def __call__(self, **kwargs):
                payload = json.loads(kwargs["user_prompt"])
                if "patient_message" in payload and "claims" in payload:
                    self.count += 1
                    if self.count == 1:
                        raise TimeoutError("offline one-shot timeout")
                return super().__call__(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            manager = SimulationManager(directory)
            run = manager.create("stable", model_mode="scripted", background=False, delay_seconds=0)
            finished = manager.execute(run["run_id"], model_override=OneTimeout())
            self.assertEqual(finished["followup_status"], "completed")
            self.assertTrue(finished["evaluation"]["pass"])
            with connection.using_database(manager.store.database(run["run_id"])):
                loaded = repository.read_session(finished["session_id"])
            self.assertIn("MODEL_RETRY", [row["event"] for row in loaded["trace"]])
            self.assertNotIn("extraction_pending", loaded)


if __name__ == "__main__":
    unittest.main()

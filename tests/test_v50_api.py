"""Actual V5 API flows, isolated SQLite and explicit offline scripted models.

The fixture never reads .env or accesses a real patient database or model network.
"""

import contextlib
import copy
import importlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from database import collaboration, connection, repository, tasks


class V50APIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("dotenv.load_dotenv", return_value=False), patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1", "LLM_AGENT_V50": "0"
        }), contextlib.redirect_stdout(io.StringIO()):
            cls.backend = importlib.import_module("app")
            cls.v5_api = importlib.import_module("backend.v5_api")
            cls.guidance = importlib.import_module("medical_tools.guidance")
            cls.model_client = importlib.import_module("backend.model_client")
            cls.simulation = importlib.import_module("backend.simulation_v5")
            cls.runtime = importlib.import_module("backend.runtime_v5")
            cls.scripted = importlib.import_module("simulation.scripted")

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1",
            "LLM_AGENT_V50": "0", "LLM_AGENT_V42": "0", "LLM_AGENT_V4": "0"
        }))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.production_path = directory / "isolated-production.sqlite"
        self.stack.enter_context(patch.object(connection, "DB_PATH", self.production_path))
        self.stack.enter_context(patch.object(connection, "DATA_DIR", directory))
        self.stack.enter_context(connection.using_database(self.production_path))
        self.stack.enter_context(patch.object(self.backend, "DEMO_PATIENT", {
            "patient_id": "v50-api-p", "patient_name": "虚构API患者", "age": 72, "conditions": ["高血压"]
        }))
        self.stack.enter_context(patch.object(self.backend, "DEMO_BP_HISTORY", []))
        self.network = self.stack.enter_context(patch.object(self.model_client, "call_json_model",
                    side_effect=AssertionError("Offline API fixture forbids network model requests")))
        self.stack.enter_context(patch.object(self.guidance, "retrieve_medical_guidance", return_value=[{
            "block_id": "api-test", "source_id": "fictional-reference", "source_title": "测试医学依据",
            "text": "本条仅为测试证据。家庭血压记录应提交医生核对，不可自行调整原处方。"
        }]))
        self.stack.enter_context(patch.object(self.guidance, "knowledge_revision", return_value="isolated-api-test"))
        self.manager = self.simulation.SimulationManager(directory / "runs")
        self.stack.enter_context(patch.object(self.v5_api, "_manager", self.manager))
        self.http = self.stack.enter_context(TestClient(self.backend.app))
        self.addCleanup(self.drain_background)
        self.tokens = {}
        for identifier, role in (("api-doctor", "doctor"), ("api-family", "family")):
            collaboration.provision_user(identifier, role)
            collaboration.link_patient(identifier, "v50-api-p")
            self.tokens[role] = collaboration.issue_token(identifier)

    def drain_background(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with self.manager._guard:
                simulation_jobs = bool(self.manager._jobs)
            with self.v5_api._job_lock:
                api_jobs = bool(self.v5_api._jobs)
            if not simulation_jobs and not api_jobs:
                return
            time.sleep(0.01)
        self.fail("An offline background API operation did not stop within its budget")

    def start(self, scenario="stable", **options):
        response = self.http.post("/api/v5/simulations", json={
            "scenario_id": scenario, "model_mode": "scripted", "delay_seconds": 0, **options
        })
        self.assertEqual(response.status_code, 202, response.text)
        result = response.json()
        self.assertEqual(result["options"]["model_mode"], "scripted")
        self.assertTrue(result["is_simulation"])
        self.assertTrue({"run_id", "run_token", "doctor_token", "family_token"} <= result.keys())
        return result

    def poll(self, run, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.http.get("/api/v5/simulations/" + run["run_id"],
                                     headers={"X-Demo-Token": run["run_token"]})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            if result["status"] in {"completed", "failed", "budget_exhausted"}:
                return result
            time.sleep(0.01)
        self.fail("Scripted simulation did not reach a persisted terminal state")

    @staticmethod
    def run_headers(run, role="doctor"):
        return {"Authorization": "Bearer " + run[role + "_token"], "X-Simulation-Run-ID": run["run_id"]}

    def production_headers(self, role="doctor"):
        return {"Authorization": "Bearer " + self.tokens[role]}

    def production_count(self, table):
        with connection.get_connection(self.production_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_demo_page_and_scenario_metadata_are_available(self):
        page = self.http.get("/v5-demo")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("V5.0", page.text)
        self.assertIn("离线脚本演示", page.text)
        self.assertIn("未配置", page.text)
        response = self.http.get("/api/v5/scenarios")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["is_simulation"])
        for scenario in response.json()["items"]:
            self.assertNotIn("hidden_facts", scenario)
            self.assertNotIn("patient", scenario)
        self.assertEqual(self.http.post("/api/v5/simulations", json={
            "scenario_id": "stable", "model_mode": "scripted", "hidden_facts": {}
        }).status_code, 422)

    def test_scripted_start_poll_report_and_station_delivery_complete(self):
        run = self.start()
        result = self.poll(run)
        self.assertEqual(result["status"], "completed", json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["followup_status"], "completed")
        self.assertEqual(result["real_model_requests"], 0)
        self.assertTrue(result["evaluation"])
        self.assertTrue(any(turn["role"] == "patient" for turn in result["transcript"]))
        self.assertEqual(result["task"]["status"], "completed")
        self.assertTrue(result["task"]["report_generated"])
        self.assertEqual(result["task"]["station_messages"], 2)
        self.assertEqual(result["task"]["station_delivery_status"], "delivered")
        self.assertEqual(result["task"]["delivery_status"], "not_configured")
        stages = {entry["stage"] for entry in result["trace"]}
        self.assertTrue({"trigger", "planner", "information_extraction", "report", "notification"} <= stages)
        for table in ("followup_sessions", "reports", "report_messages", "bp_records", "proactive_tasks"):
            self.assertEqual(self.production_count(table), 0, table)
        self.network.assert_not_called()

    def test_doctor_handling_family_read_and_unauthenticated_denials(self):
        run = self.start()
        result = self.poll(run)
        self.assertEqual(result["status"], "completed")
        doctor, family = self.run_headers(run), self.run_headers(run, "family")
        reports = self.http.get("/api/reports", headers=doctor)
        self.assertEqual(reports.status_code, 200)
        row = reports.json()["items"][0]
        report_id = row["report_id"]
        self.assertTrue(row["doctor_view"])
        handling = self.http.patch(f"/api/reports/{report_id}/handling", headers=doctor,
                                  json={"status": "followup_required", "notes": "虚构报告已核对，安排复测。"})
        self.assertEqual(handling.status_code, 200, handling.text)
        self.assertEqual(handling.json()["status"], "followup_required")
        self.assertEqual(handling.json()["updated_by"], "simulation-doctor")
        self.assertEqual(self.http.get(f"/api/reports/{report_id}", headers=doctor).json()["notes"],
                         "虚构报告已核对，安排复测。")
        family_report = self.http.get(f"/api/reports/{report_id}", headers=family)
        self.assertEqual(family_report.status_code, 200)
        self.assertTrue(family_report.json()["family_view"])
        for forbidden in ("doctor_view", "notes", "medication_detail"):
            self.assertNotIn(forbidden, json.dumps(family_report.json()))
        family_messages = self.http.get("/api/messages", headers=family).json()["items"]
        self.assertEqual(len(family_messages), 1)
        self.assertEqual(family_messages[0]["recipient_role"], "family")
        read = self.http.post(f"/api/messages/{family_messages[0]['message_id']}/read", headers=family)
        self.assertEqual(read.status_code, 200)
        self.assertEqual(read.json()["status"], "read")
        self.assertEqual(self.http.get("/api/messages?unread_only=true", headers=family).json()["items"], [])
        self.assertEqual(self.http.patch(f"/api/reports/{report_id}/handling", headers=family,
                                       json={"status": "closed"}).status_code, 403)
        self.assertEqual(self.http.post(f"/api/reports/{report_id}/push", headers=family).status_code, 403)
        sid = result["session_id"]
        self.assertEqual(self.http.get(f"/api/v5/sessions/{sid}", headers=doctor).status_code, 200)
        self.assertEqual(self.http.get(f"/api/v5/sessions/{sid}/trace", headers=family).status_code, 403)
        self.assertEqual(self.http.post(f"/api/v5/sessions/{sid}/messages", headers=family,
                                      json={"message": "继续"}).status_code, 403)
        self.assertEqual(self.http.post(f"/api/v5/sessions/{sid}/messages", headers=doctor,
                                      json={"message": "继续"}).status_code, 409)
        for url in ("/api/reports", "/api/messages", f"/api/v5/sessions/{sid}"):
            self.assertEqual(self.http.get(url).status_code, 401)
        self.assertEqual(self.http.get("/api/v5/simulations/" + run["run_id"]).status_code, 403)

    def test_run_tokens_and_bearer_credentials_cannot_cross_databases(self):
        first, second = self.start(), self.start()
        one, two = self.poll(first), self.poll(second)
        self.assertEqual(one["status"], "completed")
        self.assertEqual(two["status"], "completed")
        bad_headers = {"Authorization": "Bearer " + first["doctor_token"],
                       "X-Simulation-Run-ID": second["run_id"]}
        self.assertEqual(self.http.get("/api/reports", headers=bad_headers).status_code, 403)
        self.assertEqual(self.http.get("/api/reports", headers={
            "Authorization": "Bearer " + first["doctor_token"]}).status_code, 401)
        self.assertEqual(self.http.get("/api/v5/simulations/" + first["run_id"], headers={
            "X-Demo-Token": second["run_token"]}).status_code, 403)
        self.assertEqual(self.http.get("/api/reports", headers={
            **self.run_headers(first), "X-Simulation-Run-ID": str(uuid.uuid4())}).status_code, 404)
        self.assertEqual(self.http.get("/api/v5/simulations/" + first["run_id"], headers={
            "X-Demo-Token": "wrong"}).status_code, 403)
        comparison = self.http.post("/api/v5/evaluations/compare", json={"runs": [
            {"run_id": first["run_id"], "run_token": first["run_token"]},
            {"run_id": second["run_id"], "run_token": second["run_token"]}
        ]})
        self.assertEqual(comparison.status_code, 200)
        self.assertEqual(len(comparison.json()["items"]), 2)
        self.assertNotIn("transcript", comparison.text)
        self.assertNotIn("doctor_token", comparison.text)

    def test_trace_excludes_raw_prompts_tokens_and_private_reasoning(self):
        run = self.start()
        result = self.poll(run)
        self.assertEqual(result["status"], "completed")
        sentinel = "DO_NOT_EXPOSE_RAW_SECRET"
        with connection.using_database(self.manager.store.database(run["run_id"])):
            state = copy.deepcopy(repository.read_session(result["session_id"]))
            state["trace"].append({"event": "LLM_CALL", "detail": {
                "purpose": "PLAN", "status": "success", "ok": True,
                "system_prompt": sentinel, "user_prompt": sentinel, "reasoning": sentinel,
                "api_key": sentinel, "token": sentinel, "raw_response": sentinel
            }})
            repository.persist_session(result["session_id"], state)
        response = self.http.get(f"/api/v5/sessions/{result['session_id']}/trace", headers=self.run_headers(run))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["items"])
        self.assertNotIn(sentinel, response.text)
        for entry in response.json()["items"]:
            self.assertNotIn("message", entry["detail"])
        simulation_view = self.http.get("/api/v5/simulations/" + run["run_id"],
                                        headers={"X-Demo-Token": run["run_token"]})
        self.assertEqual(simulation_view.status_code, 200)
        self.assertNotIn(sentinel, simulation_view.text)
        for credential in (run["doctor_token"], run["family_token"], run["run_token"]):
            self.assertNotIn(credential, simulation_view.text)

    def test_dangerous_scripted_patient_has_actual_escalated_report(self):
        run = self.start("danger")
        result = self.poll(run)
        self.assertEqual(result["status"], "completed", json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["followup_status"], "escalated")
        self.assertEqual(result["task"]["status"], "completed")
        reports = self.http.get("/api/reports", headers=self.run_headers(run)).json()["items"]
        self.assertEqual(reports[0]["report"]["completion_outcome"], "escalated")
        self.assertEqual(reports[0]["report"]["ai_summary"]["status"], "not_requested")
        self.assertTrue(any(event["stage"] == "safety" for event in result["trace"]))
        self.assertEqual(result["real_model_requests"], 0)
        self.network.assert_not_called()

    def test_manual_tick_only_mutates_current_doctors_authorized_patients(self):
        conn = connection.get_connection()
        try:
            with conn:
                conn.execute("INSERT INTO patients(patient_id,patient_name) VALUES('ungranted','虚构未授权患者')")
        finally:
            conn.close()
        now = datetime.now(timezone.utc) - timedelta(seconds=1)
        denied = tasks.create_task("ungranted", "unauthorized", "scheduled", due_at=now - timedelta(days=1))
        denied_plan = tasks.create_plan("ungranted", "scheduled", now, plan_id="ungranted-plan")
        allowed = tasks.create_task("v50-api-p", "authorized", "scheduled", due_at=now)
        scripted_runtime = self.runtime.AgentRuntime(self.scripted.ScriptedDemoModel())
        with patch.object(self.v5_api, "AgentRuntime", return_value=scripted_runtime):
            response = self.http.post("/api/v5/engine/tick", headers=self.production_headers())
            self.assertEqual(response.status_code, 202)
            self.drain_background()
        self.assertEqual(tasks.get_task(allowed["task_id"])["status"], "waiting_for_user")
        self.assertEqual(tasks.get_task(denied["task_id"]), denied)
        self.assertEqual(tasks.get_plan(denied_plan["plan_id"]), denied_plan)
        self.assertEqual(self.http.post("/api/v5/engine/tick", headers=self.production_headers("family")).status_code, 403)
        self.assertEqual(self.http.post("/api/v5/engine/tick").status_code, 401)
        self.network.assert_not_called()

    def test_plans_and_bp_events_use_authorized_persisted_evidence(self):
        doctor, family = self.production_headers(), self.production_headers("family")
        due = datetime.now(timezone.utc) + timedelta(hours=1)
        plan = self.http.post("/api/v5/plans", headers=doctor, json={
            "patient_id": "v50-api-p", "plan_type": "missed_measurement", "due_at": due.isoformat(),
            "interval_seconds": 3600, "window_seconds": 3600
        })
        self.assertEqual(plan.status_code, 200, plan.text)
        plan_id = plan.json()["plan_id"]
        self.assertEqual(self.http.get("/api/v5/plans?patient_id=v50-api-p", headers=doctor).json()["items"][0]["plan_id"], plan_id)
        updated = self.http.patch("/api/v5/plans/" + plan_id, headers=doctor, json={"enabled": False})
        self.assertEqual(updated.status_code, 200)
        self.assertFalse(updated.json()["enabled"])
        self.assertEqual(self.http.patch("/api/v5/plans/" + plan_id, headers=family,
                                        json={"enabled": True}).status_code, 403)
        event_payload = {"patient_id": "v50-api-p", "event_id": "real-api-reading",
                         "systolic": 168, "diastolic": 104, "measured_at": datetime.now(timezone.utc).isoformat()}
        event = self.http.post("/api/v5/bp-events", headers=doctor, json=event_payload)
        self.assertEqual(event.status_code, 200, event.text)
        actual = event.json()
        self.assertEqual(actual["task"]["measurement"]["id"], actual["measurement"]["id"])
        self.assertEqual(actual["task"]["status"], "queued")
        self.assertEqual(actual["task"]["delivery_status"], "not_configured")
        self.assertEqual(actual["task"]["station_messages"], 0)
        repeated = self.http.post("/api/v5/bp-events", headers=doctor, json=event_payload)
        self.assertEqual(repeated.status_code, 200)
        self.assertTrue(repeated.json()["duplicate"])
        self.assertEqual(repeated.json()["task"]["task_id"], actual["task"]["task_id"])
        conflicting = self.http.post("/api/v5/bp-events", headers=doctor,
                                     json={**event_payload, "systolic": 170})
        self.assertEqual(conflicting.status_code, 422)
        invalid = self.http.post("/api/v5/bp-events", headers=doctor,
                                json={**event_payload, "event_id": "invalid", "systolic": 40})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.http.post("/api/v5/bp-events", headers=family, json=event_payload).status_code, 403)
        self.assertEqual(self.production_count("bp_records"), 1)
        self.assertEqual(self.production_count("proactive_tasks"), 1)

    def test_missed_measurement_simulation_uses_real_empty_plan_window(self):
        run = self.start("stable", trigger_type="missed_measurement")
        result = self.poll(run)
        self.assertEqual(result["status"], "completed", json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["task"]["trigger_type"], "missed_measurement")
        self.assertEqual(result["task"]["payload"]["measurement_count"], 0)
        self.assertTrue(result["task"]["payload"]["plan_id"])
        self.assertEqual(tasks.utc_datetime(result["task"]["payload"]["window_end"]) -
                         tasks.utc_datetime(result["task"]["payload"]["window_start"]), timedelta(seconds=60))
        report = self.http.get("/api/reports", headers=self.run_headers(run)).json()["items"][0]["report"]
        self.assertIsNone(report["blood_pressure"]["systolic"])
        self.assertIsNone(report["blood_pressure"]["diastolic"])
        self.assertEqual(result["real_model_requests"], 0)

    def test_busy_session_rejects_second_input_instead_of_silently_dropping_it(self):
        """The asynchronous API acknowledges only the message it actually dispatches."""
        from agent.state_manager import create_patient_state

        sid = "v50-concurrent-api"
        state = create_patient_state("scheduled", DEMO_PATIENT=self.backend.DEMO_PATIENT,
                                     DEMO_BP_HISTORY=[], TRIGGER_CONFIGS=self.backend.TRIGGER_CONFIGS)
        state.update(agent_version="V5.0", session_id=sid, followup_status="waiting_for_user")
        repository.persist_session(sid, state)
        entered, release = threading.Event(), threading.Event()
        received = []
        worker_path = []

        def blocked_turn(session_id, message):
            worker_path.append(connection.current_database_path())
            received.append((session_id, message))
            entered.set()
            if not release.wait(3):
                raise AssertionError("The API worker was not released")

        runtime = Mock()
        runtime.turn.side_effect = blocked_turn
        url = f"/api/v5/sessions/{sid}/messages"
        try:
            with patch.object(self.v5_api, "AgentRuntime", return_value=runtime):
                first = self.http.post(url, headers=self.production_headers(), json={"message": "第一条患者回答"})
                self.assertEqual(first.status_code, 202, first.text)
                self.assertTrue(entered.wait(1), "The accepted message was not executed")
                second = self.http.post(url, headers=self.production_headers(), json={"message": "第二条患者回答"})
                self.assertEqual(second.status_code, 409, second.text)
                self.assertEqual(received, [(sid, "第一条患者回答")])
                self.assertEqual(worker_path, [self.production_path.resolve()])
                release.set()
                self.drain_background()
                # The explicitly rejected message can be submitted after the busy operation ends.
                retry = self.http.post(url, headers=self.production_headers(), json={"message": "第二条患者回答"})
                self.assertEqual(retry.status_code, 202, retry.text)
                self.drain_background()
                self.assertEqual(received, [(sid, "第一条患者回答"), (sid, "第二条患者回答")])
        finally:
            release.set()

    def test_dispatch_busy_keys_are_scoped_to_database_and_release_after_completion(self):
        entered, release, other_entered = threading.Event(), threading.Event(), threading.Event()

        def blocked():
            entered.set()
            if not release.wait(3):
                raise AssertionError("The dispatch worker was not released")

        try:
            self.assertTrue(self.v5_api.dispatch(blocked, "same-session"))
            self.assertTrue(entered.wait(1))
            self.assertFalse(self.v5_api.dispatch(lambda: None, "same-session"))
            with connection.using_database(self.production_path.parent / "other-scope.sqlite"):
                self.assertTrue(self.v5_api.dispatch(other_entered.set, "same-session"))
            self.assertTrue(other_entered.wait(1), "A different isolated database was blocked")
        finally:
            release.set()
            self.drain_background()
        self.assertTrue(self.v5_api.dispatch(lambda: None, "same-session"))
        self.drain_background()

    def test_compatible_chat_endpoint_does_not_globally_block_other_sessions(self):
        from agent.state_manager import create_patient_state

        for sid in ("legacy-slow", "legacy-independent"):
            state = create_patient_state("scheduled", DEMO_PATIENT=self.backend.DEMO_PATIENT,
                                         DEMO_BP_HISTORY=[], TRIGGER_CONFIGS=self.backend.TRIGGER_CONFIGS)
            state.update(agent_version="V5.0", session_id=sid, followup_status="waiting_for_user")
            repository.persist_session(sid, state)
        slow_entered, release, independent_done = threading.Event(), threading.Event(), threading.Event()
        results, failures, workers = {}, [], []

        def planner(state):
            if state["session_id"] == "legacy-slow":
                slow_entered.set()
                if not release.wait(3):
                    raise AssertionError("The slow chat request was not released")
            state["followup_status"] = "waiting_for_user"
            return {"message": "请回答当前问题。", "state": state, "trace": state["trace"]}

        def request(sid):
            try:
                with connection.using_database(self.production_path):
                    results[sid] = self.http.post("/api/chat", json={"session_id": sid, "message": "患者回答"})
                if sid == "legacy-independent":
                    independent_done.set()
            except BaseException as error:
                failures.append(error)

        try:
            with patch.object(self.backend, "run_agent", side_effect=planner), \
                    patch.object(self.backend, "extract_user_patch", return_value={"topic_status": {}}), \
                    patch.object(self.backend, "_refresh_memory_safely", return_value={}):
                for sid in ("legacy-slow", "legacy-independent"):
                    worker = threading.Thread(target=request, args=(sid,), daemon=True)
                    workers.append(worker)
                    worker.start()
                    if sid == "legacy-slow":
                        self.assertTrue(slow_entered.wait(1))
                self.assertTrue(independent_done.wait(1), "The existing chat route still held a global session lock")
                release.set()
                for worker in workers:
                    worker.join(timeout=3)
                    self.assertFalse(worker.is_alive())
                self.assertEqual(failures, [])
                for response in results.values():
                    self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(results), 2)
        finally:
            release.set()
            for worker in workers:
                worker.join(timeout=3)

    def test_accepted_message_background_failure_is_durably_visible_to_doctor(self):
        from agent.state_manager import create_patient_state

        sid = "background-failure"
        state = create_patient_state("scheduled", DEMO_PATIENT=self.backend.DEMO_PATIENT,
                                     DEMO_BP_HISTORY=[], TRIGGER_CONFIGS=self.backend.TRIGGER_CONFIGS)
        state.update(agent_version="V5.0", session_id=sid, followup_status="waiting_for_user")
        repository.persist_session(sid, state)
        runtime = Mock()
        runtime.turn.side_effect = RuntimeError("PRIVATE_FAILURE_DETAIL")
        with patch.object(self.v5_api, "AgentRuntime", return_value=runtime):
            accepted = self.http.post(f"/api/v5/sessions/{sid}/messages", headers=self.production_headers(),
                                      json={"message": "真实接受的虚构患者回答"})
            self.assertEqual(accepted.status_code, 202, accepted.text)
            self.drain_background()
        response = self.http.get(f"/api/v5/sessions/{sid}", headers=self.production_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["followup_status"], "interrupted")
        self.assertIn("处理未完成", response.json()["message"])
        self.assertNotIn("PRIVATE_FAILURE_DETAIL", response.text)
        # Discard the process cache to demonstrate that the failure survived in SQLite.
        repository.sessions.pop(sid, None)
        stored = repository.read_session(sid)
        self.assertEqual(stored["interruption_reason"], "background_execution_failed")
        failure = [entry for entry in stored["trace"] if entry["event"] == "TASK_INTERRUPTED"]
        self.assertEqual(len(failure), 1)
        self.assertEqual(failure[0]["detail"]["error_type"], "RuntimeError")
        self.assertIsNone(stored.get("report"))

    def test_restart_recovers_processing_without_fabricating_another_patient_message(self):
        from agent.state_manager import add_trace, create_patient_state

        sid = "restart-processing"
        state = create_patient_state("scheduled", DEMO_PATIENT=self.backend.DEMO_PATIENT,
                                     DEMO_BP_HISTORY=[], TRIGGER_CONFIGS=self.backend.TRIGGER_CONFIGS)
        state.update(agent_version="V5.0", session_id=sid, followup_status="processing")
        add_trace(state, "USER_MESSAGE", {"message": "崩溃前已保存的虚构患者回答"})
        repository.persist_session(sid, state)
        repository.sessions.pop(sid, None)
        runtime = self.runtime.AgentRuntime(self.scripted.ScriptedDemoModel())
        with patch.object(self.v5_api, "AgentRuntime", return_value=runtime), \
                patch.object(self.runtime, "extract_user_patch_v4") as extraction:
            self.v5_api.recover_pending_sessions()
            self.drain_background()
            self.v5_api.recover_pending_sessions()
            self.drain_background()
        recovered = repository.read_session(sid)
        self.assertEqual(recovered["followup_status"], "waiting_for_user")
        self.assertEqual(recovered["background_recovery_attempts"], 1)
        messages = [entry["detail"]["message"] for entry in recovered["trace"] if entry["event"] == "USER_MESSAGE"]
        self.assertEqual(messages, ["崩溃前已保存的虚构患者回答"])
        self.assertTrue(recovered["last_agent_response"])
        extraction.assert_not_called()
        self.network.assert_not_called()

    def test_restart_recovers_terminal_report_once_and_preserves_message_read_state(self):
        run = self.start()
        result = self.poll(run)
        self.assertEqual(result["status"], "completed")
        path = self.manager.store.database(run["run_id"])
        sid = "restart-terminal-report"
        runtime = self.runtime.AgentRuntime(self.scripted.ScriptedDemoModel())
        with connection.using_database(path):
            # Reuse the completed fictional facts to seed the exact crash boundary:
            # a terminal session was saved before its report or messages were written.
            state = copy.deepcopy(repository.read_session(result["session_id"]))
            state.update(session_id=sid, report=None, report_generation={"status": "pending"})
            state.pop("proactive_task_id", None)
            state.pop("background_recovery_attempts", None)
            state["trace"] = [entry for entry in state["trace"]
                              if entry["event"] not in {"REPORT_GENERATED", "MESSAGE_DELIVERY"}]
            repository.persist_session(sid, state)
            repository.sessions.pop(sid, None)
            with patch.object(self.v5_api, "AgentRuntime", return_value=runtime):
                self.v5_api.recover_pending_sessions()
                self.drain_background()
                recovered = repository.read_session(sid)
                self.assertEqual(recovered["report_generation"]["status"], "generated")
                self.assertTrue(recovered["report"])
                report_id = None
                with connection.get_connection() as conn:
                    reports = conn.execute("SELECT id FROM reports WHERE session_id=?", (sid,)).fetchall()
                    self.assertEqual(len(reports), 1)
                    report_id = reports[0][0]
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM report_messages WHERE report_id=?",
                                                  (report_id,)).fetchone()[0], 2)
                family = collaboration.authenticate_token(run["family_token"])
                message = next(item for item in collaboration.list_messages(family) if item["report_id"] == report_id)
                read = collaboration.mark_read(family, message["id"])
                for report_status in ("summary_pending", "failed"):
                    recovered = copy.deepcopy(repository.read_session(sid))
                    recovered["report_generation"] = {"status": report_status}
                    repository.persist_session(sid, recovered)
                    self.v5_api.recover_pending_sessions()
                    self.drain_background()
                # A fourth scan sees generated work and does not create another retry.
                self.v5_api.recover_pending_sessions()
                self.drain_background()
                self.assertEqual(repository.read_session(sid)["background_recovery_attempts"], 3)
                with connection.get_connection() as conn:
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM reports WHERE session_id=?", (sid,)).fetchone()[0], 1)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM report_messages WHERE report_id=?",
                                                  (report_id,)).fetchone()[0], 2)
                current_message = next(item for item in collaboration.list_messages(family) if item["report_id"] == report_id)
                self.assertEqual(current_message["status"], "read")
                self.assertEqual(current_message["read_at"], read["read_at"])
        self.network.assert_not_called()

    def test_recovery_attempt_cap_is_rechecked_for_stale_concurrent_scans(self):
        from agent.state_manager import create_patient_state

        sid = "stale-recovery"
        state = create_patient_state("scheduled", DEMO_PATIENT=self.backend.DEMO_PATIENT,
                                     DEMO_BP_HISTORY=[], TRIGGER_CONFIGS=self.backend.TRIGGER_CONFIGS)
        state.update(agent_version="V5.0", session_id=sid, followup_status="partially_completed",
                     completion_outcome="partial", report=None, report_generation={"status": "failed"},
                     background_recovery_attempts=2)
        repository.persist_session(sid, state)
        callbacks = []

        def postpone(function, key):
            callbacks.append(function)
            return True

        with patch.object(self.v5_api, "dispatch", side_effect=postpone):
            # Both workers read attempt 2 before either callback executes.
            self.v5_api.recover_pending_sessions()
            self.v5_api.recover_pending_sessions()
        self.assertEqual(len(callbacks), 2)
        runtime = Mock()
        runtime.finalize.side_effect = RuntimeError("offline report generation failure")
        with patch.object(self.v5_api, "AgentRuntime", return_value=runtime):
            with self.assertRaises(RuntimeError):
                callbacks[0]()
            callbacks[1]()
        self.assertEqual(repository.read_session(sid)["background_recovery_attempts"], 3)
        runtime.finalize.assert_called_once_with(sid)


if __name__ == "__main__":
    unittest.main()

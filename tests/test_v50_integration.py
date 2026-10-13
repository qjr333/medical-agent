"""Real V5 task/Agent/report/collaboration chains using an offline model.

Each fictional run owns a disposable SQLite database. No API credentials,
existing patient database, external embedding cache or real DeepSeek is used.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.runtime_v5 import AgentRuntime
from backend.simulation_v5 import SimulationManager
from database import collaboration, connection, repository, tasks
from medical_tools import guidance
from simulation.patient import get_scenario


EVIDENCE = [{"block_id": "integration-fixture", "source_id": "integration-source",
             "source_title": "模拟家庭监测及用药管理依据", "year": 2026, "source_pages": "1",
             "text": "记录真实家庭血压和服药情况。具体药物及漏服处理应核对原医嘱，并咨询医生或药师，不自行调整处方。"}]


class V50IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.production_path = self.root / "isolated-production-control.db"
        self.stack.enter_context(patch.object(connection, "DB_PATH", self.production_path))
        self.stack.enter_context(patch.object(connection, "DATA_DIR", self.root))
        self.stack.enter_context(connection.using_database(self.production_path))
        connection.init_db()
        self.stack.enter_context(patch.object(guidance, "knowledge_revision", return_value="v50-integration-fixture"))
        self.retrieval = self.stack.enter_context(patch.object(guidance, "retrieve_medical_guidance", return_value=EVIDENCE))
        self.manager = SimulationManager(self.root / "simulations")

    def query(self, sql, parameters=()):
        conn = connection.get_connection()
        try:
            return [dict(row) for row in conn.execute(sql, parameters).fetchall()]
        finally:
            conn.close()

    def run_scenario(self, scenario_id, **options):
        created = self.manager.create(scenario_id, model_mode="scripted", delay_seconds=0,
                                      max_turns=20, background=False, **options)
        result = self.manager.execute(created["run_id"])
        self.assertEqual(result["status"], "completed", json.dumps(result, ensure_ascii=False, default=str))
        scenario = get_scenario(scenario_id)
        self.assertIn(result["followup_status"], {"completed": "completed", "partial": "partially_completed",
                                                 "escalated": "escalated"}.values())
        with connection.using_database(self.manager.store.database(created["run_id"])):
            state = repository.read_session(result["session_id"])
            self.assertIn(state["completion_outcome"], scenario["expected_outcomes"])
            self.assertTrue(state["report"])
            self.assertEqual(len(self.query("SELECT id FROM reports WHERE session_id=?", (result["session_id"],))), 1)
            self.assertEqual(len(self.query("SELECT id FROM report_messages")), 2)
            self.assertEqual(tasks.get_task(result["task_id"])["status"], "completed")
        view = self.manager.view(created["run_id"], created["run_token"])
        self.assertEqual(view["real_model_requests"], 0)
        self.assertFalse(self.query("SELECT id FROM bp_records"))
        self.assertFalse(self.query("SELECT session_id FROM followup_sessions"))
        return created, result, state

    def test_stable_scenario_collects_current_facts_and_delivers_report(self):
        created, result, state = self.run_scenario("stable")
        facts = get_scenario("stable")["hidden_facts"]
        self.assertEqual(state["completion_outcome"], "completed")
        self.assertEqual((state["bp_systolic"], state["bp_diastolic"]), (facts["bp_systolic"], facts["bp_diastolic"]))
        self.assertEqual(state["medication_status"], "taken")
        self.assertTrue(result["evaluation"]["pass"], result["evaluation"])

    def test_high_bp_scenario_uses_the_real_trigger_measurement_and_measurement_context(self):
        created, result, state = self.run_scenario("high_bp")
        self.assertEqual(state["trigger_type"], "high_bp")
        self.assertEqual((state["bp_systolic"], state["bp_diastolic"]), (168, 104))
        self.assertTrue(state["measurement_context"])
        self.assertEqual(state["completion_outcome"], "completed")
        self.assertTrue(result["evaluation"]["pass"], result["evaluation"])

    def test_nonadherent_scenario_uses_registered_medical_tools_without_dose_changes(self):
        created, result, state = self.run_scenario("nonadherent")
        self.assertEqual(state["medication_status"], "missed")
        executed = {event["detail"]["tool"] for event in state["trace"] if event["event"] == "V4_TOOL_CALL"
                    and event["detail"]["status"] in {"success", "fallback"}}
        self.assertTrue({"RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE"} <= executed)
        self.assertTrue(result["evaluation"]["pass"], result["evaluation"])

    def test_unclear_scenario_is_partial_and_does_not_invent_todays_reading(self):
        created, result, state = self.run_scenario("unclear")
        self.assertEqual(state["completion_outcome"], "partial")
        self.assertIsNone(state["bp_systolic"])
        self.assertEqual(state["medication_status"], "unknown")
        self.assertTrue(state["report"]["unconfirmed_items"])
        self.assertTrue(result["evaluation"]["pass"], result["evaluation"])

    def test_danger_scenario_escalates_before_regular_planning_or_retrieval(self):
        created, result, state = self.run_scenario("danger")
        self.assertEqual(state["completion_outcome"], "escalated")
        self.assertEqual(state["risk_level"], "high")
        danger_index = next(index for index, event in enumerate(state["trace"]) if event["event"] == "USER_MESSAGE")
        subsequent = state["trace"][danger_index + 1:]
        self.assertFalse(any(event["event"] == "LLM_CALL" and event["detail"].get("purpose") in {"PLAN", "EXTRACT"}
                             and event["detail"].get("called") for event in subsequent))
        self.retrieval.assert_not_called()
        self.assertTrue(result["evaluation"]["pass"], result["evaluation"])

    def test_refusing_scenario_stops_with_unconfirmed_facts_and_partial_report(self):
        created, result, state = self.run_scenario("refusing")
        self.assertEqual(state["completion_outcome"], "partial")
        self.assertIsNone(state["bp_systolic"])
        self.assertEqual(state["medication_status"], "unknown")
        self.assertTrue(state["report"]["unconfirmed_items"])
        self.assertTrue(result["evaluation"]["pass"], result["evaluation"])

    def test_missing_measurement_objective_needs_a_real_patient_monitoring_plan(self):
        created, result, state = self.run_scenario("stable", trigger_type="missed_measurement")
        self.assertEqual(state["trigger_type"], "missed_measurement")
        self.assertIsNone(state["bp_systolic"])
        evidence = state["completion_evaluation"]["objective_evidence"]["monitoring_plan"]
        patient_messages = [entry["message"] for entry in result["transcript"] if entry["role"] == "patient"]
        self.assertTrue(any(evidence in message for message in patient_messages))

    def test_collaboration_role_authentication_and_repeated_delivery_are_idempotent(self):
        created, result, state = self.run_scenario("stable")
        with connection.using_database(self.manager.store.database(created["run_id"])):
            doctor = collaboration.authenticate_token(created["doctor_token"])
            family = collaboration.authenticate_token(created["family_token"])
            self.assertEqual(doctor["role"], "doctor")
            self.assertEqual(family["role"], "family")
            report = collaboration.list_reports(doctor)[0]
            self.assertIn("doctor_view", report)
            family_report = collaboration.get_report({**family, "role": "doctor"}, report["report_id"])
            self.assertIn("family_view", family_report)
            self.assertNotIn("doctor_view", family_report)
            with self.assertRaises(PermissionError):
                collaboration.update_report_status(family, report["report_id"], "reviewed", "private clinician note")
            collaboration.update_report_status(doctor, report["report_id"], "reviewed", "private clinician note")
            messages = collaboration.list_messages(family)
            read = collaboration.mark_read(family, messages[0]["id"])
            read_again = collaboration.mark_read(family, messages[0]["id"])
            self.assertEqual(read["read_at"], read_again["read_at"])
            self.assertNotIn("private clinician note", json.dumps(read_again, ensure_ascii=False))
            self.assertEqual(collaboration.repush_report(doctor, report["report_id"])["created_messages"], 0)
            AgentRuntime().finalize(result["session_id"])
            self.assertEqual(len(self.query("SELECT id FROM reports")), 1)
            self.assertEqual(len(self.query("SELECT id FROM report_messages")), 2)
            collaboration.link_patient(family["user_id"], state["patient_id"], authorized=False)
            self.assertEqual(collaboration.list_messages(family), [])
            with self.assertRaises(PermissionError):
                collaboration.get_report(family, report["report_id"])

    def test_simulation_access_token_and_scope_isolation(self):
        first, result, state = self.run_scenario("stable")
        second = self.manager.create("stable", model_mode="scripted", background=False, delay_seconds=0)
        with self.assertRaises(PermissionError):
            self.manager.view(first["run_id"], second["run_token"])
        with connection.using_database(self.manager.store.database(second["run_id"])):
            self.assertIsNone(repository.read_session(result["session_id"]))
            self.assertFalse(self.query("SELECT id FROM reports"))
        with connection.using_database(self.manager.store.database(first["run_id"])):
            self.assertEqual(repository.read_session(result["session_id"])["patient_id"], state["patient_id"])

    def test_scoped_session_mapping_keeps_same_id_isolated_and_legacy_operations_work(self):
        first, second = self.root / "scope-a.db", self.root / "scope-b.db"
        with connection.using_database(first):
            repository.sessions["same-id"] = {"patient_id": "a"}
            first_lock = repository.session_lock("same-id")
        with connection.using_database(second):
            self.assertNotIn("same-id", repository.sessions)
            repository.sessions["same-id"] = {"patient_id": "b"}
            self.assertIsNot(repository.session_lock("same-id"), first_lock)
            self.assertEqual(list(repository.sessions), ["same-id"])
            self.assertEqual(repository.sessions.pop("same-id")["patient_id"], "b")
            repository.sessions["same-id"] = {"patient_id": "b"}
            repository.sessions.clear()
            self.assertEqual(len(repository.sessions), 0)
        with connection.using_database(first):
            self.assertEqual(repository.sessions["same-id"]["patient_id"], "a")
            repository.sessions.clear()

    def http_client(self):
        from backend import collaboration_api, v5_api
        app = FastAPI()
        app.include_router(v5_api.router)
        app.include_router(collaboration_api.router)
        app.middleware("http")(v5_api.simulation_scope)
        self.stack.enter_context(patch.object(v5_api, "_manager", self.manager))
        return self.stack.enter_context(TestClient(app))

    def test_v5_demo_api_uses_scripted_mode_and_checks_run_tokens_and_request_schema(self):
        http = self.http_client()
        scenarios = http.get("/api/v5/scenarios")
        self.assertEqual(scenarios.status_code, 200)
        self.assertEqual(len(scenarios.json()["items"]), 6)
        self.assertNotIn("hidden_facts", json.dumps(scenarios.json()))
        with patch.object(self.manager, "launch"):
            response = http.post("/api/v5/simulations", json={
                "scenario_id": "stable", "model_mode": "scripted", "delay_seconds": 0,
            })
        self.assertEqual(response.status_code, 202, response.text)
        created = response.json()
        result = self.manager.execute(created["run_id"])
        self.assertEqual(result["status"], "completed")
        path = "/api/v5/simulations/" + created["run_id"]
        self.assertEqual(http.get(path).status_code, 403)
        self.assertEqual(http.get(path, headers={"X-Demo-Token": "wrong-token"}).status_code, 403)
        view = http.get(path, headers={"X-Demo-Token": created["run_token"]})
        self.assertEqual(view.status_code, 200)
        self.assertEqual(view.json()["real_model_requests"], 0)
        self.assertEqual(http.post(path + "/resume", headers={"X-Demo-Token": created["run_token"]}).status_code, 409)
        extra = http.post("/api/v5/simulations", json={"scenario_id": "stable", "model_mode": "scripted",
                                                       "database_path": "/not/a/client-selected/database"})
        self.assertEqual(extra.status_code, 422)
        bad_mode = http.post("/api/v5/simulations", json={"scenario_id": "stable", "model_mode": "pretend_live"})
        self.assertEqual(bad_mode.status_code, 422)

    def test_v5_http_collaboration_keeps_role_views_and_trace_access_separate(self):
        created, result, state = self.run_scenario("stable")
        http = self.http_client()
        doctor_headers = {"X-Simulation-Run-ID": created["run_id"], "Authorization": "Bearer " + created["doctor_token"]}
        family_headers = {"X-Simulation-Run-ID": created["run_id"], "Authorization": "Bearer " + created["family_token"]}
        doctor = http.get("/api/reports", headers=doctor_headers)
        family = http.get("/api/reports", headers=family_headers)
        self.assertEqual(doctor.status_code, 200, doctor.text)
        self.assertEqual(family.status_code, 200, family.text)
        report_id = doctor.json()["items"][0]["report_id"]
        self.assertIn("doctor_view", doctor.json()["items"][0])
        self.assertNotIn("doctor_view", family.json()["items"][0])
        trace_path = "/api/v5/sessions/" + result["session_id"] + "/trace"
        self.assertEqual(http.get(trace_path, headers=doctor_headers).status_code, 200)
        self.assertEqual(http.get(trace_path, headers=family_headers).status_code, 403)
        handling_path = "/api/reports/" + str(report_id) + "/handling"
        self.assertEqual(http.patch(handling_path, headers=family_headers, json={"status": "reviewed"}).status_code, 403)
        reviewed = http.patch(handling_path, headers=doctor_headers, json={"status": "reviewed", "notes": "医生内部核对"})
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        family = http.get("/api/reports/" + str(report_id), headers=family_headers)
        self.assertNotIn("医生内部核对", family.text)
        terminal_message = http.post("/api/v5/sessions/" + result["session_id"] + "/messages",
                                    headers=doctor_headers, json={"message": "重复患者输入"})
        self.assertEqual(terminal_message.status_code, 409)


if __name__ == "__main__":
    unittest.main()

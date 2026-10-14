"""Direct Agent and model-client regressions, independent of FastAPI routes."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "RAG_DISABLE_EMBEDDING": "1"}):
    from backend.config import load_demo_data
    from backend import model_client
    from agent import extractor, planner, question_planner, runner, state_manager
    from medical_tools.registry import get_tool_registry


ROOT = Path(__file__).resolve().parents[1]


def make_state(trigger="missed_measurement"):
    data = load_demo_data()
    return state_manager.create_patient_state(
        trigger, DEMO_PATIENT=data["patient"],
        DEMO_BP_HISTORY=data["bp_history"], TRIGGER_CONFIGS=data["trigger_configs"],
    )


class AgentModuleTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_direct_extractor_and_state_merge_preserve_danger_and_rule_overrides(self):
        state = make_state()
        message = "安静休息后180/120，剧烈胸痛，今天忘记吃药"
        patch_data = extractor.extract_user_patch(
            message, call_json_model=lambda **kwargs: {
                "bp_systolic": 120, "bp_diastolic": 70,
                "medication_status": "taken",
            },
        )
        state_manager.merge_user_patch(state, patch_data, message)
        self.assertEqual((state["bp_systolic"], state["bp_diastolic"]), (180, 120))
        self.assertEqual(state["medication_status"], "missed")
        self.assertIn("胸痛", state["red_flags"])
        self.assertTrue(state["coverage"]["measurement_context"])

    def test_selector_keeps_escalation_before_medication_questions(self):
        state = make_state()
        state.update(risk_level="high", latest_patient_question="能不能多吃一片药？")
        self.assertEqual(planner.select_next_action(state, []), "ESCALATE")
        self.assertIn("ESCALATE", planner.get_available_actions(state))
        state["risk_level"] = "normal"
        self.assertNotIn("ESCALATE", planner.get_available_actions(state))
        self.assertEqual(planner.select_next_action(state, []), "RETRIEVE_MEDICAL_GUIDANCE")
        state["last_guidance_query"] = state["latest_patient_question"]
        self.assertEqual(planner.select_next_action(state, []), "MEDICATION_GUIDANCE")

    def test_question_planner_updates_goal_and_uses_memory_without_filling_answers(self):
        state = make_state()
        state["patient_memory"] = {"blood_pressure": {"latest_recorded": {
            "systolic": 145, "diastolic": 85, "measured_at": "2026-10-10T08:00:00+00:00",
        }}}
        question = question_planner.generate_question(state)
        self.assertIn("145/85", question)
        self.assertEqual(state["current_goal"], "collect_bp")
        self.assertFalse(state["coverage"]["bp"])
        self.assertEqual(state["trace"][-1]["event"], "MEMORY_USED")

    def test_runner_uses_registry_and_bypasses_normal_planning_for_high_risk(self):
        state = make_state()
        state["red_flags"] = ["胸痛"]
        tools = {name: Mock(side_effect=function) for name, function in
                 get_tool_registry(call_json_model=lambda **kwargs: {}).items()}
        with patch.object(runner, "select_next_action", side_effect=AssertionError("unsafe planning")):
            response = runner.run_agent(state, tool_registry=tools)
        self.assertEqual(state["risk_level"], "high")
        self.assertEqual(state["followup_status"], "escalated")
        self.assertEqual(response["report"]["risk"]["level"], "high")
        tools["ASSESS_RISK"].assert_called_once_with(state)
        tools["CREATE_ALERT"].assert_called_once_with(state)
        for name in ("CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            tools[name].assert_not_called()

    def test_runner_trace_and_finished_report_use_existing_fields(self):
        state = make_state()
        state.update(bp_systolic=145, bp_diastolic=85, medication_status="taken",
                     symptom_status="denied", red_flag_status="denied")
        state["coverage"] = dict.fromkeys(state["coverage"], True)
        tools = {name: Mock(side_effect=function) for name, function in
                 get_tool_registry(call_json_model=lambda **kwargs: {}).items()}
        response = runner.run_agent(state, tool_registry=tools)
        self.assertEqual(state["followup_status"], "completed")
        self.assertEqual(set(response), {"message", "state", "report", "trace"})
        self.assertEqual(response["report"], state["report"])
        self.assertIn("FINISH", {item["event"] for item in state["trace"]})
        tools["CHECK_BP_HISTORY"].assert_called_once_with(state)
        self.assertEqual(tools["ASSESS_RISK"].call_count, 2)

    def test_packages_can_import_without_app_or_heavy_embedding_import(self):
        code = '''
import importlib, sys
for name in (
    "agent.extractor", "agent.state_manager", "agent.planner",
    "agent.question_planner", "agent.runner", "medical_tools.registry",
    "memory.patient_memory", "database.repository", "backend.model_client",
):
    importlib.import_module(name)
assert "app" not in sys.modules
assert "sentence_transformers" not in sys.modules
assert "torch" not in sys.modules
'''
        environment = dict(os.environ, DEEPSEEK_API_KEY="", RAG_DISABLE_EMBEDDING="1")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class ModelClientTests(unittest.TestCase):
    def response(self, content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def fake_client(self, create):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_json_mode_arguments_and_parsing_with_mock_client(self):
        create = Mock(return_value=self.response('```json\n{"answer": "测试"}\n```'))
        result = model_client.call_json_model("system", "user", 0.2,
                                              client_override=self.fake_client(create),
                                              model_name="test-model")
        self.assertEqual(result, {"answer": "测试"})
        arguments = create.call_args.kwargs
        self.assertEqual(arguments["model"], "test-model")
        self.assertEqual(arguments["temperature"], 0.2)
        self.assertEqual(arguments["response_format"], {"type": "json_object"})
        self.assertEqual(arguments["messages"], [
            {"role": "system", "content": "system"}, {"role": "user", "content": "user"},
        ])

    def test_json_mode_failure_uses_existing_plain_mode_fallback(self):
        create = Mock(side_effect=[RuntimeError("JSON mode unavailable"), self.response('{"ok": true}')])
        with contextlib.redirect_stdout(io.StringIO()):
            result = model_client.call_json_model("system", "user", client_override=self.fake_client(create))
        self.assertEqual(result, {"ok": True})
        self.assertEqual(create.call_count, 2)
        self.assertNotIn("response_format", create.call_args.kwargs)

    def test_failed_or_unconfigured_client_retains_empty_fallback(self):
        self.assertEqual(model_client.call_json_model("system", "user", client_override=None), {})
        create = Mock(side_effect=RuntimeError("offline test"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = model_client.call_json_model("system", "user", client_override=self.fake_client(create))
        self.assertEqual(result, {})
        self.assertEqual(create.call_count, 2)
        self.assertEqual(model_client.parse_json_text("invalid"), {})

    def test_final_model_timeout_is_observable_and_retains_safe_fallback(self):
        failures = []
        token = model_client.model_failure_observer.set(failures.append)
        create = Mock(side_effect=TimeoutError("simulated timeout"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = model_client.call_json_model("system", "user", client_override=self.fake_client(create))
        finally:
            model_client.model_failure_observer.reset(token)
        self.assertEqual(result, {})
        self.assertEqual(create.call_count, 2)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], TimeoutError)


class ProcessRestartTests(unittest.TestCase):
    def test_sqlite_session_resumes_in_a_fresh_python_process(self):
        code = '''
import contextlib, io, json, sys
from pathlib import Path
from fastapi.testclient import TestClient
from database import connection
path = Path(sys.argv[1])
connection.DATA_DIR = path.parent
connection.DB_PATH = path
with contextlib.redirect_stdout(io.StringIO()):
    import app, rag
    with TestClient(app.app) as client:
        assert rag.embedding_model is None
        if sys.argv[2] == "create":
            body = client.post("/api/trigger", json={"trigger_type":"missed_measurement"}).json()
            sid = body["session_id"]
            reply = client.post("/api/chat", json={"session_id":sid,"message":"休息后血压145/85"})
            assert reply.status_code == 200
            assert reply.json()["state"]["current_goal"] == "check_medication"
            result = {"session_id":sid}
        else:
            sid = sys.argv[3]
            assert app.sessions == {}
            response = client.get("/api/debug/session/" + sid)
            assert response.status_code == 200
            assert response.json()["bp_systolic"] == 145
            for message in ("今天按时吃了降压药", "都没有"):
                response = client.post("/api/chat", json={"session_id":sid,"message":message})
                assert response.status_code == 200
            body = response.json()
            assert body["state"]["followup_status"] == "completed"
            assert body["report"]
            result = {"session_id":sid,"status":body["state"]["followup_status"]}
print(json.dumps(result))
'''
        # This regression exercises the original V3 session-restoration chain.
        # Pin its mode so a developer's .env opting into V5 cannot switch the
        # subprocess to a live semantic engine with no configured model key.
        environment = dict(os.environ, DEEPSEEK_API_KEY="", RAG_DISABLE_EMBEDDING="1",
                           LLM_AGENT_V4="0", LLM_AGENT_V42="0", LLM_AGENT_V50="0")
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "restart.db")
            first = subprocess.run([sys.executable, "-c", code, path, "create"],
                                   cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stderr)
            sid = json.loads(first.stdout.splitlines()[-1])["session_id"]
            second = subprocess.run([sys.executable, "-c", code, path, "resume", sid],
                                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(second.stdout.splitlines()[-1]), {
                "session_id": sid, "status": "completed",
            })


if __name__ == "__main__":
    unittest.main()

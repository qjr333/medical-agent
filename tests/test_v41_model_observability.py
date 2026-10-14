"""Offline provider-call tracing, fallback and concurrent observer regressions."""

import contextlib
import importlib
import importlib.util
import io
import json
import os
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch


MODEL_PATH = Path(__file__).resolve().parents[1] / "backend" / "model_client.py"
PRIVATE_TEXT = "private patient input with telephone 13812345678"
PRIVATE_KEY = "sk-private-observability-fixture-key"


def sdk_response(content):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=content),
    )])


def fake_sdk(*, responses=None, create=None):
    create = create if create is not None else Mock(side_effect=responses)
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create),
    ))


class ModelObservabilityV41Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing the Agent does not load the model SDK or call a provider.
        # The embedding dependency stays disabled throughout these fixtures.
        with patch.dict(os.environ, {"RAG_DISABLE_EMBEDDING": "1"}), \
                contextlib.redirect_stdout(io.StringIO()):
            cls.llm = importlib.import_module("agent.followup_agent")

    def setUp(self):
        self.module, _ = self.load_client_module()

    def load_client_module(self, *, api_key="", constructor=None):
        # Isolate configuration so no .env, real API key, SDK initialization or
        # external connection is needed even when the whole suite imports app.
        backend = importlib.import_module("backend")
        config = types.ModuleType("backend.config")
        config.DEEPSEEK_API_KEY = api_key
        config.DEEPSEEK_BASE_URL = "https://offline-observability.invalid"
        config.DEEPSEEK_MODEL = "offline-fixture-model"
        sdk = types.ModuleType("openai")
        sdk.OpenAI = constructor or Mock()
        spec = importlib.util.spec_from_file_location("isolated_v41_model_client", MODEL_PATH)
        module = importlib.util.module_from_spec(spec)
        output = io.StringIO()
        with patch.dict("sys.modules", {"backend.config": config, "openai": sdk}), \
                patch.object(backend, "config", config, create=True), \
                contextlib.redirect_stdout(output):
            spec.loader.exec_module(module)
        return module, output.getvalue()

    def call_observed(self, client):
        events = []
        token = self.module.model_call_observer.set(events.append)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                result = self.module.call_json_model(
                    "system fixture", PRIVATE_TEXT, client_override=client,
                )
        finally:
            self.module.model_call_observer.reset(token)
        return result, events, output.getvalue()

    def event_statuses(self, events):
        return [(event["mode"], event["status"]) for event in events]

    def llm_call(self, model=None, *, state=None):
        state = {} if state is None else state
        output = io.StringIO()
        with patch.dict("sys.modules", {"backend.model_client": self.module}), \
                contextlib.redirect_stdout(output):
            result = self.llm._call(model, state, "PLAN", "system fixture", {"text": PRIVATE_TEXT})
        detail = [event["detail"] for event in state["trace"]
                  if event["event"] == "LLM_CALL"][-1]
        return result, detail, output.getvalue()

    def test_json_sdk_success_records_actual_request_and_response(self):
        sdk = fake_sdk(responses=[sdk_response('{"answer":"ok"}')])
        result, events, _ = self.call_observed(sdk)
        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(self.event_statuses(events), [("json", "started"), ("json", "success")])
        self.assertTrue(all(event["provider"] == "deepseek" for event in events))
        self.assertTrue(all(set(event) == {"provider", "mode", "status", "error_type"}
                            for event in events))
        self.assertEqual(sdk.chat.completions.create.call_count, 1)
        self.assertEqual(sdk.chat.completions.create.call_args.kwargs["response_format"],
                         {"type": "json_object"})
        self.assertIsNone(self.module.model_call_observer.get())

    def test_json_failure_keeps_plain_mode_compatibility_and_records_both_requests(self):
        sdk = fake_sdk(responses=[RuntimeError(PRIVATE_TEXT),
                                  sdk_response('```json\n{"answer":"plain"}\n```')])
        result, events, output = self.call_observed(sdk)
        self.assertEqual(result, {"answer": "plain"})
        self.assertEqual(self.event_statuses(events), [
            ("json", "started"), ("json", "error"),
            ("plain", "started"), ("plain", "success"),
        ])
        self.assertEqual(events[1]["error_type"], "RuntimeError")
        calls = sdk.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("response_format", calls[0].kwargs)
        self.assertNotIn("response_format", calls[1].kwargs)
        self.assertEqual(calls[0].kwargs["messages"], calls[1].kwargs["messages"])
        self.assertNotIn(PRIVATE_TEXT, output)
        self.assertNotIn(PRIVATE_TEXT, json.dumps(events))

    def test_both_sdk_failures_record_errors_and_do_not_print_private_details(self):
        final_error = TimeoutError(PRIVATE_TEXT + " " + PRIVATE_KEY)
        sdk = fake_sdk(responses=[ValueError(PRIVATE_KEY), final_error])
        failures = []
        token = self.module.model_failure_observer.set(failures.append)
        try:
            result, events, output = self.call_observed(sdk)
        finally:
            self.module.model_failure_observer.reset(token)
        self.assertEqual(result, {})
        self.assertEqual(self.event_statuses(events), [
            ("json", "started"), ("json", "error"),
            ("plain", "started"), ("plain", "error"),
        ])
        self.assertEqual(failures, [final_error])
        self.assertEqual(events[-1]["error_type"], "TimeoutError")
        public = output + json.dumps(events)
        self.assertNotIn(PRIVATE_TEXT, public)
        self.assertNotIn(PRIVATE_KEY, public)

    def test_missing_client_has_no_fake_provider_request(self):
        result, events, output = self.call_observed(None)
        self.assertEqual(result, {})
        self.assertEqual(events, [])
        self.assertEqual(output, "")

    def test_sdk_initialization_error_does_not_print_exception_or_key(self):
        constructor = Mock(side_effect=RuntimeError(PRIVATE_KEY + " " + PRIVATE_TEXT))
        module, output = self.load_client_module(api_key=PRIVATE_KEY, constructor=constructor)
        self.assertIsNone(module.client)
        constructor.assert_called_once()
        self.assertIn("RuntimeError", output)
        self.assertNotIn(PRIVATE_KEY, output)
        self.assertNotIn(PRIVATE_TEXT, output)

    def test_injected_mock_does_not_claim_deepseek_provider_request(self):
        model = Mock(return_value={"action": "ASK_USER"})
        result, detail, _ = self.llm_call(model)
        self.assertEqual(result, {"action": "ASK_USER"})
        self.assertTrue(detail["called"])
        self.assertTrue(detail["ok"])
        self.assertFalse(detail["deepseek_called"])
        self.assertEqual(detail["provider_events"], [])
        model.assert_called_once()

    def test_llm_default_sdk_channel_reports_provider_request(self):
        sdk = fake_sdk(responses=[sdk_response('{"action":"ASK_USER"}')])
        with patch.object(self.module, "client", sdk):
            result, detail, _ = self.llm_call()
        self.assertEqual(result, {"action": "ASK_USER"})
        self.assertTrue(detail["deepseek_called"])
        self.assertEqual(self.event_statuses(detail["provider_events"]),
                         [("json", "started"), ("json", "success")])
        self.assertEqual(sdk.chat.completions.create.call_count, 1)

    def test_llm_sdk_failure_is_observed_without_private_error_text(self):
        sdk = fake_sdk(responses=[ValueError(PRIVATE_KEY), TimeoutError(PRIVATE_TEXT)])
        with patch.object(self.module, "client", sdk):
            result, detail, output = self.llm_call()
        self.assertIsNone(result)
        self.assertFalse(detail["ok"])
        self.assertEqual(detail["status"], "error")
        self.assertTrue(detail["deepseek_called"])
        self.assertEqual(self.event_statuses(detail["provider_events"]), [
            ("json", "started"), ("json", "error"),
            ("plain", "started"), ("plain", "error"),
        ])
        public = output + json.dumps(detail)
        self.assertNotIn(PRIVATE_KEY, public)
        self.assertNotIn(PRIVATE_TEXT, public)

    def test_llm_missing_client_is_failed_model_with_no_provider_request(self):
        with patch.object(self.module, "client", None):
            result, detail, _ = self.llm_call()
        self.assertIsNone(result)
        self.assertFalse(detail["ok"])
        self.assertTrue(detail["called"])
        self.assertFalse(detail["deepseek_called"])
        self.assertEqual(detail["provider_events"], [])

    def test_concurrent_llm_requests_keep_observers_and_traces_separate(self):
        barrier = Barrier(2)
        outer_events = {"retry": [], "success": []}

        def create(**kwargs):
            route = json.loads(kwargs["messages"][1]["content"])["route"]
            if "response_format" in kwargs:
                barrier.wait(timeout=3)
                if route == "retry":
                    raise ValueError(PRIVATE_TEXT)
            return sdk_response(json.dumps({"route": route}))

        sdk = fake_sdk(create=Mock(side_effect=create))

        def request(route):
            state = {}
            observer = outer_events[route].append
            token = self.module.model_call_observer.set(observer)
            try:
                result = self.llm._call(None, state, "PLAN", "system fixture", {"route": route})
                self.assertIs(self.module.model_call_observer.get(), observer)
                return result, state["trace"][-1]["detail"]
            finally:
                self.module.model_call_observer.reset(token)

        with patch.dict("sys.modules", {"backend.model_client": self.module}), \
                patch.object(self.module, "client", sdk), \
                contextlib.redirect_stdout(io.StringIO()), \
                ThreadPoolExecutor(max_workers=2) as executor:
            pending = [executor.submit(request, route) for route in ("retry", "success")]
            retry, success = [future.result(timeout=5) for future in pending]
        self.assertEqual(retry[0], {"route": "retry"})
        self.assertEqual(success[0], {"route": "success"})
        self.assertEqual(self.event_statuses(retry[1]["provider_events"]), [
            ("json", "started"), ("json", "error"),
            ("plain", "started"), ("plain", "success"),
        ])
        self.assertEqual(self.event_statuses(success[1]["provider_events"]),
                         [("json", "started"), ("json", "success")])
        self.assertEqual(outer_events, {"retry": [], "success": []})
        self.assertIsNone(self.module.model_call_observer.get())


if __name__ == "__main__":
    unittest.main()

"""Transport timeouts and retry ownership without real provider calls."""

import contextlib
import importlib
import importlib.util
import io
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


MODEL_PATH = Path(__file__).resolve().parents[1] / "backend" / "model_client.py"


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


def response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])


class ModelReliabilityTests(unittest.TestCase):
    def load_client(self, *, configured=False, timeout=50):
        backend = importlib.import_module("backend")
        config = ModuleType("backend.config")
        config.DEEPSEEK_API_KEY = "offline-key" if configured else ""
        config.DEEPSEEK_BASE_URL = "https://offline.invalid"
        config.DEEPSEEK_MODEL = "offline-model"
        config.DEEPSEEK_TIMEOUT_SECONDS = timeout
        sdk = ModuleType("openai")
        sdk.OpenAI = Mock()
        sdk.APITimeoutError = APITimeoutError
        sdk.APIConnectionError = APIConnectionError
        spec = importlib.util.spec_from_file_location("isolated_reliable_model", MODEL_PATH)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"backend.config": config, "openai": sdk}), \
                patch.object(backend, "config", config, create=True), \
                contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
        return module, sdk.OpenAI

    def observed_call(self, module, results, **kwargs):
        create = Mock(side_effect=results)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        events, failures = [], []
        call_token = module.model_call_observer.set(events.append)
        failure_token = module.model_failure_observer.set(failures.append)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = module.call_json_model("system", "user", client_override=client, **kwargs)
        finally:
            module.model_call_observer.reset(call_token)
            module.model_failure_observer.reset(failure_token)
        return result, create, events, failures

    def test_configured_http_timeout_disables_sdk_retries(self):
        module, constructor = self.load_client(configured=True, timeout=57)
        self.assertIs(module.client, constructor.return_value)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 57)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)

    def test_environment_timeout_defaults_and_invalid_values(self):
        config = importlib.import_module("backend.config")
        for value in ("", "invalid", "0", "-1", "nan", "inf", "301"):
            with self.subTest(value=value), patch.dict(os.environ, {"TEST_MODEL_TIMEOUT": value}):
                self.assertEqual(config._timeout_seconds("TEST_MODEL_TIMEOUT", 50), 50)
        with patch.dict(os.environ, {"TEST_MODEL_TIMEOUT": "60"}):
            self.assertEqual(config._timeout_seconds("TEST_MODEL_TIMEOUT", 50), 60)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config._timeout_seconds("TEST_MODEL_TIMEOUT", 50), 50)

    def test_temporary_errors_are_raised_before_plain_fallback(self):
        module, _ = self.load_client()
        for error, status in (
            (TimeoutError("private details"), "timeout"),
            (APITimeoutError("private details"), "timeout"),
            (APIConnectionError("private details"), "connection_error"),
        ):
            with self.subTest(error=type(error).__name__):
                create = Mock(side_effect=error)
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                events, failures = [], []
                call_token = module.model_call_observer.set(events.append)
                failure_token = module.model_failure_observer.set(failures.append)
                try:
                    with self.assertRaises(type(error)):
                        module.call_json_model("system", "user", client_override=client,
                                               raise_transient_errors=True)
                finally:
                    module.model_call_observer.reset(call_token)
                    module.model_failure_observer.reset(failure_token)
                self.assertEqual(create.call_count, 1)
                self.assertEqual(events[-1]["status"], status)
                self.assertEqual(error.model_request_mode, "json")
                self.assertEqual(failures, [error])
                self.assertNotIn("private details", str(events))

    def test_temporary_http_statuses_are_retryable_and_client_errors_are_not(self):
        module, _ = self.load_client()
        for status in (429, 500, 503, 599):
            error = RuntimeError("provider error")
            error.status_code = status
            self.assertTrue(module.is_temporary_model_error(error))
        for status in (400, 401, 403, 404, 422):
            error = RuntimeError("provider error")
            error.status_code = status
            self.assertFalse(module.is_temporary_model_error(error))

    def test_json_format_unsupported_switches_to_plain_only_once(self):
        module, _ = self.load_client()
        result, create, events, _ = self.observed_call(
            module, [RuntimeError("response_format is unsupported"), response()],
            raise_transient_errors=True,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(create.call_count, 2)
        self.assertNotIn("response_format", create.call_args.kwargs)
        self.assertEqual([(event["mode"], event["status"]) for event in events], [
            ("json", "started"), ("json", "error"), ("plain", "started"), ("plain", "success"),
        ])

    def test_non_transient_failure_does_not_launch_another_provider_call(self):
        module, _ = self.load_client()
        error = RuntimeError("invalid credentials")
        result, create, events, failures = self.observed_call(
            module, [error], raise_transient_errors=True,
        )
        self.assertEqual(result, {})
        self.assertEqual(create.call_count, 1)
        self.assertEqual(events[-1]["status"], "error")
        self.assertEqual(failures, [error])

    def test_plain_timeout_retry_keeps_plain_mode_and_skips_json_probe(self):
        module, _ = self.load_client()
        timeout = APITimeoutError("offline")
        create = Mock(side_effect=[RuntimeError("response_format unsupported"), timeout, response()])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(APITimeoutError) as caught:
                module.call_json_model("system", "user", client_override=client,
                                       raise_transient_errors=True)
            self.assertEqual(caught.exception.model_request_mode, "plain")
            result = module.call_json_model("system", "user", client_override=client,
                                           raise_transient_errors=True,
                                           request_mode=caught.exception.model_request_mode)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(create.call_count, 3)
        self.assertIn("response_format", create.call_args_list[0].kwargs)
        self.assertNotIn("response_format", create.call_args_list[1].kwargs)
        self.assertNotIn("response_format", create.call_args_list[2].kwargs)

    def test_legacy_call_retains_safe_plain_fallback(self):
        module, _ = self.load_client()
        result, create, events, failures = self.observed_call(
            module, [TimeoutError("offline"), TimeoutError("offline")],
        )
        self.assertEqual(result, {})
        self.assertEqual(create.call_count, 2)
        self.assertEqual([event["status"] for event in events], ["started", "error", "started", "error"])
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()

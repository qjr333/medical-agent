"""Regression checks for the small backend split, without network or real DB writes."""

import copy
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


# Preserve offline behavior even when a developer has a real key in .env.
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["RAG_DISABLE_EMBEDDING"] = "1"

from backend import config, extraction
from backend.patient_state import create_patient_state
from backend.utils import unique_list


def demo_fixture():
    return {
        "patient": {
            "patient_id": "test-patient",
            "patient_name": "测试患者",
            "age": 72,
            "conditions": ["高血压"],
        },
        "bp_history": [{"label": "上次", "systolic": 145, "diastolic": 85}],
        "trigger_configs": {
            "scheduled": {
                "label": "定时随访",
                "followup_reason": "计划随访",
                "bp_systolic": 150,
                "bp_diastolic": 90,
            }
        },
    }


class ConfigTests(unittest.TestCase):
    def test_project_paths_still_point_to_app_directory(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(Path(config.BASE_DIR), root)
        self.assertEqual(Path(config.DEMO_DATA_FILE), root / "demo_patient.json")
        self.assertEqual(Path(config.INDEX_FILE), root / "static" / "index.html")

    def test_demo_json_validation_uses_supplied_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patient.json"
            expected = demo_fixture()
            path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(config.load_demo_data(path), expected)

            invalid = copy.deepcopy(expected)
            invalid["patient"]["age"] = "72"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "patient.age"):
                config.load_demo_data(path)

            path.write_text("invalid json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "UTF-8 JSON"):
                config.load_demo_data(path)
            with self.assertRaisesRegex(RuntimeError, "缺少演示患者数据文件"):
                config.load_demo_data(Path(temporary) / "missing.json")


class PatientStateTests(unittest.TestCase):
    def setUp(self):
        self.data = demo_fixture()

    def make_state(self, trigger_type="scheduled"):
        return create_patient_state(
            trigger_type,
            DEMO_PATIENT=self.data["patient"],
            DEMO_BP_HISTORY=self.data["bp_history"],
            TRIGGER_CONFIGS=self.data["trigger_configs"],
        )

    def test_initial_state_fields_and_defaults_are_preserved(self):
        state = self.make_state()
        self.assertEqual(state["patient_id"], "test-patient")
        self.assertEqual((state["bp_systolic"], state["bp_diastolic"]), (150, 90))
        self.assertEqual(state["coverage"], {
            "bp": True,
            "medication": False,
            "current_symptoms": False,
            "red_flags": False,
            "measurement_context": False,
        })
        self.assertEqual(state["symptom_status"], "unknown")
        self.assertEqual(state["red_flag_status"], "unknown")
        self.assertEqual(state["followup_status"], "active")
        self.assertEqual(state["risk_level"], "normal")
        self.assertEqual(state["patient_memory"], {})
        self.assertIsNone(state["report"])

    def test_sessions_do_not_share_mutable_state_or_fixture_lists(self):
        first = self.make_state()
        second = self.make_state()
        first["conditions"].append("测试")
        first["bp_history"][0]["systolic"] = 200
        first["symptoms"].append("头晕")
        first["coverage"]["medication"] = True
        self.assertEqual(second["conditions"], ["高血压"])
        self.assertEqual(second["bp_history"][0]["systolic"], 145)
        self.assertEqual(second["symptoms"], [])
        self.assertFalse(second["coverage"]["medication"])
        self.assertEqual(self.data, demo_fixture())

    def test_unknown_trigger_preserves_original_factory_fallback(self):
        state = self.make_state("unknown")
        self.assertEqual(state["trigger_type"], "unknown")
        self.assertEqual(state["followup_reason"], "计划随访")


class ExtractionTests(unittest.TestCase):
    def test_bp_and_medication_rule_priority(self):
        self.assertEqual(extraction.extract_bp_from_text("血压162／96"), (162, 96))
        self.assertEqual(extraction.extract_bp_from_text("血压162/96"), (162, 96))
        self.assertEqual(extraction.extract_bp_from_text("血压1162/96"), (None, None))
        self.assertEqual(extraction.detect_medication_status("今天忘了吃药，后来吃了"), "missed")
        self.assertEqual(extraction.detect_medication_status("自己停药"), "stopped")
        self.assertEqual(extraction.detect_medication_status("按时吃了"), "taken")

    def test_symptom_negation_and_stable_deduplication(self):
        self.assertEqual(extraction.detect_symptoms("没有胸痛，但有头晕"), (["头晕"], ["胸痛"]))
        self.assertEqual(unique_list([" a ", "a", "", None, {"b": 1}, {"b": 1}]), ["a", {"b": 1}])

    def test_deterministic_values_override_model_with_injected_dependencies(self):
        model = Mock(return_value={
            "bp_systolic": 100,
            "bp_diastolic": 60,
            "medication_status": "taken",
            "answered_topics": ["bp"],
        })
        detector = Mock(return_value=True)
        message = "坐着休息后162/96，忘了吃药，没有胸痛，怎么办？"
        result = extraction.extract_user_patch(
            message,
            call_json_model=model,
            contains_red_flag_language=detector,
        )
        self.assertEqual((result["bp_systolic"], result["bp_diastolic"]), (162, 96))
        self.assertEqual(result["medication_status"], "missed")
        self.assertEqual(result["denied_symptoms"], ["胸痛"])
        self.assertEqual(result["patient_question"], message)
        self.assertEqual(result["answered_topics"], ["bp", "medication", "current_symptoms", "red_flags", "measurement_context"])
        model.assert_not_called()
        detector.assert_called_once_with(message)

    def test_non_object_model_output_uses_rule_fallback(self):
        result = extraction.extract_user_patch(
            "150/90，按时吃了药",
            call_json_model=lambda **kwargs: [],
            contains_red_flag_language=lambda text: False,
        )
        self.assertEqual(result["bp_systolic"], 150)
        self.assertEqual(result["medication_status"], "taken")
        self.assertEqual(result["symptoms"], [])


class AppCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = importlib.import_module("app")

    def test_existing_model_patch_point_and_extraction_exports_still_work(self):
        self.assertIs(self.app_module.detect_symptoms, extraction.detect_symptoms)
        self.assertIs(self.app_module.unique_list, unique_list)
        with patch.object(self.app_module, "call_json_model", return_value={"additional_information": "模型提取信息"}) as model:
            result = self.app_module.extract_user_patch("150/90，没有胸痛")
        self.assertEqual(result["additional_information"], "")
        self.assertEqual(result["bp_systolic"], 150)
        self.assertIn("red_flags", result["answered_topics"])
        model.assert_not_called()

        # 模型注入点仍用于未被简单规则解释的回答，只接受原话支持的内容。
        with patch.object(self.app_module, "call_json_model", return_value={
            "additional_information": "不确定", "medication_status": "taken",
        }) as model:
            result = self.app_module.extract_user_patch("不确定")
        model.assert_called_once()
        self.assertEqual(result["additional_information"], "不确定")
        self.assertEqual(result["medication_status"], "unknown")

    def test_state_adapter_keeps_replaceable_demo_configuration(self):
        data = demo_fixture()
        with patch.multiple(self.app_module,
                            DEMO_PATIENT=data["patient"],
                            DEMO_BP_HISTORY=data["bp_history"],
                            TRIGGER_CONFIGS=data["trigger_configs"]):
            state = self.app_module.create_patient_state("scheduled")
        self.assertEqual(state["patient_id"], "test-patient")
        self.assertEqual(state["bp_systolic"], 150)


if __name__ == "__main__":
    unittest.main()

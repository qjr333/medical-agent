"""Trigger-aware LLM V4 planning; no real network/model calls."""
import json
import unittest
from pathlib import Path

from agent.llm_v4 import PLANNING_PROMPT, _planner_payload, run_agent_v4
from agent.state_manager import create_patient_state

ROOT = Path(__file__).resolve().parents[1]

class TriggerAwareV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo = json.loads((ROOT / "demo_patient.json").read_text(encoding="utf-8"))
        cls.demo = demo

    def make_state(self, kind):
        return create_patient_state(
            kind,
            DEMO_PATIENT=self.demo["patient"],
            DEMO_BP_HISTORY=self.demo["bp_history"],
            TRIGGER_CONFIGS=self.demo["trigger_configs"],
        )

    def test_trigger_fixtures_are_not_the_same_measurement_event(self):
        scheduled = self.make_state("scheduled")
        high = self.make_state("high_bp")
        self.assertIsNone(scheduled["bp_systolic"])
        self.assertFalse(scheduled["coverage"]["bp"])
        self.assertEqual((high["bp_systolic"], high["bp_diastolic"]), (162, 96))
        self.assertTrue(high["coverage"]["bp"])

    def test_planning_payload_has_explicit_trigger_and_objective(self):
        contexts = {}
        for kind in ("scheduled", "high_bp", "missed_measurement"):
            contexts[kind] = _planner_payload(self.make_state(kind), [])["trigger_event"]
            self.assertEqual(contexts[kind]["type"], kind)
            self.assertTrue(contexts[kind]["objective"])
        self.assertEqual(len({x["objective"] for x in contexts.values()}), 3)
        self.assertIsNone(contexts["scheduled"]["new_measurement"])
        self.assertIsNone(contexts["missed_measurement"]["new_measurement"])
        self.assertEqual(contexts["high_bp"]["new_measurement"]["systolic"], 162)
        self.assertIn("trigger_event.type", PLANNING_PROMPT)

    def test_first_llm_call_receives_correct_trigger_context(self):
        for kind in ("scheduled", "high_bp", "missed_measurement"):
            with self.subTest(kind=kind):
                state = self.make_state(kind)
                observed = []
                def fake_model(*, system_prompt, user_prompt, temperature):
                    payload = json.loads(user_prompt)
                    observed.append(payload["trigger_event"])
                    return {
                        "action": "ASK_USER",
                        "goal": "trigger_aware_followup",
                        "topic": "current_symptoms",
                        "question": "现在身体有没有不舒服？",
                        "reason": "模拟LLM收到完整的触发上下文",
                    }
                response = run_agent_v4(state, call_json_model=fake_model)
                self.assertTrue(response["message"])
                self.assertEqual(observed[0]["type"], kind)
                self.assertTrue(any(e["event"] == "LLM_CALL" and e["detail"].get("ok")
                                    for e in state["trace"]))

if __name__ == "__main__":
    unittest.main()

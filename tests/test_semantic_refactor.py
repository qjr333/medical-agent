"""V5 semantic contracts use explicit offline model responses, never real credentials."""

import json
import unittest
from copy import deepcopy
from datetime import date, timedelta
from unittest.mock import Mock, patch

from agent import extractor, followup_agent
from agent.state_manager import merge_user_patch
from backend.utils import today_local_date


def fact(kind, value, evidence, *, claim_id="c1", temporal="current", polarity="affirmed", **fields):
    return {"id": claim_id, "kind": kind, "value": value, "evidence": evidence,
            "temporal": temporal, "polarity": polarity, **fields}


class SemanticModel:
    """Test double: tests supply facts and review verdicts; it does not parse patient text."""
    def __init__(self, answer, *, reviews=None, screening_review=None):
        self.answer = deepcopy(answer)
        self.answer.setdefault("answer_status", "answered")
        self.reviews = reviews
        self.screening_review = screening_review
        self.inputs = []

    def __call__(self, *, system_prompt, user_prompt, **kwargs):
        payload = json.loads(user_prompt)
        self.inputs.append(payload)
        if "claims" not in payload:
            return deepcopy(self.answer)
        rows = self.reviews
        if rows is None:
            rows = [{key: candidate[key] for key in ("id", "polarity", "temporal", "date") if key in candidate}
                    | {"supported": True} for candidate in payload["claims"]]
        screening = self.screening_review
        if screening is None:
            screening = {**payload.get("screening", {}), "supported": True}
        return {"reviews": deepcopy(rows), "screening": deepcopy(screening)}


def current_state(topic="medication", question="今天已经服用降压药了吗？", flags=None):
    return {"agent_version": "V5.0", "patient_id": "semantic-fictional",
            "medication_status": "unknown", "medication_timing": "unknown",
            "symptoms": [], "red_flags": [], "denied_symptoms": [], "denied_red_flags": [],
            "last_agent_question": question, "current_goal": "确认患者本轮信息",
            "question_context": {"question": question, "topic": topic, "goal": "确认患者本轮信息",
                                 "kind": "medication_taken" if topic == "medication" else "symptoms",
                                 "asked_red_flags": flags or [], "asked_symptoms": []},
            "coverage": {key: False for key in followup_agent.TOPICS}, "trace": [], "followup_status": "active"}


class SemanticExtractionTests(unittest.TestCase):
    def extract(self, text, claims=None, *, state=None, reviews=None, screening=None, screening_review=None, **answer):
        model = SemanticModel({"claims": claims or [], "answer_status": "answered",
                               "screening": screening or {"adequate": False}, **answer},
                              reviews=reviews, screening_review=screening_review)
        result = followup_agent.extract_patient_information(text, state or current_state(), call_json_model=model, enable_v50=True)
        return result, model

    def test_v5_does_not_use_legacy_regex_extraction(self):
        with patch.object(followup_agent, "legacy_extract", side_effect=AssertionError("Legacy parser is not V5 semantic engine")):
            data, model = self.extract("今天量的是138/84", [fact("bp", {"systolic": 138, "diastolic": 84}, "今天量的是138/84")])
        self.assertEqual((data["bp_systolic"], data["bp_diastolic"]), (138, 84))
        self.assertEqual(len(model.inputs), 2)
        self.assertTrue(data["semantic_validated"])

    def test_compatibility_entry_dispatches_v5_to_semantic_engine(self):
        model = SemanticModel({"claims": [fact("medication", {"status": "not_taken_yet"}, "还没呢")]})
        result = extractor.extract_user_patch("还没呢", current_state(), call_json_model=model)
        self.assertEqual(result["medication_status"], "not_taken_yet")
        self.assertEqual(len(model.inputs), 2)

    def test_short_not_yet_uses_actual_previous_question(self):
        data, model = self.extract("还没呢", [fact("medication", {"status": "not_taken_yet", "timing": "unknown"}, "还没呢")])
        self.assertEqual(data["medication_status"], "not_taken_yet")
        self.assertEqual(data["topic_status"]["medication"], "clarification_needed")
        self.assertIn("已经服用", model.inputs[0]["last_agent_question"])

    def test_same_no_denies_adherence_without_inventing_taken(self):
        state = current_state(question="今天有没有漏服或自己停药？")
        data, _ = self.extract("没有", [fact("medication", {"status": "unknown", "adherence": "no_missed_or_stopped"},
                                             "没有", polarity="denied")], state=state)
        self.assertEqual(data["medication_status"], "unknown")
        self.assertEqual(data["medication_adherence"], "no_missed_or_stopped")

    def test_affirmative_taken_is_semantically_verified(self):
        data, _ = self.extract("嗯，吃过了", [fact("medication", {"status": "taken"}, "吃过了")])
        self.assertEqual(data["medication_status"], "taken")

    def test_overdue_does_not_invent_a_dose_or_clock_time(self):
        state = current_state(question="是还没到规定服药时间，还是超过了？")
        state["medication_status"] = "not_taken_yet"
        data, _ = self.extract("超过了", [fact("medication", {"status": "not_taken_yet", "timing": "overdue"}, "超过了")], state=state)
        self.assertEqual(data["medication_status"], "not_taken_yet")
        self.assertEqual(data["medication_timing"], "overdue")
        self.assertEqual(data["medication_detail"], "超过了")

    def test_unable_answer_does_not_turn_into_a_negative_fact(self):
        data, model = self.extract("我记不清了", answer_status="unable")
        self.assertEqual(data["medication_status"], "unknown")
        self.assertEqual(data["answer_status"], "unable")
        self.assertFalse(data["denied_symptoms"])
        self.assertEqual(len(model.inputs), 1)

    def test_no_model_claims_means_no_regex_backfill(self):
        data, _ = self.extract("血压140/90，今天吃药了")
        self.assertIsNone(data["bp_systolic"])
        self.assertEqual(data["medication_status"], "unknown")

    def test_fabricated_evidence_is_rejected_before_review(self):
        data, model = self.extract("今天还没测", [fact("bp", {"systolic": 145, "diastolic": 85}, "145/85")])
        self.assertIsNone(data["bp_systolic"])
        self.assertEqual(len(model.inputs), 1)

    def test_wrong_numbers_and_out_of_range_values_are_rejected(self):
        for values in ({"systolic": 168, "diastolic": 104}, {"systolic": 999, "diastolic": 998},
                       {"systolic": True, "diastolic": 85}):
            with self.subTest(values=values):
                data, _ = self.extract("今天血压138/84", [fact("bp", values, "今天血压138/84")])
                self.assertIsNone(data["bp_systolic"])

    def test_chinese_spoken_bp_numbers_keep_grounded_measurement(self):
        text = "今天早晨量的，一百三十二，低压七十八"
        data, _ = self.extract(text, [fact("bp", {"systolic": 132, "diastolic": 78}, text)])
        self.assertEqual((data["bp_systolic"], data["bp_diastolic"]), (132, 78))

    def test_quote_substring_is_not_enough_to_assert_symptom(self):
        candidate = fact("symptom", {"name": "胸痛", "red_flag": True}, "胸痛")
        rejected = {"id": "c1", "supported": False, "temporal": "current", "polarity": "affirmed"}
        data, model = self.extract("我没有胸痛", [candidate], reviews=[rejected])
        self.assertFalse(data["red_flags"])
        self.assertEqual(model.inputs[1]["patient_message"], "我没有胸痛")

    def test_history_symptom_is_not_a_current_emergency(self):
        state = current_state("current_symptoms", "现在身体怎么样？")
        text = "去年住院那天喘不上气，今天已经没有这种症状了"
        data, _ = self.extract(text, [fact("symptom", {"name": "呼吸困难", "red_flag": True}, "去年住院那天喘不上气", temporal="history")], state=state)
        self.assertFalse(data["red_flags"])
        merge_user_patch(state, data, text)
        self.assertFalse(state["red_flags"])

    def test_hypothetical_symptom_is_not_a_current_emergency(self):
        text = "如果哪天呼吸困难，应该怎么办？"
        data, _ = self.extract(text, [fact("symptom", {"name": "呼吸困难", "red_flag": True}, "如果哪天呼吸困难", temporal="hypothetical")])
        self.assertFalse(data["red_flags"])

    def test_reviewed_temporal_disagreement_rejects_the_claim(self):
        review = {"id": "c1", "supported": True, "temporal": "history", "polarity": "affirmed"}
        data, _ = self.extract("昨天血压160/100", [fact("bp", {"systolic": 160, "diastolic": 100}, "160/100")], reviews=[review])
        self.assertIsNone(data["bp_systolic"])

    def test_current_and_historical_medication_keep_separate_dates(self):
        today = today_local_date()
        yesterday = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
        text = "昨天忘记吃药，今天按时吃了"
        data, _ = self.extract(text, [fact("medication", {"status": "missed"}, "昨天忘记吃药", temporal="history", date=yesterday),
                                    fact("medication", {"status": "taken"}, "今天按时吃了", claim_id="c2")])
        self.assertEqual(data["medication_status"], "taken")
        self.assertEqual({(row["date"], row["status"]) for row in data["medication_observations"]}, {(today, "taken"), (yesterday, "missed")})

    def test_current_dose_with_past_date_is_not_written_today(self):
        yesterday = (date.fromisoformat(today_local_date()) - timedelta(days=1)).isoformat()
        data, _ = self.extract("昨天吃了", [fact("medication", {"status": "taken"}, "昨天吃了", date=yesterday)])
        self.assertEqual(data["medication_status"], "unknown")

    def test_meaningful_screening_records_context_without_six_denials(self):
        question = "现在有没有胸痛或喘不上气？"
        state = current_state("red_flags", question, ["胸痛", "呼吸困难"])
        screening = {"adequate": True, "evidence": "都没有", "scope": ["胸痛", "呼吸困难"], "question": question}
        data, _ = self.extract("都没有", state=state, screening=screening)
        self.assertTrue(data["_semantic"]["screening"]["adequate"])
        self.assertEqual(data["_semantic"]["screening"]["scope"], ["胸痛", "呼吸困难"])
        self.assertFalse(data["denied_red_flags"])

    def test_vague_fine_is_not_complete_danger_screen(self):
        question = "最近怎样？"
        bad = {"adequate": True, "evidence": "挺好", "scope": ["胸痛"], "question": question}
        data, _ = self.extract("挺好", state=current_state("current_symptoms", question), screening=bad,
                               screening_review={**bad, "supported": False, "adequate": False})
        self.assertNotIn("screening", data["_semantic"])
        self.assertNotIn("red_flags", data["answered_topics"])

    def test_screening_from_another_question_is_not_accepted(self):
        bad = {"adequate": True, "evidence": "没有", "scope": ["胸痛"], "question": "另一个问题"}
        data, _ = self.extract("没有", screening=bad)
        self.assertNotIn("screening", data["_semantic"])

    def test_monitoring_plan_and_end_request_use_verified_semantics(self):
        text = "明天我会测血压，今天先聊到这里吧"
        data, _ = self.extract(text, [fact("monitoring_plan", True, "明天我会测血压"),
                                    fact("end_request", True, "今天先聊到这里吧", claim_id="c2")])
        self.assertTrue(data["_semantic"]["monitoring_plan"]["confirmed"])
        self.assertTrue(data["_semantic"]["patient_requested_end"]["confirmed"])

    def test_explicit_no_end_clears_requested_end_assessment(self):
        data, _ = self.extract("先别结束", [fact("end_request", False, "先别结束", polarity="denied")])
        self.assertFalse(data["_semantic"]["patient_requested_end"]["confirmed"])

    def test_review_failure_is_visible_and_does_not_publish_facts(self):
        model = Mock(side_effect=[{"claims": [fact("medication", {"status": "taken"}, "吃了")]}, {}])
        state = current_state()
        data = followup_agent.extract_patient_information("吃了", state, call_json_model=model, enable_v50=True)
        self.assertEqual(data["medication_status"], "unknown")
        self.assertFalse(data["semantic_validated"])
        self.assertFalse(state["llm_v4_last_extraction_ok"])

    def test_simple_current_emergency_survives_model_failure(self):
        for text in ("我现在喘不上气", "没有胸痛，但是呼吸困难", "我现在有突发肢体无力和突发言语异常", "我快死了"):
            with self.subTest(text=text):
                model = Mock(side_effect=AssertionError("Immediate safety handling must not wait for a model"))
                data = followup_agent.extract_patient_information(text, current_state(), call_json_model=model, enable_v50=True)
                self.assertTrue(data["red_flags"])
                model.assert_not_called()

    def test_malformed_structured_claims_do_not_crash_or_publish_facts(self):
        responses = [{"claims": 3}, {"claims": [None, "bad", {"kind": [], "evidence": "吃了"}]},
                     {"claims": [fact("medication", {"status": []}, "吃了")], "answer_status": []},
                     {"claims": [fact("medication", {"status": "taken"}, "吃了", temporal=[])]}]
        for response in responses:
            with self.subTest(response=response):
                model = Mock(return_value=response)
                data = followup_agent.extract_patient_information("吃了", current_state(), call_json_model=model, enable_v50=True)
                self.assertEqual(data["medication_status"], "unknown")
                self.assertFalse(data["medication_observations"])

    def test_red_flag_denial_does_not_invent_a_general_symptom_denial(self):
        data, _ = self.extract("没有胸痛", [fact("symptom", {"name": "胸痛", "red_flag": True}, "没有胸痛", polarity="denied")])
        self.assertEqual(data["denied_red_flags"], ["胸痛"])
        self.assertFalse(data["denied_symptoms"])
        self.assertNotIn("current_symptoms", data["answered_topics"])

    def test_canonical_danger_cannot_be_downgraded_by_model_boolean(self):
        for name in ("呼吸困难", "喘不上气"):
            with self.subTest(name=name):
                text = "从刚刚开始感觉" + name
                data, _ = self.extract(text, [fact("symptom", {"name": name, "red_flag": False}, text)])
                self.assertIn("呼吸困难", data["red_flags"])

    def test_flat_or_invalid_extraction_schema_is_not_a_successful_unknown_answer(self):
        for answer in ({"answer_status": "answered"}, {"claims": [], "answer_status": []},
                       {"claims": [fact("medication", {"status": "taken"}, "编造原话")], "answer_status": "answered"}):
            with self.subTest(answer=answer):
                state = current_state()
                data = followup_agent.extract_patient_information("从刚刚开始感觉身体明显变坏", state,
                                                  call_json_model=Mock(return_value=answer), enable_v50=True)
                self.assertFalse(state["llm_v4_last_extraction_ok"])
                self.assertFalse(data["medication_observations"])

    def test_unsupported_claim_requires_pause_instead_of_finishing_old_state(self):
        state = current_state()
        text = "之前的症状现在好像又有点变化"
        data, _ = self.extract(text, [fact("symptom", {"name": "头晕", "red_flag": False}, text)], state=state,
                               reviews=[{"id": "c1", "supported": False, "temporal": "current", "polarity": "affirmed"}])
        self.assertFalse(state["llm_v4_last_extraction_ok"])
        self.assertFalse(data["symptoms"])

    def test_screen_scope_cannot_expand_the_actual_previous_question(self):
        question = "现在有没有胸痛？"
        state = current_state("red_flags", question, ["胸痛"])
        wrong = {"adequate": True, "evidence": "没有", "question": question, "scope": ["呼吸困难"]}
        data, _ = self.extract("没有", state=state, screening=wrong)
        self.assertNotIn("screening", data["_semantic"])
        self.assertFalse(state["llm_v4_last_extraction_ok"])

    def test_history_denial_or_hypothesis_does_not_trigger_phrase_backstop(self):
        for text in ("昨天喘不上气", "我没有胸痛和呼吸困难", "如果胸痛或呼吸困难", "以前有突发肢体无力和突发言语异常"):
            with self.subTest(text=text):
                data, model = self.extract(text)
                self.assertFalse(data["red_flags"])
                self.assertEqual(len(model.inputs), 1)


if __name__ == "__main__":
    unittest.main()

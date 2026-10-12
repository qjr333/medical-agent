"""上下文提取回归：不联网，不使用现有患者数据或真实数据库。"""

import unittest
from datetime import date
from unittest.mock import Mock, patch

from agent import extractor
from medical_tools.safety import DANGER_PATTERNS, detect_danger_signs


class ContextExtractorTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(extractor, "_today", return_value=date(2026, 10, 11))
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def state(self, kind="medication_taken", question=None, goal="check_medication", **extra):
        state = {
            "current_goal": goal,
            "medication_status": "unknown",
            "question_context": {
                "kind": kind,
                "goal": goal,
                "topic": "medication",
                "question": question or "今天是否按照医生安排服用了降压药？",
            },
            "topic_status": {},
        }
        state.update(extra)
        return state

    def extract(self, text, state=None, model=None):
        model = model or Mock(return_value={})
        result = extractor.extract_user_patch(text, state, call_json_model=model)
        return result, model

    def test_simple_medication_answers_have_no_model_cost(self):
        cases = [
            ("没有呢", "not_taken_yet", "unknown", "clarification_needed"),
            ("还没吃", "not_taken_yet", "unknown", "clarification_needed"),
            ("已经吃了", "taken", "unknown", "answered"),
            ("还没到吃药时间", "not_taken_yet", "not_due", "answered"),
            ("今天忘记吃药了", "missed", "unknown", "answered"),
            ("是的", "taken", "unknown", "answered"),
        ]
        for text, status, timing, topic_status in cases:
            with self.subTest(text=text):
                result, model = self.extract(text, self.state())
                self.assertEqual(result["medication_status"], status)
                self.assertEqual(result["medication_timing"], timing)
                self.assertEqual(result["topic_status"]["medication"], topic_status)
                self.assertEqual(result["medication_observations"][0]["date"], "2026-10-11")
                self.assertEqual("medication" in result["answered_topics"], topic_status == "answered")
                model.assert_not_called()

    def test_denial_of_missed_or_stopped_is_not_non_adherence(self):
        for text in ("没有", "没有呢", "没有漏服", "没有停药", "没忘记吃药"):
            with self.subTest(text=text):
                state = self.state("adherence", "今天有没有漏服或者自行停药？")
                result, model = self.extract(text, state)
                self.assertEqual(result["medication_status"], "unknown")
                self.assertEqual(result["medication_adherence"], "no_missed_or_stopped")
                self.assertEqual(result["topic_status"]["medication"], "clarification_needed")
                self.assertEqual(result["medication_observations"], [])
                self.assertNotIn("medication", result["answered_topics"])
                model.assert_not_called()

    def test_same_no_answer_uses_goal_and_actual_question(self):
        taking, _ = self.extract("没有", self.state())
        adherence, _ = self.extract("没有", self.state("adherence", "今天有没有漏服或者停药？"))
        symptoms_state = self.state("symptoms", "有没有头晕、胸痛或者呼吸困难？", "check_symptoms")
        symptoms_state["question_context"].update(topic="current_symptoms", asked_symptoms=["头晕", "胸痛", "呼吸困难"], asked_red_flags=["胸痛", "呼吸困难"])
        symptoms, _ = self.extract("没有", symptoms_state)
        self.assertEqual(taking["medication_status"], "not_taken_yet")
        self.assertEqual(adherence["medication_status"], "unknown")
        self.assertEqual(symptoms["medication_status"], "unknown")
        self.assertEqual(symptoms["denied_symptoms"], ["头晕", "胸痛", "呼吸困难"])
        self.assertEqual(symptoms["denied_red_flags"], ["胸痛", "呼吸困难"])
        self.assertNotIn("晕厥", symptoms["denied_red_flags"])

    def test_short_symptom_denial_only_covers_asked_symptoms(self):
        state = self.state("symptoms", "现在有没有头晕、胸痛或者呼吸困难？", "check_symptoms")
        state["question_context"].update(topic="current_symptoms", asked_symptoms=["头晕", "胸痛", "呼吸困难"], asked_red_flags=["胸痛", "呼吸困难"])
        for text in ("都没有", "都没有不舒服", "没有呢"):
            with self.subTest(text=text):
                result, model = self.extract(text, state)
                self.assertEqual(result["symptoms"], [])
                self.assertEqual(result["denied_symptoms"], ["头晕", "胸痛", "呼吸困难"])
                self.assertEqual(result["denied_red_flags"], ["胸痛", "呼吸困难"])
                self.assertNotIn("medication", result["answered_topics"])
                model.assert_not_called()

    def test_explicit_generic_denial_does_not_answer_medication_question(self):
        result, model = self.extract("都没有不舒服", self.state())
        self.assertEqual(result["medication_status"], "unknown")
        self.assertEqual(result["medication_observations"], [])
        self.assertEqual(result["denied_red_flags"], [])
        model.assert_not_called()

    def test_contrast_preserves_positive_danger_after_negative_clause(self):
        for text in ("没有胸痛，但是呼吸困难", "没有胸痛，但是现在喘不上气", "没有头晕、胸痛，不过现在无法呼吸"):
            with self.subTest(text=text):
                result, model = self.extract(text)
                self.assertIn("胸痛", result["denied_symptoms"])
                self.assertIn("呼吸困难", result["symptoms"])
                self.assertNotIn("呼吸困难", result["denied_red_flags"])
                danger = []
                for clause in extractor.split_answer_clauses(text):
                    danger.extend(detect_danger_signs(clause))
                self.assertIn("呼吸困难", danger)
                model.assert_not_called()

    def test_negation_list_does_not_cross_clause_boundary(self):
        self.assertEqual(extractor.detect_symptoms("没有头晕、胸痛或者呼吸困难"), ([], ["头晕", "胸痛", "呼吸困难"]))
        self.assertEqual(extractor.detect_symptoms("没有胸痛，有呼吸困难"), (["呼吸困难"], ["胸痛"]))
        self.assertEqual(extractor.detect_symptoms("没有胸痛，但现在胸痛"), (["胸痛"], []))

    def test_yes_to_two_alternatives_is_ambiguous(self):
        state = self.state("timing", "还没到规定服药时间，还是已经超过原定时间？", medication_status="not_taken_yet")
        result, model = self.extract("是的", state)
        self.assertEqual(result["medication_status"], "unknown")
        self.assertEqual(result["medication_timing"], "unknown")
        self.assertEqual(result["topic_status"]["medication"], "clarification_needed")
        model.assert_not_called()

    def test_timing_confirmation_does_not_invent_missed_dose(self):
        state = self.state("timing", "还没到规定服药时间，还是已经超过原定时间？", medication_status="not_taken_yet")
        for text, timing in (("还没到时间", "not_due"), ("已经超过时间了", "overdue")):
            with self.subTest(text=text):
                result, model = self.extract(text, state)
                self.assertEqual(result["medication_status"], "not_taken_yet")
                self.assertEqual(result["medication_timing"], timing)
                self.assertEqual(result["topic_status"]["medication"], "answered")
                model.assert_not_called()

    def test_uncertain_model_cannot_invent_facts_or_coverage(self):
        model = Mock(return_value={
            "bp_systolic": 180, "bp_diastolic": 120, "medication_status": "taken",
            "symptoms": ["胸痛"], "denied_symptoms": ["呼吸困难"], "denied_red_flags": list(DANGER_PATTERNS),
            "answered_topics": ["medication", "red_flags"], "topic_status": {"medication": "answered"},
            "additional_information": "患者正常按时服药，所有危险症状均已排除",
        })
        result, model = self.extract("不确定", self.state(), model)
        self.assertEqual(result["medication_status"], "unknown")
        self.assertEqual((result["bp_systolic"], result["bp_diastolic"]), (None, None))
        self.assertEqual(result["symptoms"], [])
        self.assertEqual(result["denied_symptoms"], [])
        self.assertEqual(result["denied_red_flags"], [])
        self.assertEqual(result["answered_topics"], [])
        self.assertEqual(result["additional_information"], "")
        self.assertEqual(result["topic_status"]["medication"], "clarification_needed")
        model.assert_called_once()
        self.assertIn("question_context", model.call_args.kwargs["user_prompt"])

    def test_complex_model_extract_can_only_retain_verbatim_information(self):
        text = "这两天晚上睡得不太好，具体原因我也说不清"
        model = Mock(return_value={"additional_information": "晚上睡得不太好", "medication_status": "missed"})
        result, model = self.extract(text, self.state(), model)
        self.assertEqual(result["additional_information"], "晚上睡得不太好")
        self.assertEqual(result["medication_status"], "unknown")
        model.assert_called_once()

    def test_uncertain_symptoms_are_not_converted_to_denials(self):
        result, _ = self.extract("不确定有没有胸痛", self.state("symptoms", "有没有胸痛？", "check_symptoms"))
        self.assertEqual(result["denied_symptoms"], [])
        self.assertEqual(result["denied_red_flags"], [])
        self.assertNotEqual(result["topic_status"].get("current_symptoms"), "answered")
        self.assertNotIn("red_flags", result["answered_topics"])

    def test_dated_medication_observations_do_not_overwrite_current_day(self):
        yesterday, _ = self.extract("昨天忘记吃药了", self.state())
        self.assertEqual(yesterday["medication_status"], "unknown")
        self.assertEqual(yesterday["medication_observations"], [{"date": "2026-10-10", "status": "missed", "detail": "昨天忘记吃药了", "timing": "unknown"}])
        both, _ = self.extract("昨天漏服，今天已经吃了", self.state())
        self.assertEqual(both["medication_status"], "taken")
        self.assertEqual([(item["date"], item["status"]) for item in both["medication_observations"]], [("2026-10-10", "missed"), ("2026-10-11", "taken")])
        future, _ = self.extract("明天还没到吃药时间", self.state("timing", medication_status="not_taken_yet"))
        self.assertEqual(future["medication_status"], "unknown")
        self.assertEqual(future["medication_observations"][0]["date"], "2026-10-12")
        previous, _ = self.extract("前天停药了", self.state())
        self.assertEqual(previous["medication_status"], "unknown")
        self.assertEqual(previous["medication_observations"][0]["date"], "2026-10-09")

    def test_bp_and_nonmedication_food_are_preserved(self):
        result, model = self.extract("坐着休息后血压150／90，没有胸痛")
        self.assertEqual((result["bp_systolic"], result["bp_diastolic"]), (150, 90))
        self.assertEqual(result["denied_symptoms"], ["胸痛"])
        self.assertIn("measurement_context", result["answered_topics"])
        model.assert_not_called()
        self.assertEqual(extractor.detect_medication_status("刚吃了晚饭"), "unknown")
        self.assertEqual(extractor.detect_medication_status("还没吃饭"), "unknown")
        self.assertEqual(extractor.detect_medication_status("今天忘记测血压了"), "unknown")
        self.assertEqual(extractor.detect_medication_status("忘记吃晚饭了"), "unknown")

    def test_short_answers_without_context_remain_unknown(self):
        for text in ("没有呢", "是的"):
            with self.subTest(text=text):
                result, model = self.extract(text)
                self.assertEqual(result["medication_status"], "unknown")
                self.assertEqual(result["symptoms"], [])
                self.assertEqual(result["denied_symptoms"], [])
                self.assertEqual(result["denied_red_flags"], [])
                self.assertEqual(result["answered_topics"], [])
                model.assert_not_called()

    def test_uncertain_symptom_answer_does_not_copy_confirmed_medication(self):
        state = self.state("symptoms", "现在有没有胸痛？", "check_symptoms", medication_status="taken", symptoms=["头晕"])
        state["question_context"]["topic"] = "current_symptoms"
        result, model = self.extract("不确定", state)
        self.assertEqual(result["medication_status"], "unknown")
        self.assertEqual(result["symptoms"], [])
        self.assertEqual(result["answered_topics"], [])
        self.assertEqual(result["topic_status"]["current_symptoms"], "clarification_needed")
        self.assertEqual(state["medication_status"], "taken")
        self.assertEqual(state["symptoms"], ["头晕"])
        model.assert_called_once()

    def test_historical_prefix_does_not_change_current_single_question(self):
        state = self.state("medication_taken", "之前报告过漏服。今天是否按医生安排服用了降压药？")
        result, model = self.extract("是的", state)
        self.assertEqual(result["medication_status"], "taken")
        self.assertEqual(result["topic_status"]["medication"], "answered")
        model.assert_not_called()

    def test_food_and_other_activity_do_not_create_medication_observations(self):
        for text in ("已经吃了", "还没吃", "还没到下班时间", "还没到吃饭时间", "已经吃了饭", "吃了早餐", "作息不规律", "不吃了"):
            with self.subTest(text=text):
                result, _ = self.extract(text, {"current_goal": "check_symptoms"})
                self.assertEqual(result["medication_status"], "unknown")
                self.assertEqual(result["medication_observations"], [])
        for text in ("还没到下班时间", "还没到吃饭时间", "已经吃了饭", "吃了早餐", "这个蛋糕不吃了"):
            with self.subTest(contextual_text=text):
                result, _ = self.extract(text, self.state("timing", medication_status="not_taken_yet"))
                self.assertEqual(result["medication_status"], "unknown")
                self.assertEqual(result["medication_observations"], [])
        self.assertEqual(extractor.detect_medication_status("还没到下班时间"), "unknown")
        self.assertEqual(extractor.detect_medication_status("按时吃了"), "taken")
        explicit, _ = self.extract("今天已经吃了降压药", {"current_goal": "check_symptoms"})
        self.assertEqual(explicit["medication_status"], "taken")

    def test_nonpunctuated_contrast_does_not_hide_danger(self):
        for text in ("没有胸痛只有呼吸困难", "没有胸痛只是喘不上气", "没有头晕现在呼吸困难", "没有胸痛有呼吸困难"):
            with self.subTest(text=text):
                result, model = self.extract(text)
                self.assertIn("呼吸困难", result["symptoms"])
                self.assertNotIn("呼吸困难", result["denied_symptoms"])
                dangers = [label for clause in extractor.split_answer_clauses(text) for label in detect_danger_signs(clause)]
                self.assertIn("呼吸困难", dangers)
                model.assert_not_called()

    def test_last_question_fallback_records_the_actual_symptom_list(self):
        result, model = self.extract("都没有", {"current_goal": "check_symptoms", "last_agent_question": "有没有头晕、胸痛或者呼吸困难？"})
        self.assertEqual(result["denied_symptoms"], ["头晕", "胸痛", "呼吸困难"])
        self.assertEqual(result["denied_red_flags"], ["胸痛", "呼吸困难"])
        model.assert_not_called()

    def test_grounded_model_bp_accepts_only_labeled_original_numbers(self):
        for text in ("这次测量的高压是150，低压是90", "收缩压150舒张压90"):
            with self.subTest(text=text):
                state = self.state("bp", "本次血压是多少？", "collect_bp")
                state["question_context"]["topic"] = "bp"
                model = Mock(return_value={"bp_systolic": 150, "bp_diastolic": 90, "evidence": {"bp": text}})
                result, model = self.extract(text, state, model)
                self.assertEqual((result["bp_systolic"], result["bp_diastolic"]), (150, 90))
                self.assertEqual(result["topic_status"]["bp"], "answered")
                model.assert_called_once()

    def test_grounded_model_bp_rejects_ambiguous_forged_or_unlabeled_numbers(self):
        cases = [
            ("不确定高压150低压90", "高压150低压90"),
            ("并不是高压150低压90", "高压150低压90"),
            ("最近觉得不舒服", "高压150低压90"),
            ("这两个数字是150和90", "这两个数字是150和90"),
            ("高压160低压95", "高压160低压95"),
        ]
        for text, evidence in cases:
            with self.subTest(text=text):
                state = self.state("bp", "本次血压是多少？", "collect_bp")
                state["question_context"]["topic"] = "bp"
                model = Mock(return_value={"bp_systolic": 150, "bp_diastolic": 90, "evidence": {"bp": evidence}})
                result, model = self.extract(text, state, model)
                self.assertEqual((result["bp_systolic"], result["bp_diastolic"]), (None, None))
                self.assertEqual(result["topic_status"]["bp"], "clarification_needed")
                self.assertNotIn("bp", result["answered_topics"])

    def test_model_cannot_use_truncated_evidence_to_invent_medication_or_danger(self):
        text = "不确定有没有服药，也记不清有没有胸痛"
        model = Mock(return_value={
            "medication_status": "taken", "symptoms": ["胸痛"], "denied_symptoms": ["呼吸困难"],
            "answered_topics": ["medication", "current_symptoms", "red_flags"],
            "evidence": {"medication": "服药", "symptoms": {"胸痛": "胸痛"}, "denied_symptoms": {"呼吸困难": "没有"}},
        })
        result, model = self.extract(text, self.state(), model)
        self.assertEqual(result["medication_status"], "unknown")
        self.assertEqual(result["symptoms"], [])
        self.assertEqual(result["denied_symptoms"], [])
        self.assertEqual(result["denied_red_flags"], [])
        self.assertEqual(result["answered_topics"], [])
        model.assert_called_once()

    def test_specific_timing_answer_resolves_later_medication_details_question(self):
        state = self.state("medication_details", "医生原来安排的服药时间是什么时候？", "clarify_medication_timing", medication_status="not_taken_yet")
        result, model = self.extract("已经超过原定服药时间了", state)
        self.assertEqual(result["medication_status"], "not_taken_yet")
        self.assertEqual(result["medication_timing"], "overdue")
        self.assertEqual(result["topic_status"]["medication"], "answered")
        self.assertEqual(result["medication_observations"][0]["date"], "2026-10-11")
        model.assert_not_called()
        future, _ = self.extract("明天还没到服药时间", state)
        self.assertEqual(future["medication_status"], "unknown")
        self.assertNotIn("medication", future["answered_topics"])
        known_taken, _ = self.extract("已经超过原定服药时间了", self.state("medication_details", goal="clarify_medication_timing", medication_status="taken"))
        self.assertEqual(known_taken["medication_status"], "unknown")
        self.assertNotIn("medication", known_taken["answered_topics"])

    def test_symptom_denial_clause_survives_other_information_in_same_answer(self):
        state = self.state("symptoms", "现在有没有头晕、胸痛或呼吸困难？", "check_symptoms")
        state["question_context"].update(topic="current_symptoms", asked_symptoms=["头晕", "胸痛", "呼吸困难"], asked_red_flags=["胸痛", "呼吸困难"])
        for text in ("都没有不舒服，已经吃了", "都没有不舒服，药还没吃"):
            with self.subTest(text=text):
                result, model = self.extract(text, state)
                self.assertEqual(result["denied_symptoms"], ["头晕", "胸痛", "呼吸困难"])
                self.assertEqual(result["denied_red_flags"], ["胸痛", "呼吸困难"])
                self.assertEqual(result["topic_status"]["current_symptoms"], "answered")
                model.assert_not_called()
        danger, model = self.extract("都没有不舒服，不过现在喘不上气", state)
        self.assertEqual(danger["symptoms"], ["呼吸困难"])
        self.assertNotIn("呼吸困难", danger["denied_symptoms"])
        self.assertNotIn("呼吸困难", danger["denied_red_flags"])
        model.assert_not_called()
        medication, _ = self.extract("没有呢，药还没吃", self.state())
        self.assertNotIn("current_symptoms", medication["answered_topics"])
        self.assertEqual(medication["denied_symptoms"], [])
        off_topic, _ = self.extract("都没有不舒服，药还没吃", self.state())
        self.assertEqual(off_topic["topic_status"]["current_symptoms"], "answered")
        self.assertEqual(off_topic["denied_red_flags"], [])

    def test_not_taken_and_timing_in_one_message_merge_only_same_date(self):
        for text, expected_timing in (("还没吃，已经超过原定服药时间了", "overdue"), ("还没吃，还没到规定服药时间", "not_due")):
            with self.subTest(text=text):
                result, model = self.extract(text, self.state())
                self.assertEqual(result["medication_status"], "not_taken_yet")
                self.assertEqual(result["medication_timing"], expected_timing)
                self.assertEqual(result["topic_status"]["medication"], "answered")
                self.assertTrue(all(item["date"] == "2026-10-11" for item in result["medication_observations"]))
                model.assert_not_called()
        for text in ("还没吃，昨天已经超过原定服药时间了", "还没吃，明天还没到规定服药时间", "还没吃，不确定是否已经超过原定服药时间"):
            with self.subTest(text=text):
                result, _ = self.extract(text, self.state())
                self.assertEqual(result["medication_status"], "not_taken_yet")
                self.assertEqual(result["medication_timing"], "unknown")
                self.assertEqual(result["topic_status"]["medication"], "clarification_needed")
        explicit, _ = self.extract("药还没吃，已经超过原定时间了", {"current_goal": "check_symptoms"})
        self.assertEqual(explicit["medication_status"], "not_taken_yet")
        self.assertEqual(explicit["medication_timing"], "overdue")

    def test_last_question_fallback_understands_adherence_without_explicit_kind(self):
        state = {"current_goal": "check_medication", "last_agent_question": "今天有没有漏服或者自行停药？"}
        result, model = self.extract("没有", state)
        self.assertEqual(result["medication_status"], "unknown")
        self.assertEqual(result["medication_adherence"], "no_missed_or_stopped")
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()

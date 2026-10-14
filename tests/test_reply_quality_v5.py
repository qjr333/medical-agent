"""Communication quality regressions using mock models, never the patient DB."""

import unittest

from agent import llm_v4
from tests.test_llm_autonomy import Model, ask, patient_state, valid_review


class ReplyQualityV5Tests(unittest.TestCase):
    def run_agent(self, state, model):
        return llm_v4.run_agent_v4(state, call_json_model=model, enable_v50=True)

    def plans(self, model):
        return [payload for payload in model.payloads if "trigger_event" in payload]

    def test_first_turn_double_greeting_gets_feedback_and_regenerates(self):
        state = patient_state()
        state["latest_patient_message"] = ""
        model = Model([
            ask("您好，今天测过血压了吗？", topic="bp", reply="您好，我是银龄守护随访助手。"),
            ask("今天测过血压了吗？", topic="bp", reply="您好，我是银龄守护随访助手。"),
        ])
        response = self.run_agent(state, model)
        self.assertEqual(response["message"].count("您好"), 1)
        self.assertEqual(state["last_agent_question"], "今天测过血压了吗？")
        feedback = self.plans(model)[1]["planner_validation_feedback"]
        self.assertEqual(feedback["violations"][0]["code"], "communication_repeat")
        self.assertEqual(len([p for p in model.payloads if "proposal" in p]), 1)

    def test_confirmation_repeated_between_fields_is_not_published(self):
        state = patient_state()
        state.update(medication_status="taken", latest_patient_message="今天按时吃药了")
        model = Model([
            ask("您今天已按时服药。现在有不舒服吗？", reply="您今天已按时服药，我记下了。"),
            ask("现在有不舒服吗？", reply="今天按时服药的情况已记下。"),
        ])
        response = self.run_agent(state, model)
        self.assertEqual(response["message"], "今天按时服药的情况已记下。\n\n现在有不舒服吗？")
        self.assertEqual(self.plans(model)[1]["planner_validation_feedback"]["violations"][0]["code"],
                         "communication_repeat")

    def test_same_question_in_reply_and_question_is_regenerated(self):
        model = Model([ask(reply="头晕是刚出现的，还是之前就有？"), ask(reply="您说的头晕我记下了。")])
        response = self.run_agent(patient_state(), model)
        self.assertEqual(response["message"].count("头晕是刚出现的"), 1)

    def test_semantic_duplicate_confirmation_gets_model_feedback(self):
        duplicate = valid_review(valid=False, violations=[{
            "code": "communication_repeat", "evidence": "用药情况已记录；已经记下您按时服药",
            "reason": "两个字段意思重复，question直接询问新的症状信息",
        }])
        model = Model([
            ask("已经记下您按时服药，现在感觉怎么样？", reply="用药情况已记录。"),
            ask("现在感觉怎么样？", reply="用药情况已记录。"),
        ], [duplicate, valid_review()])
        response = self.run_agent(patient_state(), model)
        self.assertNotIn("已经记下", response["message"])
        self.assertIn("communication_repeat", str(self.plans(model)[1]["planner_validation_feedback"]))

    def test_unnecessary_full_recap_is_returned_to_model_for_regeneration(self):
        unnecessary = valid_review(valid=False, violations=[{
            "code": "unnecessary_recap", "reason": "本轮只需要核实新症状，不要再次罗列全部已确认记录",
        }])
        model = Model([
            ask(reply="今天的血压、按时服药和之前的症状情况都已经逐项整理，再为您完整重述一遍。"),
            ask(reply="我们再核实一下新出现的头晕。"),
        ], [unnecessary, valid_review()])
        response = self.run_agent(patient_state(), model)
        self.assertNotIn("完整重述", response["message"])
        self.assertIn("unnecessary_recap", str(self.plans(model)[1]["planner_validation_feedback"]))

    def test_necessary_answer_to_patient_question_is_preserved(self):
        state = patient_state()
        state.update(latest_patient_message="刚才那句话没听清，您说的是什么？",
                     v41_pending_patient_question="刚才那句话没听清，您说的是什么？")
        explanation = "我是想核实头晕发生的时间。您可以只告诉我刚开始还是持续了一段时间，不需要一次回答所有情况。"
        question = "头晕是刚开始的吗？"
        response = self.run_agent(state, Model([ask(question, reply=explanation)]))
        self.assertEqual(response["message"], explanation + "\n\n" + question)
        self.assertEqual(state["v41_pending_patient_question"], "")

    def test_model_can_follow_new_information_on_completed_topic(self):
        state = patient_state()
        state["coverage"]["current_symptoms"] = True
        state["topic_status"]["current_symptoms"] = "answered"
        question = "您说现在头晕更明显了，是站起来时加重吗？"
        response = self.run_agent(state, Model([ask(question, reply="头晕的变化已记下。")]))
        self.assertIn(question, response["message"])
        self.assertEqual(state["last_agent_question"], question)
        self.assertEqual(state["question_context"]["topic"], "current_symptoms")


if __name__ == "__main__":
    unittest.main()

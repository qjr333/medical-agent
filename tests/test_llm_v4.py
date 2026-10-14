"""LLM V4 tests, with a fake model to avoid charges/unstable external network."""
import unittest
from agent.followup_agent import extract_patient_information, run_followup_agent
from agent.state_manager import merge_user_patch


def state_with_question(question, flags=None):
    return {
        'patient_id': 'T001', 'age': 72, 'conditions': ['高血压'],
        'trigger_type': 'scheduled', 'followup_reason': '例行随访',
        'bp_systolic': 146, 'bp_diastolic': 88, 'bp_history': [], 'bp_trend': 'unknown',
        'medication_status': 'unknown', 'medication_timing': 'unknown',
        'symptoms': [], 'denied_symptoms': [], 'red_flags': [], 'denied_red_flags': [],
        'risk_level': 'normal', 'trace': [], 'patient_memory': {}, 'memory_context': '',
        'current_goal': 'check_symptoms', 'last_agent_question': question,
        'question_context': {'question': question, 'topic': 'current_symptoms', 'kind': 'symptoms',
                             'goal': 'check_symptoms', 'asked_red_flags': flags or [],
                             'asked_symptoms': []},
        'coverage': {'bp': True, 'medication': False, 'current_symptoms': False,
                     'red_flags': False, 'measurement_context': False},
        'topic_status': {'bp': 'answered', 'medication': 'unanswered',
                         'current_symptoms': 'unanswered', 'red_flags': 'unanswered',
                         'measurement_context': 'unanswered'},
        'followup_status': 'active',
    }


class LLMV4Test(unittest.TestCase):
    def test_reports_chest_pain_and_escalates(self):
        state = state_with_question('您有没有胸痛？', ['胸痛'])
        def model(**kwargs):
            return {'symptoms': [{'name': '胸痛', 'evidence': '胸痛'}],
                    'red_flags': [{'name': '胸痛', 'evidence': '胸痛'}],
                    'answer_status': 'answered'}
        patch = extract_patient_information('胸痛', state, call_json_model=model)
        self.assertIn('胸痛', patch['red_flags'])
        merge_user_patch(state, patch, '胸痛')
        result = run_followup_agent(state, call_json_model=lambda **kw: {'action': 'ASK_USER', 'question': '您好？'})
        self.assertEqual(result['state']['followup_status'], 'escalated')

    def test_all_refers_to_last_asked_red_flags(self):
        state = state_with_question('有没有胸痛和呼吸困难？', ['胸痛', '呼吸困难'])
        patch = extract_patient_information('都有', state, call_json_model=lambda **kwargs: {'answer_status': 'answered'})
        self.assertEqual(set(patch['red_flags']), {'胸痛', '呼吸困难'})

    def test_denied_chest_pain_is_not_positive(self):
        state = state_with_question('您有胸痛吗？', ['胸痛'])
        patch = extract_patient_information('没有胸痛', state, call_json_model=lambda **kwargs: {
            'red_flags': [{'name':'胸痛','evidence':'胸痛'}],
            'denied_red_flags': [{'name':'胸痛','evidence':'没有胸痛'}],
            'answer_status': 'answered'})
        self.assertNotIn('胸痛', patch['red_flags'])

    def test_dynamic_question_is_from_model(self):
        state = state_with_question('', [])
        state['last_agent_question'] = ''
        state['question_context'] = {}
        def model(**kwargs):
            return {'action': 'ASK_USER', 'goal': 'ask_medication', 'topic': 'medication',
                    'question': '今天的降压药您服用了吗？', 'reason': '需要了解今天的服药情况'}
        result = run_followup_agent(state, call_json_model=model)
        self.assertEqual(result['state']['followup_status'], 'waiting_for_user')
        self.assertIn('今天的降压药', result['message'])
        self.assertTrue(any(ev['event'] == 'LLM_DECISION' for ev in state['trace']))

    def test_unable_to_answer_becomes_unable_not_diagnosis(self):
        state = state_with_question('医生原来安排几点服药？')
        state['question_context'].update(topic='medication', kind='medication_details')
        patch = extract_patient_information('记不清', state, call_json_model=lambda **kw: {
            'answer_status': 'unable', 'medication_status': 'unknown'})
        self.assertEqual(patch['answer_status'], 'unable')
        self.assertEqual(patch['medication_status'], 'unknown')

    def test_rag_is_a_planner_selected_tool(self):
        from unittest.mock import patch
        state = state_with_question('为什么要记录家庭血压？')
        state['last_agent_question'] = ''
        state['question_context'] = {}
        calls = {'n': 0}
        def model(**kw):
            calls['n'] += 1
            if calls['n'] == 1:
                return {'action': 'RETRIEVE_MEDICAL_GUIDANCE',
                        'query': '高血压家庭血压监测意义', 'reason': '患者询问医学知识'}
            return {'action': 'ASK_USER', 'topic': 'current_symptoms',
                    'question': '最近测量血压时有没有不舒服？',
                    'reply': '家庭血压记录能帮助了解血压变化；具体建议需结合医学证据。',
                    'reason': '结合证据回答并继续随访'}
        def retrieve(st, q):
            st['retrieved_guidance'] = [{'text':'家庭血压监测可辅助管理','source_title':'测试指南'}]
            return {'status':'ok', 'results':st['retrieved_guidance']}
        with patch('agent.followup_agent.tool_retrieve_medical_guidance', retrieve):
            result = run_followup_agent(state, call_json_model=model)
        self.assertEqual(calls['n'], 2)
        self.assertIn('家庭血压', result['message'])
        self.assertTrue(any(x['event'] == 'V4_TOOL_CALL' for x in state['trace']))

    def test_model_failure_is_visible_not_silent_rule_answer(self):
        state = state_with_question('', [])
        state['last_agent_question'] = ''
        state['question_context'] = {}
        result = run_followup_agent(state, call_json_model=lambda **kwargs: {})
        self.assertEqual(result['state']['followup_status'], 'interrupted')
        self.assertIn('模型暂时不可用', result['message']) if '模型暂时不可用' in result['message'] else self.assertIn('智能随访模型暂时不可用', result['message'])


if __name__ == '__main__':
    unittest.main()

"""Full V4.2 API flows with temporary SQLite and actual registered backend paths."""
import contextlib
import copy
import importlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from collections import deque
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.utils import today_local_date
from database import collaboration, connection, repository
from medical_tools.safety import DANGER_PATTERNS


class ScriptedModel:
    def __init__(self):
        self.plans = deque()
        self.outcome = 'completed'
        self.summary_failure = False
        self.completion_failure = False
        self.objective_evidence = None
        self.payloads = []

    def __call__(self, *, system_prompt, user_prompt, temperature):
        payload = json.loads(user_prompt)
        self.payloads.append(payload)
        if 'evaluation_context' in payload:
            if self.completion_failure:
                return {}
            value = {'outcome': self.outcome, 'reason': '按当前事实核实任务',
                     'pending_items': [], 'next_goal': ''}
            if self.objective_evidence:
                value['objective_evidence'] = {'monitoring_plan': self.objective_evidence}
            return value
        if 'facts' in payload:
            if self.summary_failure:
                return {}
            sections = ('summary_fact_ids', 'doctor_attention_fact_ids', 'family_assistance_fact_ids')
            return {section: [fact['fact_id'] for fact in payload['facts']
                              if section in fact['allowed_sections']] for section in sections}
        if 'trigger_event' in payload:
            return self.plans.popleft() if self.plans else {
                'action': 'ASK_USER', 'topic': 'additional_information',
                'question': '请补充您当前想说明的情况。'}
        return {'answer_status': 'answered'}


def finish():
    return {'action': 'FINISH', 'reason': '模拟模型申请完成', 'reply': '本次记录已整理。'}


class V42APIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': '', 'RAG_DISABLE_EMBEDDING': '1'}), contextlib.redirect_stdout(io.StringIO()):
            cls.backend = importlib.import_module('app')

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.dict(os.environ, {'LLM_AGENT_V42': '1', 'LLM_AGENT_V4': '0'}))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(connection, 'DB_PATH', directory / 'v42.db'))
        self.stack.enter_context(patch.object(connection, 'DATA_DIR', directory))
        self.stack.enter_context(patch.object(self.backend, 'DEMO_PATIENT', {
            'patient_id': 'v42-p', 'patient_name': '模拟患者', 'age': 72, 'conditions': ['高血压']}))
        self.model = ScriptedModel()
        self.model_mock = self.stack.enter_context(patch.object(self.backend, 'call_json_model', side_effect=self.model))
        repository.sessions.clear()
        self.addCleanup(repository.sessions.clear)
        self.http = self.stack.enter_context(TestClient(self.backend.app))
        repository.seed_demo_patient({'patient_id': 'other-p', 'patient_name': '另一模拟患者', 'age': 70, 'conditions': []}, [])
        self.tokens = {}
        for user_id, role, patient in (('d1', 'doctor', 'v42-p'), ('d2', 'doctor', 'v42-p'),
                                       ('f1', 'family', 'v42-p'), ('d-other', 'doctor', 'other-p'),
                                       ('f-denied', 'family', None)):
            collaboration.provision_user(user_id, role)
            self.tokens[user_id] = collaboration.issue_token(user_id)
            if patient:
                collaboration.link_patient(user_id, patient)

    def headers(self, name):
        return {'Authorization': 'Bearer ' + self.tokens[name]}

    def seed(self, complete=True, kind='scheduled', **changes):
        state = self.backend.create_patient_state(kind)
        state.update(bp_systolic=135, bp_diastolic=85, medication_status='taken',
                     medication_date=today_local_date(), medication_detail='按原方案服药',
                     symptom_status='denied', red_flag_status='denied',
                     denied_symptoms=['头晕'], denied_red_flags=list(DANGER_PATTERNS),
                     latest_patient_message='今天135/85，按时服药，没有不适',
                     measurement_context='静坐休息5分钟后测量')
        state['coverage'] = dict.fromkeys(state['coverage'], complete)
        state['topic_status'] = dict.fromkeys(state['coverage'], 'answered' if complete else 'unanswered')
        state['trace'] = [{'event': 'USER_MESSAGE', 'detail': {'message': state['latest_patient_message']}}]
        state.update(changes)
        repository.persist_session('test-session', state)
        return state

    def chat(self, text='继续'):
        response = self.http.post('/api/chat', json={'session_id': 'test-session', 'message': text})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def doctor_reports(self):
        response = self.http.get('/api/reports', headers=self.headers('d1'))
        self.assertEqual(response.status_code, 200)
        return response.json()['items']

    def test_normal_completion_report_and_role_specific_messages(self):
        self.seed()
        self.model.plans.append(finish())
        body = self.chat()
        self.assertTrue({'message', 'session_id', 'state', 'trace', 'report'} <= body.keys())
        self.assertEqual(body['state']['followup_status'], 'completed')
        self.assertNotIn('doctor_view', body['report'])
        self.assertNotIn('doctor_view', body['state']['report'])
        rows = self.doctor_reports()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['report']['schema_version'], '4.2')
        self.assertEqual(rows[0]['report']['ai_summary']['status'], 'success')
        self.assertTrue(rows[0]['doctor_view'])
        family = self.http.get('/api/reports', headers=self.headers('f1')).json()['items'][0]
        self.assertTrue(family['family_view'])
        self.assertNotIn('doctor_view', family)
        self.assertNotIn('medication_detail', json.dumps(family))
        self.assertEqual(len(self.http.get('/api/messages', headers=self.headers('d1')).json()['items']), 1)
        self.assertEqual(len(self.http.get('/api/messages', headers=self.headers('f1')).json()['items']), 1)
        purposes = [e['detail']['purpose'] for e in repository.read_session('test-session')['trace'] if e['event'] == 'LLM_CALL']
        self.assertIn('COMPLETE', purposes)
        self.assertIn('REPORT_SUMMARY', purposes)
        self.assertEqual(self.http.post('/api/chat', json={'session_id': 'test-session', 'message': '继续'}).status_code, 409)

    def test_missing_information_blocks_early_finish(self):
        state = self.seed(complete=False, medication_status='unknown', symptom_status='unknown', red_flag_status='unknown')
        self.model.plans.extend([finish(), {'action': 'ASK_USER', 'topic': 'medication', 'question': '今天服药了吗？'}])
        body = self.chat('我记得刚才的血压')
        self.assertEqual(body['state']['followup_status'], 'waiting_for_user')
        self.assertFalse(self.doctor_reports())
        self.assertIn('medication', body['state']['completion_evaluation']['pending_items'])
        self.assertEqual(body['state']['medication_status'], 'unknown')
        self.assertFalse(body['state'].get('report'))

    def test_unable_information_becomes_partial_without_reasking(self):
        state = self.seed()
        state['coverage']['medication'] = False
        state['topic_status']['medication'] = 'clarification_needed'
        state['medication_status'] = 'unknown'
        state['deferred_topics'] = {'medication': 'unable'}
        repository.persist_session('test-session', state)
        self.model.plans.extend([{'action': 'ASK_USER', 'topic': 'medication', 'question': '今天服药了吗？'}, finish()])
        self.model.outcome = 'partial'
        body = self.chat('继续')
        self.assertEqual(body['state']['followup_status'], 'partially_completed')
        self.assertIn('不能据此排除风险', body['message'])
        report = self.doctor_reports()[0]['report']
        self.assertIn('medication', report['unconfirmed_items'])
        self.assertEqual(report['medication']['status'], 'unknown')
        self.assertTrue(any(e['event'] == 'DEFERRED_QUESTION_SKIPPED' for e in body['trace']))
        repository.sessions.clear()
        self.assertEqual(repository.read_session('test-session')['followup_status'], 'partially_completed')
        self.assertEqual(self.http.post('/api/chat', json={'session_id': 'test-session', 'message': '继续'}).status_code, 409)

    def test_high_bp_requires_actual_measurement_context(self):
        state = self.seed(kind='high_bp')
        state['measurement_context'] = ''
        repository.persist_session('test-session', state)
        self.model.plans.extend([finish(), {'action': 'ASK_USER', 'topic': 'measurement_context', 'question': '测量前休息了吗？'}])
        first = self.chat()
        self.assertEqual(first['state']['followup_status'], 'waiting_for_user')
        self.assertIn('measurement_context', first['state']['completion_evaluation']['pending_items'])
        self.model.plans.append(finish())
        second = self.chat('安静休息五分钟后测量')
        self.assertEqual(second['state']['measurement_context'], '安静休息五分钟后测量')
        self.assertEqual(second['state']['followup_status'], 'completed')

    def test_missed_measurement_finishes_with_real_plan_without_fabricating_bp(self):
        state = self.seed(kind='missed_measurement', bp_systolic=None, bp_diastolic=None)
        state['coverage']['bp'] = False
        state['topic_status']['bp'] = 'unanswered'
        repository.persist_session('test-session', state)
        self.model.objective_evidence = '我明天会测量血压并上传记录'
        self.model.plans.append(finish())
        body = self.chat('我明天会测量血压并上传记录')
        self.assertEqual(body['state']['followup_status'], 'completed')
        self.assertIsNone(body['state']['bp_systolic'])
        self.assertTrue(body['state']['coverage']['monitoring_plan'])
        self.assertIsNone(self.doctor_reports()[0]['report']['blood_pressure']['systolic'])

    def test_semantic_measurement_answer_is_grounded_and_persisted(self):
        from agent.followup_agent import extract_patient_information
        from agent.state_manager import merge_user_patch
        from medical_tools.guidance import execution_fingerprint
        state = self.seed(kind='high_bp')
        state.update(agent_version='V4.2', measurement_context='')
        text = '我是刚从菜场回来就量的'
        before = execution_fingerprint(state, 'RETRIEVE_MEDICAL_GUIDANCE', '测量条件')
        patch = extract_patient_information(text, state, call_json_model=lambda **kw: {
            'answer_status': 'answered', 'measurement_context': text, 'measurement_context_evidence': text})
        merge_user_patch(state, patch, text)
        self.assertEqual(state['measurement_context'], text)
        self.assertTrue(state['coverage']['measurement_context'])
        self.assertNotEqual(before, execution_fingerprint(state, 'RETRIEVE_MEDICAL_GUIDANCE', '测量条件'))
        invalid = extract_patient_information('不知道', state, call_json_model=lambda **kw: {
            'answer_status': 'unable', 'measurement_context': '已静坐五分钟', 'measurement_context_evidence': '已静坐五分钟'})
        self.assertNotEqual(invalid.get('measurement_context'), '已静坐五分钟')

    def test_danger_response_is_sent_before_report_generation_and_no_llm(self):
        self.seed(complete=False)
        sent = Event()
        underlying = self.backend.app
        class Capture:
            async def __call__(self, scope, receive, send):
                async def wrapped(message):
                    await send(message)
                    if message['type'] == 'http.response.body' and not message.get('more_body'):
                        sent.set()
                await underlying(scope, receive, wrapped)
        original = self.backend._finalize_urgent_report
        def background(session_id):
            self.assertTrue(sent.is_set(), 'urgent body must be sent before report work')
            original(session_id)
        with patch.object(self.backend, '_finalize_urgent_report', side_effect=background), TestClient(Capture()) as http:
            response = http.post('/api/chat', json={'session_id': 'test-session', 'message': '我现在胸痛，还有点喘不上气'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['state']['followup_status'], 'escalated')
        self.assertIn('急救', response.json()['message'])
        self.model_mock.assert_not_called()
        rows = self.doctor_reports()
        self.assertEqual(rows[0]['report']['completion_outcome'], 'escalated')
        self.assertEqual(rows[0]['report']['ai_summary']['status'], 'not_requested')
        self.assertTrue(self.http.get('/api/messages', headers=self.headers('f1')).json()['items'])

    def test_pending_emergency_report_can_be_retried_idempotently(self):
        self.seed(complete=False)
        with patch.object(self.backend, '_finalize_urgent_report', return_value=None):
            body = self.chat('我现在胸痛')
        self.assertEqual(body['state']['report_generation']['status'], 'pending')
        self.assertFalse(self.doctor_reports())
        for _ in range(2):
            result = self.http.post('/api/reports/session/test-session/generate', headers=self.headers('d1'))
            self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(len(self.doctor_reports()), 1)
        self.assertEqual(len(self.http.get('/api/messages', headers=self.headers('f1')).json()['items']), 1)
        self.model_mock.assert_not_called()

    def test_database_read_failure_still_preempts_plain_chest_pain(self):
        self.seed(complete=False)
        with patch.object(self.backend, 'read_session', side_effect=sqlite3.OperationalError('offline')):
            response = self.http.post('/api/chat', json={'session_id': 'test-session', 'message': '我现在胸痛'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('急救', response.json()['detail'])
        self.assertEqual(repository.sessions['test-session']['followup_status'], 'escalated')
        self.model_mock.assert_not_called()

    def test_doctor_handling_family_read_and_repeated_push_persist(self):
        self.seed()
        self.model.plans.append(finish())
        self.chat()
        report_id = self.doctor_reports()[0]['id']
        handling = self.http.patch(f'/api/reports/{report_id}/handling', headers=self.headers('d1'),
                                  json={'status': 'followup_required', 'notes': '请核对测量时间'})
        self.assertEqual(handling.status_code, 200)
        self.assertEqual(handling.json()['updated_by'], 'd1')
        self.assertEqual(self.http.get(f'/api/reports/{report_id}', headers=self.headers('d2')).json()['notes'], '')
        msg = self.http.get('/api/messages', headers=self.headers('f1')).json()['items'][0]
        read = self.http.post(f"/api/messages/{msg['id']}/read", headers=self.headers('f1')).json()
        self.assertEqual(read['status'], 'read')
        for _ in range(2):
            pushed = self.http.post(f'/api/reports/{report_id}/push', headers=self.headers('d1'))
            self.assertEqual(pushed.status_code, 200)
            self.assertEqual(pushed.json()['created_messages'], 0)
            generated = self.http.post('/api/reports/session/test-session/generate', headers=self.headers('d1'))
            self.assertEqual(generated.status_code, 200)
        repository.sessions.clear()
        fresh = self.http.get(f"/api/messages/{msg['id']}", headers=self.headers('f1')).json()
        self.assertEqual(fresh['read_at'], read['read_at'])
        self.assertEqual(self.doctor_reports()[0]['status'], 'followup_required')
        self.assertEqual(self.doctor_reports()[0]['notes'], '请核对测量时间')
        self.assertEqual(len(self.doctor_reports()), 1)
        family = self.http.get(f'/api/reports/{report_id}', headers=self.headers('f1')).json()
        self.assertNotIn('notes', family)
        self.assertNotIn('请核对测量时间', json.dumps(family, ensure_ascii=False))

    def test_untrusted_identity_no_grant_revocation_and_debug_bypass(self):
        self.seed()
        self.model.plans.append(finish())
        self.chat()
        rid = self.doctor_reports()[0]['id']
        self.assertEqual(self.http.get('/api/reports', headers={'X-User-ID': 'd1', 'X-Role': 'doctor'}).status_code, 401)
        for user in ('f-denied', 'd-other'):
            self.assertEqual(self.http.get(f'/api/reports/{rid}', headers=self.headers(user)).status_code, 403)
        self.assertEqual(self.http.patch(f'/api/reports/{rid}/handling', headers=self.headers('f1'),
                                        json={'status': 'closed', 'user_id': 'd1'}).status_code, 422)
        self.assertEqual(self.http.patch(f'/api/reports/{rid}/handling', headers=self.headers('f1'),
                                        json={'status': 'closed'}).status_code, 403)
        self.assertEqual(self.http.get('/api/debug/session/test-session').status_code, 401)
        self.assertEqual(self.http.get('/api/debug/session/test-session', headers=self.headers('f1')).status_code, 403)
        self.assertEqual(self.http.get('/api/debug/session/test-session', headers=self.headers('d1')).status_code, 200)
        self.assertEqual(self.http.get('/api/patients/v42-p/history').status_code, 401)
        mid = self.http.get('/api/messages', headers=self.headers('f1')).json()['items'][0]['id']
        collaboration.link_patient('f1', 'v42-p', False)
        self.assertEqual(self.http.get(f'/api/reports/{rid}', headers=self.headers('f1')).status_code, 403)
        self.assertEqual(self.http.get(f'/api/messages/{mid}', headers=self.headers('f1')).status_code, 403)
        self.assertEqual(self.http.post(f'/api/messages/{mid}/read', headers=self.headers('f1')).status_code, 403)
        self.assertEqual(self.http.get('/api/messages', headers=self.headers('f1')).json()['items'], [])
        self.assertEqual(self.http.get(f'/api/messages/{mid}', headers=self.headers('d1')).status_code, 403)

    def test_model_completion_failure_has_no_report_or_fake_completion(self):
        self.seed()
        self.model.plans.append(finish())
        self.model.completion_failure = True
        body = self.chat()
        self.assertEqual(body['state']['followup_status'], 'interrupted')
        self.assertEqual(body['state']['completion_evaluation']['model_status'], 'failed')
        self.assertFalse(self.doctor_reports())
        self.assertFalse(body['state'].get('report'))

    def test_many_patient_turns_have_no_fixed_completion_limit(self):
        self.seed(complete=False)
        for index in range(12):
            self.model.plans.append({'action': 'ASK_USER', 'topic': 'additional_information',
                                     'question': f'请补充您想说明的第{index + 1}件事。'})
            body = self.chat(f'我还想说明第{index + 1}件事')
            self.assertEqual(body['state']['followup_status'], 'waiting_for_user')
        self.assertFalse(self.doctor_reports())

    def test_ai_summary_failure_is_explicit_and_fact_report_still_persists(self):
        self.seed()
        self.model.plans.append(finish())
        self.model.summary_failure = True
        body = self.chat()
        self.assertEqual(body['state']['followup_status'], 'completed')
        report = self.doctor_reports()[0]['report']
        self.assertEqual(report['ai_summary']['status'], 'failed')
        self.assertEqual(report['doctor_view']['summary_kind'], 'recorded_facts')
        self.assertIn('135/85', report['ai_summary']['summary'])


if __name__ == '__main__':
    unittest.main()

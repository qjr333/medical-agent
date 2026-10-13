"""V5 execution service shared by proactive tasks and isolated simulations."""
import copy
import json
from time import monotonic

from agent.llm_v4 import _call, extract_user_patch_v4, run_agent_v4
from agent.state_manager import add_trace, create_patient_state, merge_user_patch
from agent.reporting_v42 import build_report_v42
from database.connection import get_connection
from database import repository, tasks
from memory.patient_memory import refresh_patient_memory
from medical_tools.guidance import execution_fingerprint
from medical_tools.safety import valid_bp_measurement

TERMINAL = {"completed", "partially_completed", "escalated"}
REASONS = {"scheduled": "计划时间到达，开展例行随访。",
           "high_bp": "收到新的异常血压记录，核实症状、测量情况和用药。",
           "missed_measurement": "计划窗口内没有血压测量记录，了解原因并恢复监测。"}


class AgentRuntime:
    def __init__(self, model=None):
        self.model = model

    def model_call(self, state, purpose, prompt, payload):
        return _call(self.model, state, purpose, prompt, payload, deadline=monotonic() + 20)

    def start_task(self, task):
        sid = task["session_id"]
        with repository.session_lock(sid):
            state = repository.read_session(sid)
            if state is None:
                conn = get_connection()
                try:
                    row = conn.execute("SELECT * FROM patients WHERE patient_id=?", (task["patient_id"],)).fetchone()
                finally:
                    conn.close()
                if row is None:
                    raise LookupError("Patient does not exist")
                patient = dict(row)
                patient["conditions"] = json.loads(patient.pop("conditions_json"))
                trigger = task["trigger_type"]
                measurement = task.get("measurement") or {}
                config = {t: {"followup_reason": r, "bp_systolic": None, "bp_diastolic": None}
                          for t, r in REASONS.items()}
                if trigger == "high_bp":
                    config[trigger].update(bp_systolic=measurement["systolic"], bp_diastolic=measurement["diastolic"])
                state = create_patient_state(trigger, DEMO_PATIENT=patient, DEMO_BP_HISTORY=[], TRIGGER_CONFIGS=config)
                state.update(agent_version="V5.0", session_id=sid, proactive_task_id=task["task_id"])
                add_trace(state, "TRIGGER_ENGINE", {"trigger_type": trigger, "task_id": task["task_id"],
                          "event_id": task["event_id"], "external_delivery": "not_configured"})
                repository.persist_session(sid, state)
            if state.get("followup_status") in TERMINAL or state.get("followup_status") == "waiting_for_user":
                self.finalize(sid)
                return self.task_result(sid)
            self.turn(sid)
            return self.task_result(sid)

    def turn(self, sid, message=None, *, turn_index=None):
        with repository.session_lock(sid):
            stored = repository.read_session(sid)
            if stored is None:
                raise LookupError("Session does not exist")
            state = copy.deepcopy(stored)
            if turn_index is not None and state.get("simulation_turn_index", -1) >= turn_index:
                if state.get("simulation_turn_stage") == "completed" or state.get("followup_status") in TERMINAL:
                    self.finalize(sid)
                    return repository.read_session(sid)
                message = None
            if state.get("followup_status") in TERMINAL:
                if message is not None:
                    raise ValueError("Follow-up already ended")
                return state
            state["agent_version"] = "V5.0"
            if message is not None:
                message = message.strip()
                if not message:
                    raise ValueError("Patient message required")
                before = execution_fingerprint(state, "PATIENT_FACTS", query="")
                state["followup_status"] = "active"
                add_trace(state, "USER_MESSAGE", {"message": message})
                patch = extract_user_patch_v4(message, state, call_json_model=self.model, enable_v50=True)
                bp = None
                if valid_bp_measurement(patch.get("bp_systolic"), patch.get("bp_diastolic")):
                    bp = {"systolic": patch["bp_systolic"], "diastolic": patch["bp_diastolic"], "source": "patient_chat"}
                merge_user_patch(state, patch, message)
                if before != execution_fingerprint(state, "PATIENT_FACTS", query=""):
                    state.update(visit_advice=None, medication_guidance=None)
                med = [{"status": item["status"], "detail": item.get("detail") or message}
                       for item in patch.get("medication_observations", [])]
                add_trace(state, "STATE_UPDATE", {"coverage": state.get("coverage"), "patch": patch})
                if turn_index is not None:
                    state.update(simulation_turn_index=turn_index, simulation_turn_stage="input_saved")
                repository.persist_session(sid, state, new_bp=bp, new_medication=med)
            refresh_patient_memory(state, sid)
            response = run_agent_v4(state, call_json_model=self.model, enable_v50=True)
            state["last_agent_response"] = response["message"]
            add_trace(state, "AGENT_RESPONSE", {"message": response["message"], "followup_status": state["followup_status"]})
            repository.persist_session(sid, state)
            # Resume internally with the saved execution context, never a fabricated patient reply.
            for _ in range(3):
                if state.get("followup_status") != "processing":
                    break
                add_trace(state, "EXECUTION_RESUMED", {"automatic": True})
                response = run_agent_v4(state, call_json_model=self.model, enable_v50=True)
                state["last_agent_response"] = response["message"]
                add_trace(state, "AGENT_RESPONSE", {"message": response["message"], "followup_status": state["followup_status"]})
                repository.persist_session(sid, state)
            if state.get("followup_status") == "processing":
                state.update(followup_status="interrupted", interruption_reason="automatic_resume_exhausted")
                state["last_agent_response"] = "本次自动处理尚未完成，当前记录已保留。请稍后重试；不能据此认为医学评估已完成。"
                add_trace(state, "TASK_INTERRUPTED", {"reason": "automatic_resume_exhausted", "next_status": "interrupted"})
                repository.persist_session(sid, state)
            if turn_index is not None:
                state.update(simulation_turn_index=turn_index, simulation_turn_stage="completed")
                repository.persist_session(sid, state)
            # Emergency response has already been persisted before report work begins.
            self.finalize(sid)
            state = repository.read_session(sid)
            result = self.task_result(sid)
            tasks.sync_session_status(sid, result["status"], report_generated=result["report_generated"],
                                      station_messages=result["station_messages"])
            return state

    def finalize(self, sid):
        with repository.session_lock(sid):
            original = repository.read_session(sid)
            if not original or original.get("followup_status") not in TERMINAL:
                return original
            status = (original.get("report_generation") or {}).get("status")
            if status == "generated" and original.get("report"):
                # Re-persist repairs station delivery after grants or interrupted transactions.
                repository.persist_session(sid, original)
                return original
            state = copy.deepcopy(original)
            old_report = state.get("report")
            state["report"] = None
            callback = None if state.get("followup_status") == "escalated" else lambda purpose, prompt, payload: self.model_call(state, purpose, prompt, payload)
            report = build_report_v42(state, model_call=callback)
            if old_report:
                report["generated_at"] = old_report["generated_at"]
                report["doctor_view"]["generated_at"] = old_report["generated_at"]
            state["report_generation"] = {"status": "generated", "ai_status": report["ai_summary"]["status"]}
            add_trace(state, "REPORT_GENERATED", {"version": "5.0", "outcome": state["completion_outcome"],
                      "ai_status": report["ai_summary"]["status"]})
            repository.persist_session(sid, state)
            count = self.task_result(sid)["station_messages"]
            add_trace(state, "MESSAGE_DELIVERY", {"channel": "station", "recipient_count": count,
                      "status": "delivered" if count else "no_recipients", "external_delivery": "not_configured"})
            repository.persist_session(sid, state)
            return state

    @staticmethod
    def task_result(sid):
        state = repository.read_session(sid)
        conn = get_connection()
        try:
            count = conn.execute("SELECT COUNT(*) FROM report_messages m JOIN reports r ON r.id=m.report_id WHERE r.session_id=?", (sid,)).fetchone()[0]
        finally:
            conn.close()
        return {"session_id": sid, "status": "completed" if state["followup_status"] in TERMINAL else state["followup_status"],
                "followup_status": state["followup_status"], "message": state.get("last_agent_response", ""),
                "report_generated": bool(state.get("report")), "station_messages": count}

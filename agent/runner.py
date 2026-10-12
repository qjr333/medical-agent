"""Agent Loop V3：安全门禁、状态驱动决策、有限执行与可恢复工具失败。"""

from time import perf_counter
import re

from medical_tools.registry import get_tool_registry
from medical_tools.guidance import build_guidance_query, execution_fingerprint, tool_is_current
from medical_tools.safety import tool_assess_risk, HIGH_BP_SYS, HIGH_BP_DIA
from medical_tools.medication import tool_medication_guidance
from medical_tools.visit_advice import tool_generate_visit_advice
from .planner import select_next_action, select_next_decision
from .question_planner import generate_question, record_question
from .state_manager import add_trace, ensure_dialogue_state
from .reporting import build_finish_message, build_urgent_message, generate_report

MAX_AGENT_STEPS = 12


def _pending_tasks(state):
    required = state.get("required_topics") or ["bp", "medication", "current_symptoms", "red_flags"]
    if state.get("risk_level") == "attention" and "measurement_context" not in required:
        required = list(required) + ["measurement_context"]
    pending = [topic for topic in required if not state.get("coverage", {}).get(topic)
               or state.get("topic_status", {}).get(topic) == "clarification_needed"]
    if state.get("pending_question_followup"):
        pending.append("answer_medication_question")
    return pending


def _response(state, message, prefix=""):
    response = {"message": "\n\n".join(part for part in (prefix, message) if part),
                "state": state, "trace": state.get("trace", [])}
    if state.get("report") is not None:
        response["report"] = state["report"]
    if state.get("followup_status") == "escalated":
        response["alerts"] = state.get("alerts")
    return response


def _record_error(state, tool, error, *, recoverable, duration_ms=0):
    error_type = error if isinstance(error, str) else type(error).__name__
    entry = {"tool": tool, "error_type": error_type, "recoverable": recoverable}
    errors = state.setdefault("task_errors", [])
    if entry not in errors:
        errors.append(entry)
    del errors[:-8]
    add_trace(state, "TOOL_ERROR", {**entry, "duration_ms": round(duration_ms, 3),
                                   "next_status": "active" if recoverable else "interrupted"})


def _interrupt(state, reason, *, prefix="", event="TASK_INTERRUPTED"):
    state["followup_status"] = "interrupted"
    state["interruption_reason"] = reason
    if reason == "safety_assessment_failed":
        state["current_goal"] = "safety_assessment"
    elif not state.get("current_goal"):
        state["current_goal"] = "resume_processing"
    state["pending_tasks"] = _pending_tasks(state)
    if not state["pending_tasks"]:
        state["pending_tasks"] = [state.get("loop_state", {}).get("next_action")
                                  or state.get("last_action") or "finalize_followup"]
    add_trace(state, event, {"reason": reason, "goal": state.get("current_goal"),
                            "pending_tasks": state["pending_tasks"], "next_status": "interrupted"})
    return _response(state, "本次随访仍有信息或处理步骤未完成，已保留当前目标。"
                     "目前不能据此排除风险或调整处方药。可以稍后补充信息继续随访；"
                     "如有新的明显不适，请及时联系医疗专业人员。", prefix)


def run_agent(state: dict, *, call_json_model=None, tool_registry=None, max_steps=MAX_AGENT_STEPS):
    if state.get("followup_status") in {"completed", "escalated"}:
        message = (build_urgent_message(state) if state["followup_status"] == "escalated"
                   else build_finish_message(state))
        return _response(state, message)
    ensure_dialogue_state(state)
    state["report"] = None
    if "tool_execution" not in state:
        # 旧会话没有可验证的输入签名，不沿用旧医学建议或仅凭查询字符串复用证据。
        state.update(last_guidance_query="", last_medication_question="", retrieved_guidance=[],
                     knowledge_sources=[], medication_guidance=None, visit_advice=None)
        state.pop("guidance_fingerprint", None)
    state["followup_status"] = "active"
    state["interruption_reason"] = ""
    state.setdefault("tool_execution", {})
    state["actions_completed"] = list(dict.fromkeys(state.get("actions_completed") or []))
    # 可恢复失败只在下一次执行重试；本次循环仍用保守回退避免无限重试。
    for action, entry in state["tool_execution"].items():
        if entry.get("retryable_error"):
            entry["status"] = "retry"
            if action == "RETRIEVE_MEDICAL_GUIDANCE":
                state.pop("guidance_fingerprint", None)
    handled = state.get("handled_patient_question") or {}
    if handled.get("retryable_error"):
        state["handled_patient_question"] = {}
    question = state.get("latest_patient_question") or state.get("pending_patient_question") or ""
    followup = state.get("pending_question_followup") or {}
    followup_waiting = False
    if followup and question not in {"", followup.get("question")}:
        state["pending_question_followup"] = {}
        followup = {}
    if followup and state.get("question_context", {}).get("goal") == "collect_medication_details":
        raw = state.get("latest_patient_message") or ""
        detail = state.get("additional_information") or ""
        short_or_unclear = bool(re.fullmatch(r"[\s，。!?？！]*(?:没有|是的|有|对|嗯|不知道|不确定|记不清|不记得)[\s，。!?？！]*", raw))
        meaningful = (detail != followup.get("last_detail") and raw and not short_or_unclear
                      and state.get("latest_answer_status") not in {"refused", "unable"}
                      and raw != followup.get("question"))
        if meaningful:
            question = followup["question"]
            state["handled_patient_question"] = {}
            handled = {}
        else:
            followup_waiting = True
    state["latest_patient_question"] = question
    prefix = ""
    if (question and handled.get("question") == question and not handled.get("retryable_error")
            and handled.get("fingerprint") == execution_fingerprint(state, "ANSWER_PATIENT_QUESTION", question)):
        prefix = state.get("last_question_response") or ""
        state["latest_patient_question"] = ""
        state["pending_patient_question"] = ""

    model_failures = []
    def guarded_model(*args, **kwargs):
        from backend.model_client import model_failure_observer
        start = perf_counter()
        def observe(error):
            model_failures.append(type(error).__name__)
            _record_error(state, "DeepSeek", error, recoverable=True,
                          duration_ms=(perf_counter() - start) * 1000)
        observer_token = model_failure_observer.set(observe)
        try:
            if call_json_model is None:
                from backend.model_client import call_json_model as model
            else:
                model = call_json_model
            result = model(*args, **kwargs)
            return result if isinstance(result, dict) else {}
        except Exception as error:
            observe(error)
            return {}
        finally:
            model_failure_observer.reset(observer_token)
    tools = get_tool_registry(call_json_model=guarded_model) if tool_registry is None else tool_registry
    turn_actions = []
    max_steps = max(0, min(int(max_steps), 64))
    state["loop_state"] = {"max_steps": max_steps, "steps": 0, "reason": "running"}

    def execute(tool, *args):
        start = perf_counter()
        result = tools[tool](*args)
        if not isinstance(result, dict):
            raise ValueError("Invalid tool result")
        if tool == "ASSESS_RISK":
            level = result.get("risk_level")
            if level not in {"normal", "attention", "high"}:
                raise ValueError("Invalid safety result")
            state["risk_level"] = level
            state["risk_reasons"] = result.get("reasons") or []
        duration = (perf_counter() - start) * 1000
        summary = {key: result.get(key) for key in (
            "status", "risk_level", "trend", "record_count", "level", "needs_followup", "needs_professional_review"
        ) if key in result}
        if "results" in result:
            summary["result_count"] = len(result.get("results") or [])
        add_trace(state, "TOOL_RESULT", {"tool": tool, "status": "ok", "result_summary": summary,
                                         "duration_ms": round(duration, 3), "next_status": "active"})
        return result

    def assess(event="ASSESS_RISK"):
        start = perf_counter()
        try:
            result = execute("ASSESS_RISK", state)
            # 注入或异常工具不能遮蔽原确定性安全规则已经要求的升级。
            if (state.get("red_flags")
                    or isinstance(state.get("bp_systolic"), int) and state["bp_systolic"] >= HIGH_BP_SYS
                    or isinstance(state.get("bp_diastolic"), int) and state["bp_diastolic"] >= HIGH_BP_DIA):
                result = tool_assess_risk(state)
            add_trace(state, event, result)
            return True
        except Exception as error:
            _record_error(state, "ASSESS_RISK", error, recoverable=False,
                          duration_ms=(perf_counter() - start) * 1000)
            tool_assess_risk(state)
            return False

    def escalate():
        state["followup_status"] = "escalated"
        state["current_goal"] = "safety_escalation"
        state["last_action"] = "ESCALATE"
        add_trace(state, "ACTION_SELECTED", {"action": "ESCALATE", "goal": "safety_escalation",
                   "reason": "确定性安全规则要求优先升级", "priority": "critical", "required_tools": ["CREATE_ALERT"],
                   "basis": {"risk_level": state.get("risk_level"), "red_flags": state.get("red_flags")}, "duration_ms": 0})
        try:
            state["alerts"] = execute("CREATE_ALERT", state)
        except Exception as error:
            state["alerts"] = None
            _record_error(state, "CREATE_ALERT", error, recoverable=True)
        add_trace(state, "ESCALATE", {"risk_level": "high", "alerts_created": bool(state.get("alerts")),
                                       "next_status": "escalated"})
        if not state.get("report"):
            try:
                generate_report(state)
            except Exception as error:
                _record_error(state, "GENERATE_REPORT", error, recoverable=True)
        state["loop_state"]["reason"] = "escalated"
        return _response(state, build_urgent_message(state))

    safe = assess()
    if state.get("risk_level") == "high":
        return escalate()
    if not safe:
        return _interrupt(state, "safety_assessment_failed")
    if followup_waiting and max_steps > 0:
        followup["attempts"] = followup.get("attempts", 1) + 1
        state["current_goal"] = "collect_medication_details"
        state["pending_tasks"] = ["answer_medication_question"]
        if (state.get("latest_answer_status") in {"refused", "unable"}
                or followup["attempts"] >= 3):
            state["loop_state"]["reason"] = "unconfirmed_medical_question"
            return _interrupt(state, "medical_question_unconfirmed", prefix=state.get("last_question_response", ""))
        clarification = "为继续核实刚才的用药问题，请查看原处方或药盒，补充药物名称和医生原来的服用安排；无法确认也请直接说明。"
        record_question(state, clarification, topic="medication", kind="medication_details", goal="collect_medication_details")
        state["followup_status"] = "waiting_for_user"
        state["last_action"] = "ASK_USER"
        state["loop_state"]["reason"] = "waiting_for_user"
        add_trace(state, "ACTION_SELECTED", {"action": "ASK_USER", "goal": "collect_medication_details",
                   "reason": "原医学问题仍需补充信息，保持待处理且不重复调用工具", "priority": "high",
                   "required_tools": [], "basis": {"pending_medical_question": True}, "duration_ms": 0})
        add_trace(state, "ASK_USER", {"goal": "collect_medication_details", "question": clarification, "next_status": "waiting_for_user"})
        return _response(state, clarification, prefix)
    safety_basis = lambda: (state.get("bp_systolic"), state.get("bp_diastolic"), tuple(state.get("red_flags") or []),
                            tuple(state.get("symptoms") or []), state.get("medication_status"), state.get("bp_trend"))
    assessed_basis = safety_basis()

    for step in range(max_steps):
        state["loop_state"]["steps"] = step + 1
        if safety_basis() != assessed_basis:
            safe = assess("REASSESS_RISK_AFTER_STATE_CHANGE")
            assessed_basis = safety_basis()
            if state.get("risk_level") == "high":
                return escalate()
            if not safe:
                return _interrupt(state, "safety_assessment_failed", prefix=prefix)
        start = perf_counter()
        try:
            decision = select_next_decision(state, turn_actions)
            action = decision["action"]
        except Exception as error:
            _record_error(state, "PLANNER", error, recoverable=False)
            return _interrupt(state, "planner_failed", prefix=prefix)
        state["last_action"] = action
        state["current_goal"] = decision.get("goal") or state.get("current_goal") or ""
        add_trace(state, "ACTION_SELECTED", {**decision, "duration_ms": round((perf_counter() - start) * 1000, 3)})
        if action not in state["actions_completed"]:
            state["actions_completed"].append(action)
        if action == "ESCALATE":
            return escalate()
        if action == "INTERRUPT":
            return _interrupt(state, decision.get("reason") or "unconfirmed_information", prefix=prefix)
        if action == "ASK_USER":
            question = state.get("latest_patient_question") or ""
            if question:
                state["pending_patient_question"] = question
            try:
                message = generate_question(state, goal=decision.get("goal"))
            except Exception as error:
                _record_error(state, "QUESTION_PLANNER", error, recoverable=False)
                return _interrupt(state, "question_generation_failed", prefix=prefix)
            state["followup_status"] = "waiting_for_user"
            state["loop_state"]["reason"] = "waiting_for_user"
            add_trace(state, "ASK_USER", {"goal": state["current_goal"], "question": message,
                                           "next_status": "waiting_for_user"})
            return _response(state, message, prefix)
        if action == "FINISH":
            if _pending_tasks(state) or state.get("latest_patient_question"):
                return _interrupt(state, "completion_requirements_not_met", prefix=prefix)
            state["followup_status"] = "completed"
            state["loop_state"]["reason"] = "completed"
            try:
                generate_report(state)
            except Exception as error:
                _record_error(state, "GENERATE_REPORT", error, recoverable=False)
                state["followup_status"] = "interrupted"
                return _interrupt(state, "report_generation_failed", prefix=prefix)
            add_trace(state, "FINISH", {"report_generated": True, "next_status": "completed"})
            return _response(state, build_finish_message(state), prefix)
        if action not in {"CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE", "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"}:
            return _interrupt(state, "unsupported_action", prefix=prefix)

        query = build_guidance_query(state)
        signature = execution_fingerprint(state, action, query)
        if tool_is_current(state, action, query):
            turn_actions.append(action)
            add_trace(state, "TOOL_CACHE", {"tool": action, "status": "cached", "duration_ms": 0,
                                            "next_status": "active"})
            continue
        start = perf_counter()
        model_error_count = len(model_failures)
        retryable_error = False
        try:
            result = execute(action, state, query) if action == "RETRIEVE_MEDICAL_GUIDANCE" else execute(action, state)
            if action == "RETRIEVE_MEDICAL_GUIDANCE" and not isinstance(result.get("results"), list):
                raise ValueError("Invalid retrieval result")
            if action == "MEDICATION_GUIDANCE" and not isinstance(result.get("message"), str):
                raise ValueError("Invalid medication result")
            if action == "GENERATE_VISIT_ADVICE" and not isinstance(result.get("advice"), str):
                raise ValueError("Invalid visit result")
        except Exception as error:
            _record_error(state, action, error, recoverable=action != "CHECK_BP_HISTORY",
                          duration_ms=(perf_counter() - start) * 1000)
            if action == "CHECK_BP_HISTORY":
                state["tool_execution"][action] = {"fingerprint": signature, "status": "failed"}
                return _interrupt(state, "bp_history_failed", prefix=prefix)
            retryable_error = True
            if action == "RETRIEVE_MEDICAL_GUIDANCE":
                state.update(retrieved_guidance=[], knowledge_sources=[], last_guidance_query=query)
                state["guidance_fingerprint"] = execution_fingerprint(state, action, query)
                result = {"status": "fallback", "query": query, "results": []}
            elif action == "MEDICATION_GUIDANCE":
                result = tool_medication_guidance(state, call_json_model=lambda **kwargs: {})
            else:
                result = tool_generate_visit_advice(state, call_json_model=lambda **kwargs: {})
            add_trace(state, "TOOL_FALLBACK", {"tool": action, "status": "fallback", "duration_ms": 0,
                                              "next_status": "active"})
        retryable_error = retryable_error or len(model_failures) > model_error_count
        # 保存工具规范化输入的签名，趋势工具会把当前读数加入历史但不应再次分析。
        state["tool_execution"][action] = {"fingerprint": execution_fingerprint(state, action, query),
            "status": "fallback" if retryable_error else "empty" if action == "RETRIEVE_MEDICAL_GUIDANCE" and not result.get("results") else "ok",
            "retryable_error": retryable_error}
        turn_actions.append(action)
        if action == "CHECK_BP_HISTORY":
            add_trace(state, "CHECK_BP_HISTORY", {key: result.get(key) for key in ("trend", "record_count", "consecutive_rise")})
            safe = assess("REASSESS_RISK_AFTER_BP_TREND")
            assessed_basis = safety_basis()
            if state.get("risk_level") == "high":
                return escalate()
            if not safe:
                return _interrupt(state, "safety_assessment_failed", prefix=prefix)
        elif action == "RETRIEVE_MEDICAL_GUIDANCE":
            add_trace(state, "RAG_CACHE" if result.get("status") == "cached" else action,
                      {"query": query, "result_count": len(result.get("results") or []), "status": result.get("status")})
        elif action in {"MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"}:
            if safety_basis() != assessed_basis:
                safe = assess("REASSESS_RISK_AFTER_STATE_CHANGE")
                assessed_basis = safety_basis()
                if state.get("risk_level") == "high":
                    return escalate()
                if not safe:
                    return _interrupt(state, "safety_assessment_failed", prefix=prefix)
            field = "medication_guidance" if action == "MEDICATION_GUIDANCE" else "visit_advice"
            state[field] = result
            add_trace(state, action, {"status": result.get("status", "ok"), "level": result.get("level"),
                                     "needs_followup": result.get("needs_followup"), "next_status": "active"})
            patient_question = state.get("latest_patient_question") or ""
            if patient_question:
                retrieval_error = bool(state["tool_execution"].get("RETRIEVE_MEDICAL_GUIDANCE", {}).get("retryable_error"))
                message = result.get("message") or result.get("advice") or ""
                state["handled_patient_question"] = {"question": patient_question,
                    "fingerprint": execution_fingerprint(state, "ANSWER_PATIENT_QUESTION", patient_question),
                    "answer_action": action, "retryable_error": retryable_error or retrieval_error}
                state["last_question_response"] = message
                state["last_medication_question"] = patient_question if action == "MEDICATION_GUIDANCE" else state.get("last_medication_question", "")
                state["latest_patient_question"] = ""
                state["pending_patient_question"] = ""
                prefix = message
                if result.get("needs_followup") and result.get("followup_question"):
                    previous_followup = state.get("pending_question_followup") or {}
                    attempts = previous_followup.get("attempts", 0) + 1
                    state["pending_question_followup"] = {"question": patient_question,
                        "last_detail": state.get("additional_information") or "", "attempts": attempts}
                    if attempts >= 3:
                        state["current_goal"] = "collect_medication_details"
                        return _interrupt(state, "medical_question_unconfirmed", prefix=prefix)
                    record_question(state, result["followup_question"], topic="medication", kind="medication_details", goal="collect_medication_details")
                    state["followup_status"] = "waiting_for_user"
                    state["loop_state"]["reason"] = "waiting_for_user"
                    add_trace(state, "ASK_USER", {"goal": state["current_goal"], "question": result["followup_question"], "next_status": "waiting_for_user"})
                    return _response(state, result["followup_question"], prefix)
                state["pending_question_followup"] = {}
    state["loop_state"]["reason"] = "max_steps"
    try:
        next_decision = select_next_decision(state, turn_actions)
        state["loop_state"]["next_action"] = next_decision["action"]
        state["current_goal"] = next_decision.get("goal") or state.get("current_goal")
    except Exception:
        state["loop_state"]["next_action"] = "resume_processing"
    return _interrupt(state, "max_steps", prefix=prefix, event="LOOP_LIMIT")

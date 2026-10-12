"""轻量状态驱动 Planner：安全优先，输出可解释的下一步决策。"""

from medical_tools.guidance import (
    build_guidance_query, execution_fingerprint, tool_is_current,
)
from medical_tools.medication import is_medication_question
from medical_tools.safety import ATTENTION_BP_DIA, ATTENTION_BP_SYS
from .state_manager import ensure_dialogue_state


def get_available_actions(state: dict):
    actions = [
        "ASK_USER", "CHECK_BP_HISTORY", "RETRIEVE_MEDICAL_GUIDANCE",
        "MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE", "INTERRUPT", "FINISH",
    ]
    if state.get("risk_level") == "high":
        actions.append("ESCALATE")
    return actions


def _needs_answer(state, topic):
    return (not state["coverage"].get(topic)
            or state["topic_status"].get(topic) == "clarification_needed")


def _can_ask(state, topic):
    return _needs_answer(state, topic) and topic not in state.get("deferred_topics", {})


def _rising_history(state):
    memory_bp = ((state.get("patient_memory") or {}).get("blood_pressure") or {})
    readings = [row for row in state.get("bp_history", [])
                if isinstance(row, dict)
                and isinstance(row.get("systolic"), int)
                and isinstance(row.get("diastolic"), int)]
    current = {"systolic": state.get("bp_systolic"), "diastolic": state.get("bp_diastolic")}
    if (all(isinstance(value, int) for value in current.values())
            and (not readings or any(readings[-1][key] != current[key] for key in current))):
        readings.append(current)
    # 与现有 BP Trend 的最近三次连续上升条件一致，仅用于追问排序。
    recent = readings[-3:]
    recent_rise = len(recent) == 3 and any(
        recent[0][key] < recent[1][key] < recent[2][key]
        for key in ("systolic", "diastolic")
    )
    return bool(
        (state.get("bp_analysis") or {}).get("consecutive_rise")
        or state.get("bp_trend") == "rising"
        or (memory_bp.get("last_7_days") or {}).get("consecutive_rise_last_3_readings")
        or recent_rise
    )


def _current_abnormal(state):
    systolic, diastolic = state.get("bp_systolic"), state.get("bp_diastolic")
    bp_abnormal = (
        isinstance(systolic, int) and systolic >= ATTENTION_BP_SYS
        or isinstance(diastolic, int) and diastolic >= ATTENTION_BP_DIA
    )
    # 历史趋势只提高核实优先级，不作为本次症状、服药或诊断的事实。
    return bool(bp_abnormal or state.get("symptoms")
                or state["coverage"].get("bp") and _rising_history(state))


def _basis(state):
    required = ["bp", "medication", "current_symptoms", "red_flags"]
    if state.get("risk_level") == "attention":
        required.append("measurement_context")
    return {
        "risk_level": state.get("risk_level", "unknown"),
        "bp": [state.get("bp_systolic"), state.get("bp_diastolic")],
        "medication_status": state.get("medication_status", "unknown"),
        "symptom_status": state.get("symptom_status", "unknown"),
        "red_flag_status": state.get("red_flag_status", "unknown"),
        "rising_history": _rising_history(state),
        "pending_topics": [topic for topic in required if _needs_answer(state, topic)],
        "deferred_topics": list(state.get("deferred_topics") or {}),
        "has_patient_question": bool(state.get("latest_patient_question")),
    }


def _decision(state, action, goal, reason, priority="normal", required_tools=()):
    return {"action": action, "goal": goal, "reason": reason,
            "priority": priority, "required_tools": list(required_tools),
            "basis": _basis(state)}


def _tool_done(state, action, turn_actions, *, query=None, legacy_done=False):
    # 新指纹存在时，旧字符串标志不能掩盖状态或知识库变化。
    if action in (state.get("tool_execution") or {}):
        return tool_is_current(state, action, query=query)
    return action in turn_actions or legacy_done


def _unhandled_question(state):
    question = (state.get("latest_patient_question") or "").strip()
    handled = state.get("handled_patient_question") or {}
    if (question and isinstance(handled, dict)
            and handled.get("question") == question
            and handled.get("fingerprint") == execution_fingerprint(
                state, "ANSWER_PATIENT_QUESTION", query=question)):
        return ""
    return question


def select_next_decision(state: dict, turn_actions: list[str]):
    """按当前事实和未完成任务决策，不把循环结束等同于随访完成。"""
    ensure_dialogue_state(state)

    if state.get("risk_level") == "high":
        return _decision(state, "ESCALATE", "safety_escalation",
                         "确定性安全规则已判定高风险，暂停普通随访并执行升级", "critical",
                         ["CREATE_ALERT"])

    if state.get("followup_status") == "completed":
        return _decision(state, "FINISH", "completed", "本次随访已经完成，复用已有报告", "low")

    safety_pending = (_needs_answer(state, "current_symptoms")
                      or _needs_answer(state, "red_flags"))
    if _current_abnormal(state) and safety_pending:
        if _can_ask(state, "current_symptoms") or _can_ask(state, "red_flags"):
            return _decision(
                state, "ASK_USER", "check_symptoms",
                "当前血压、现有不适或近期上升趋势需要关注，优先核实尚未确认的症状和危险表现",
                "high",
            )
        return _decision(state, "INTERRUPT", "check_symptoms",
                         "必要安全信息仍未确认且当前不能继续追问，保留未完成任务", "high")

    # 医学问题可先回答，再返回未完成的随访；异常且症状未确认时仍走上方安全分支。
    question = _unhandled_question(state)
    if question:
        medication_question = is_medication_question(question)
        goal = "answer_medication_question" if medication_question else "answer_patient_question"
        if not _tool_done(
            state, "RETRIEVE_MEDICAL_GUIDANCE", turn_actions, query=question,
            legacy_done=state.get("last_guidance_query") == question,
        ):
            return _decision(state, "RETRIEVE_MEDICAL_GUIDANCE", goal,
                             "患者提出当前医学问题，需要检索与本次问题和状态一致的依据",
                             "high" if medication_question else "normal", ["RETRIEVE_MEDICAL_GUIDANCE"])
        action = "MEDICATION_GUIDANCE" if medication_question else "GENERATE_VISIT_ADVICE"
        legacy_done = (state.get("last_medication_question") == question
                       if medication_question else False)
        if not _tool_done(state, action, turn_actions, query=question, legacy_done=legacy_done):
            return _decision(state, action, goal,
                             "已完成当前问题的证据检索，使用医疗工具提供安全回答；空证据采用保守处理",
                             "high" if medication_question else "normal", [action])

    if _can_ask(state, "bp"):
        return _decision(state, "ASK_USER", "collect_bp",
                         "本次血压尚未确认，需要当前读数，历史读数不能替代本次测量", "normal")

    medication_pending = (_needs_answer(state, "medication")
                          or state.get("medication_status") == "not_taken_yet"
                          and state.get("medication_timing") == "unknown")
    if medication_pending and "medication" not in state.get("deferred_topics", {}):
        if state.get("medication_status") == "not_taken_yet":
            return _decision(state, "ASK_USER", "clarify_medication_timing",
                             "患者尚未服药，需核实原定服药时间，不能直接判定漏服", "high")
        goal = ("clarify_medication_taken"
                if state["topic_status"].get("medication") == "clarification_needed"
                else "check_medication")
        reports = (((state.get("patient_memory") or {}).get("medication") or {}).get("last_7_days") or {})
        reason = ("历史记录提示服药问题，需要重新确认今天的服药情况，不能沿用历史结论"
                  if reports.get("issue_report_count", 0)
                  else "当前服药情况尚未确认，优先核实本次服药或澄清简短回答")
        return _decision(state, "ASK_USER", goal, reason, "normal")

    if _can_ask(state, "current_symptoms") or _can_ask(state, "red_flags"):
        return _decision(state, "ASK_USER", "check_symptoms",
                         "本次症状或危险表现尚未完整确认，只追问尚未确认的内容", "normal")

    if state.get("risk_level") == "attention" and _can_ask(state, "measurement_context"):
        return _decision(state, "ASK_USER", "check_measurement_context",
                         "当前风险需要关注，核实本次血压测量条件后再整合建议", "normal")

    pending = _basis(state)["pending_topics"]
    if pending or medication_pending:
        return _decision(state, "INTERRUPT", state.get("current_goal") or "incomplete_followup",
                         "必要信息仍未确认，拒绝、无法回答或连续模糊回答已达到追问限制，不能宣称完成", "high")

    if not _tool_done(state, "CHECK_BP_HISTORY", turn_actions):
        return _decision(state, "CHECK_BP_HISTORY", "analyze_bp_history",
                         "必要信息已采集，分析当前读数与历史记录的趋势", "low", ["CHECK_BP_HISTORY"])

    if state.get("risk_level") == "attention":
        query = build_guidance_query(state)
        if not _tool_done(
            state, "RETRIEVE_MEDICAL_GUIDANCE", turn_actions, query=query,
            legacy_done=state.get("last_guidance_query") == query,
        ):
            return _decision(state, "RETRIEVE_MEDICAL_GUIDANCE", "generate_visit_advice",
                             "当前风险需要关注，检索与最新状态一致的随访和就医依据", "low",
                             ["RETRIEVE_MEDICAL_GUIDANCE"])
        if not _tool_done(
            state, "GENERATE_VISIT_ADVICE", turn_actions,
            legacy_done=bool(state.get("visit_advice")),
        ):
            return _decision(state, "GENERATE_VISIT_ADVICE", "generate_visit_advice",
                             "当前证据检索已完成，整合就医建议；证据不足时采用保守处理", "low",
                             ["GENERATE_VISIT_ADVICE"])

    return _decision(state, "FINISH", "complete_followup",
                     "必要信息已确认，所需分析和建议已处理，可以生成本次随访报告", "low")


def select_next_action(state: dict, turn_actions: list[str]):
    """保留原 Action Selector 字符串出口，现有 API 与调用者无需迁移。"""
    return select_next_decision(state, turn_actions)["action"]

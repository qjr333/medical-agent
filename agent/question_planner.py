"""Store actual question context; retain legacy questions for the older Agent.

V5 questions are chosen and written by the LLM, including supplementary
questions on a previously answered topic. Coverage does not veto that choice.
"""

from medical_tools.safety import DANGER_PATTERNS
from .state_manager import add_trace, ensure_dialogue_state


RED_FLAG_QUESTIONS = {
    "胸痛": "胸痛",
    "呼吸困难": "呼吸困难",
    "晕厥": "晕倒或晕厥",
    "意识异常": "意识不清或意识模糊",
    "突发肢体无力": "突然肢体无力",
    "突发言语异常": "突然说话不清",
}


def _needs_answer(state: dict, topic: str):
    return (
        topic not in state.get("deferred_topics", {})
        and (not state["coverage"].get(topic)
             or state["topic_status"].get(topic) == "clarification_needed")
    )


def record_question(
    state, question, *, topic, kind, goal, asked_symptoms=None, asked_red_flags=None,
):
    """语义状态用于判断完成度，尝试次数只决定澄清问题的表述。"""
    ensure_dialogue_state(state)
    state["current_goal"] = goal
    state["last_agent_question"] = question
    state["question_context"] = {
        "goal": goal,
        "topic": topic,
        "kind": kind,
        "question": question,
        "asked_symptoms": list(asked_symptoms or []),
        "asked_red_flags": list(asked_red_flags or []),
    }
    key = f"{topic}.{kind}"
    state["question_attempts"][key] = state["question_attempts"].get(key, 0) + 1
    return question


def _remember_question(state, question, goal, topic, kind, *, symptoms=(), red_flags=()):
    return record_question(
        state, question, goal=goal, topic=topic, kind=kind,
        asked_symptoms=symptoms, asked_red_flags=red_flags,
    )


def generate_question(state: dict, goal=None, *, model_proposal=None):
    """V5 records a model question; older flows use their compatibility policy."""
    ensure_dialogue_state(state)
    if state.get("agent_version") == "V5.0":
        if not isinstance(model_proposal, dict):
            raise ValueError("v5_requires_model_question")
        question = model_proposal.get("question")
        topic = model_proposal.get("topic")
        next_goal = model_proposal.get("goal") or goal
        if (not isinstance(question, str) or not question.strip()
                or not isinstance(topic, str) or not topic.strip()
                or not isinstance(next_goal, str) or not next_goal.strip()):
            raise ValueError("invalid_model_question")
        return record_question(
            state, question, topic=topic,
            kind=model_proposal.get("kind") or topic, goal=next_goal,
            asked_symptoms=model_proposal.get("asked_symptoms") or [],
            asked_red_flags=model_proposal.get("asked_red_flags") or [],
        )
    memory = state.get("patient_memory") or {}
    med_7d = (memory.get("medication") or {}).get("last_7_days") or {}
    bp_7d = (memory.get("blood_pressure") or {}).get("last_7_days") or {}
    last_bp = (memory.get("blood_pressure") or {}).get("latest_recorded") or {}
    past = memory.get("last_followup") or {}
    requested_topic = {
        "collect_bp": "bp",
        "check_medication": "medication",
        "clarify_medication_timing": "medication",
        "clarify_medication_taken": "medication",
        "collect_medication_details": "medication",
        "check_symptoms": "symptoms",
        "check_red_flags": "symptoms",
        "check_measurement_context": "measurement_context",
    }.get(goal)

    if requested_topic in (None, "bp") and _needs_answer(state, "bp"):
        if state["topic_status"]["bp"] == "clarification_needed":
            return _remember_question(
                state,
                "您目前能确认最近一次血压的两个数值吗？可以查看血压计记录；如果尚未测量或记不清，请直接告诉我。",
                "collect_bp", "bp", "bp",
            )
        if last_bp:
            add_trace(state, "MEMORY_USED", {"kind": "last_bp_record", "measured_at": last_bp.get("measured_at")})
            question = (
                f"历史记录中，上次血压为{last_bp['systolic']}/{last_bp['diastolic']} mmHg，"
                f"记录日期是{str(last_bp.get('measured_at', ''))[:10]}。"
                "这次还没有新的测量结果。如果方便，请告诉我本次血压是多少，"
                "例如150/90；如果没有测量，也可以直接告诉我。"
            )
        else:
            question = "我最近没有收到您的血压记录。方便的话，请告诉我最近一次血压，例如150/90 mmHg。"
        return _remember_question(state, question, "collect_bp", "bp", "bp")

    if requested_topic in (None, "medication") and (_needs_answer(state, "medication") or (
        state.get("medication_status") == "not_taken_yet"
        and state.get("medication_timing") == "unknown"
        and "medication" not in state.get("deferred_topics", {})
    )):
        if state.get("medication_status") == "not_taken_yet":
            if state["question_attempts"].get("medication.timing", 0):
                return _remember_question(
                    state,
                    "医生原来安排您今天几点服药？如果记不清，可以直接告诉我，我们先核实原来的安排。",
                    "clarify_medication_timing", "medication", "medication_details",
                )
            return _remember_question(
                state,
                "您是还没到规定的服药时间，还是已经超过了原定服药时间？",
                "clarify_medication_timing", "medication", "timing",
            )

        if state["topic_status"]["medication"] == "clarification_needed":
            if state["question_attempts"].get("medication.taken_clarify", 0):
                return _remember_question(
                    state,
                    "关于今天服药，您目前能够确认哪些情况？例如已服下、还没服用，或者需要查看服药记录。",
                    "clarify_medication_taken", "medication", "medication_details",
                )
            return _remember_question(
                state,
                "想确认今天这次的实际服用情况：降压药已经服下了吗？如果还没服用，可以直接告诉我‘还没吃’。",
                "clarify_medication_taken", "medication", "taken_clarify",
            )

        prefix = ""
        if med_7d.get("issue_report_count", 0) > 0:
            add_trace(state, "MEMORY_USED", {"kind": "prior_medication_reports", "report_count": med_7d["issue_report_count"]})
            prefix = "之前一周的随访记录中，您曾报告过漏服或服药不规律的情况。这次需要重新确认："
        elif bp_7d.get("consecutive_rise_last_3_readings"):
            add_trace(state, "MEMORY_USED", {"kind": "bp_rising_history", "measurement_count": bp_7d.get("measurement_count")})
            prefix = "历史记录显示最近几次血压有上升的情况。先确认一下："
        elif past.get("pending_goal") == "check_medication":
            add_trace(state, "MEMORY_USED", {"kind": "unfinished_medication_question", "session_id": past.get("session_id")})
            prefix = "上一次随访还没确认您的服药情况。这次重新确认："
        return _remember_question(
            state, prefix + "您今天是否已经按医生原来的安排服用了降压药？",
            "check_medication", "medication", "medication_taken",
        )

    if requested_topic in (None, "symptoms") and (
        _needs_answer(state, "current_symptoms") or _needs_answer(state, "red_flags")
    ):
        denied = set(state.get("denied_red_flags") or [])
        reported = set(state.get("red_flags") or [])
        remaining_flags = [label for label in DANGER_PATTERNS if label not in denied | reported]
        if not _needs_answer(state, "red_flags"):
            remaining_flags = []
        asked_symptoms = []
        descriptions = [RED_FLAG_QUESTIONS[label] for label in remaining_flags]
        if _needs_answer(state, "current_symptoms") and "头晕" not in (state.get("denied_symptoms") or []):
            asked_symptoms = ["头晕"]
            descriptions.insert(0, "明显头晕")
        details = "、".join(descriptions)
        clarification = state["topic_status"].get("current_symptoms") == "clarification_needed"
        clarification = clarification or state["topic_status"].get("red_flags") == "clarification_needed"
        if clarification:
            question = "您刚才提到的情况还需要确认，具体是哪一种不舒服？"
            if details:
                question += f"目前是否有{details}？没有这些表现也请直接说明。"
        elif not _needs_answer(state, "current_symptoms") and details:
            question = f"刚才已确认的情况不用再回答，再确认其余表现：您现在有没有{details}？"
        elif details:
            question = f"您现在有没有不舒服？比如{details}等情况？如果这些表现都没有，可以直接告诉我‘都没有’。"
        else:
            question = "除了已经确认的表现，您现在还有其他不舒服吗？可以说出具体感觉，也可以告诉我没有其他不适。"
        topic = "current_symptoms" if _needs_answer(state, "current_symptoms") else "red_flags"
        return _remember_question(
            state, question, "check_symptoms", topic, "symptoms",
            symptoms=asked_symptoms, red_flags=remaining_flags,
        )

    if requested_topic in (None, "measurement_context") and _needs_answer(state, "measurement_context"):
        if state["topic_status"].get("measurement_context") == "clarification_needed":
            question = "您能确认量血压前是否已经坐着休息了几分钟吗？记不清也可以直接告诉我。"
        else:
            question = "这次量血压前，您是否已经安静坐着休息了几分钟？"
        return _remember_question(
            state, question, "check_measurement_context", "measurement_context", "measurement_context",
        )

    if requested_topic is not None:
        # 显式目标在等待期间被新事实解决时，回到未完成主题，不重复已确认问题。
        return generate_question(state)

    return _remember_question(
        state, "除了刚才这些情况，您今天还有其他想告诉我的不舒服或用药问题吗？",
        "additional_information", "additional_information", "additional_information",
    )

"""Patient State 创建、更新和 Trace 管理。"""

from datetime import date
from typing import Any

from backend.utils import now_iso, today_local_date, local_date_from_timestamp, unique_list
from medical_tools.safety import DANGER_PATTERNS, detect_danger_signs, valid_bp_measurement
from .extractor import extract_user_patch, split_answer_clauses

def create_patient_state(
    trigger_type: str,
    *,
    DEMO_PATIENT: dict,
    DEMO_BP_HISTORY: list,
    TRIGGER_CONFIGS: dict,
):

    trigger_config = (
        TRIGGER_CONFIGS.get(
            trigger_type
        )
        or
        TRIGGER_CONFIGS[
            "scheduled"
        ]
    )


    systolic = (
        trigger_config.get(
            "bp_systolic"
        )
    )


    diastolic = (
        trigger_config.get(
            "bp_diastolic"
        )
    )


    state = {

        # -----------------------------------------
        # Basic
        # -----------------------------------------

        "patient_id":
            DEMO_PATIENT[
                "patient_id"
            ],

        "patient_name":
            DEMO_PATIENT[
                "patient_name"
            ],

        "age":
            DEMO_PATIENT[
                "age"
            ],

        "conditions":
            list(
                DEMO_PATIENT[
                    "conditions"
                ]
            ),


        # -----------------------------------------
        # Trigger
        # -----------------------------------------

        "trigger_type":
            trigger_type,

        "followup_reason":
            trigger_config[
                "followup_reason"
            ],


        # -----------------------------------------
        # BP
        # -----------------------------------------

        "bp_systolic":
            systolic,

        "bp_diastolic":
            diastolic,

        "bp_trend":
            "unknown",

        "bp_history":
            [dict(item) for item in DEMO_BP_HISTORY],

        "bp_analysis": None,

        # 历史记录由数据库读取，不覆盖本轮服药状态。
        "patient_memory": {},
        "memory_context": "",


        # -----------------------------------------
        # Medication
        # -----------------------------------------

        "medication_status":
            "unknown",

        "medication_detail":
            "",


        # -----------------------------------------
        # Symptoms
        # -----------------------------------------

        "symptoms":
            [],

        "denied_symptoms":
            [],

        "red_flags":
            [],
        
        "symptom_status":
          "unknown",

        "red_flag_status":
          "unknown",
        
        "additional_information":
            "",


        # -----------------------------------------
        # Patient Question
        # -----------------------------------------

        "latest_patient_question":
            "",

        "last_medication_question":
            "",


        # -----------------------------------------
        # Coverage
        # -----------------------------------------

        "coverage": {

            "bp":
                systolic is not None,

            "medication":
                False,

            "current_symptoms":
                False,

            "red_flags":
                False,

            "measurement_context":
                False,
        },


        # -----------------------------------------
        # RAG
        # -----------------------------------------

        "knowledge_queries":
            [],

        "retrieved_guidance":
            [],

        "knowledge_sources":
            [],

        "last_guidance_query":
            "",


        # -----------------------------------------
        # Agent
        # -----------------------------------------

        "risk_level":
            "normal",

        "risk_reasons":
            [],

        "current_goal":
            "",

        "last_action":
            "",

        "actions_completed":
            [],

        "visit_advice":
            None,

        "medication_guidance":
            None,

        "followup_status":
            "active",


        # -----------------------------------------
        # Result
        # -----------------------------------------

        "alerts":
            None,

        "report":
            None,


        # -----------------------------------------
        # Trace
        # -----------------------------------------

        "trace":
            [],
    }


    return ensure_dialogue_state(state)



def add_trace(
    state: dict,
    event: str,
    detail: Any = None,
):

    trace_item = {

        "time":
            now_iso(),

        "event":
            event,

        "detail":
            detail,
    }


    state.setdefault(
        "trace",
        []
    ).append(
        trace_item
    )


DIALOGUE_TOPICS = ("bp", "medication", "current_symptoms", "red_flags", "measurement_context")
MEDICATION_STATUSES = {"taken", "not_taken_yet", "missed", "irregular", "stopped"}
MAX_UNCLEAR_ANSWERS = 3


def _restore_legacy_dialogue(state):
    """从原 Trace 恢复问题和日期，旧 Coverage 不能证明未询问的红旗已否认。"""
    legacy_flags = "denied_red_flags" not in state
    legacy_date = "medication_date" not in state
    if not legacy_flags and not legacy_date and "question_context" in state:
        return
    context, last_context, text = {}, {}, ""
    denied_flags = []
    medication_date = None
    for event in state.get("trace") or []:
        detail = event.get("detail") or {}
        if not isinstance(detail, dict):
            continue
        if event.get("event") == "ASK_USER":
            context = {"goal": detail.get("goal", ""), "question": detail.get("question", "")}
            last_context = context
        elif event.get("event") == "USER_MESSAGE":
            text = detail.get("message") or ""
        elif event.get("event") == "STATE_UPDATE":
            patch = detail.get("patch") or {}
            if legacy_date and patch.get("medication_status") in MEDICATION_STATUSES:
                medication_date = local_date_from_timestamp(event.get("time")) or medication_date
            if legacy_flags and text:
                grounded = extract_user_patch(text, {"question_context": context},
                                              call_json_model=lambda **kwargs: {})
                denied_flags.extend(grounded.get("denied_red_flags") or [])
    if last_context and not state.get("question_context"):
        if not state.get("current_goal") or state["current_goal"] == last_context["goal"]:
            state["question_context"] = dict(last_context)
            state["last_agent_question"] = last_context["question"]
    if legacy_date:
        state["medication_date"] = (medication_date or "undated"
                                    if state.get("medication_status") in MEDICATION_STATUSES
                                    else today_local_date())
    if legacy_flags:
        state["denied_red_flags"] = unique_list(denied_flags)
        if (state.get("red_flag_status") == "denied"
                and not set(DANGER_PATTERNS).issubset(denied_flags)):
            state["red_flag_status"] = "unknown"
            state.setdefault("coverage", {})["red_flags"] = False
            state.setdefault("topic_status", {})["red_flags"] = "unanswered"


def ensure_dialogue_state(state: dict):
    """兼容旧 state_json；本轮事实、待澄清项与历史服药日期分别保存。"""
    _restore_legacy_dialogue(state)
    coverage = state.setdefault("coverage", {})
    topics = state.setdefault("topic_status", {})
    for topic in DIALOGUE_TOPICS:
        coverage.setdefault(topic, False)
        topics.setdefault(topic, "answered" if coverage[topic] else "unanswered")
        if coverage[topic] and topics[topic] == "unanswered":
            topics[topic] = "answered"
    state.setdefault("question_context", {})
    state.setdefault("last_agent_question", "")
    state.setdefault("question_attempts", {})
    state.setdefault("topic_answer_attempts", {})
    state.setdefault("deferred_topics", {})
    state.setdefault("latest_patient_message", "")
    state.setdefault("task_errors", [])
    state.setdefault("interruption_reason", "")
    state.setdefault("denied_red_flags", [])
    state.setdefault("medication_timing", "unknown")
    state.setdefault("medication_observations", {})
    today = today_local_date()
    previous_date = state.setdefault("medication_date", today)
    if previous_date != today:
        if state.get("medication_status") in MEDICATION_STATUSES:
            state["medication_observations"].setdefault(previous_date, {
                "status": state["medication_status"],
                "detail": state.get("medication_detail", ""),
                "timing": state.get("medication_timing", "unknown"),
            })
        state.update(medication_date=today, medication_status="unknown",
                     medication_detail="", medication_timing="unknown")
        coverage["medication"] = False
        topics["medication"] = "unanswered"
        state["topic_answer_attempts"].pop("medication", None)
        state["deferred_topics"].pop("medication", None)
        if state["question_context"].get("topic") == "medication":
            state["question_context"] = {}
            state["last_agent_question"] = ""
            state["current_goal"] = ""
    return state


def _pending_answer_topics(state):
    context = state.get("question_context") or {}
    topic = context.get("topic")
    if not topic:
        topic = {
            "collect_bp": "bp", "check_medication": "medication",
            "clarify_medication_timing": "medication", "clarify_medication_taken": "medication",
            "collect_medication_details": "medication", "check_symptoms": "current_symptoms",
            "check_red_flags": "red_flags", "check_measurement_context": "measurement_context",
        }.get(state.get("current_goal"))
    topics = [topic] if topic in DIALOGUE_TOPICS else []
    if context.get("kind") == "symptoms" and context.get("asked_red_flags"):
        topics.append("red_flags")
    return unique_list(topics)


def _record_answer_progress(state, patch, pending_topics):
    """有限澄清只改变待办状态，不把拒答或模糊回答写成患者事实。"""
    coverage = state["coverage"]
    attempts = state["topic_answer_attempts"]
    deferred = state["deferred_topics"]
    answer_status = patch.get("answer_status", "")
    progressed = set()
    if valid_bp_measurement(patch.get("bp_systolic"), patch.get("bp_diastolic")):
        progressed.add("bp")
    if (patch.get("medication_status") in MEDICATION_STATUSES
            or patch.get("medication_adherence")):
        progressed.add("medication")
    if patch.get("symptoms") or patch.get("denied_symptoms"):
        progressed.add("current_symptoms")
    if patch.get("denied_red_flags") or state.get("red_flags"):
        progressed.add("red_flags")
    if patch.get("topic_status", {}).get("measurement_context") == "answered":
        progressed.add("measurement_context")

    for topic in DIALOGUE_TOPICS:
        if coverage[topic] or topic in progressed:
            deferred.pop(topic, None)
            attempts.pop(topic, None)
    for topic in pending_topics:
        if coverage[topic]:
            continue
        if answer_status in {"refused", "unable"}:
            deferred[topic] = answer_status
            state["topic_status"][topic] = "clarification_needed"
        elif answer_status == "unclear" and topic not in progressed:
            attempts[topic] = min(MAX_UNCLEAR_ANSWERS, attempts.get(topic, 0) + 1)
            state["topic_status"][topic] = "clarification_needed"
            if attempts[topic] >= MAX_UNCLEAR_ANSWERS:
                deferred[topic] = "unclear_limit"


def merge_user_patch(state: dict, patch: dict, raw_text: str):
    ensure_dialogue_state(state)
    state["latest_answer_status"] = patch.get("answer_status", "")
    coverage = state["coverage"]
    topics = state["topic_status"]
    state["latest_patient_message"] = raw_text
    pending_topics = _pending_answer_topics(state)
    # 兼容旧调试接口/调用者提供的稀疏 patch；上下文推断仍由统一提取器执行。
    if "topic_status" not in patch:
        fallback = extract_user_patch(raw_text, state, call_json_model=lambda **kwargs: {})
        patch = {**fallback, **patch}
        patch["topic_status"] = fallback["topic_status"]

    bp_sys, bp_dia = patch.get("bp_systolic"), patch.get("bp_diastolic")
    if valid_bp_measurement(bp_sys, bp_dia):
        state.update(bp_systolic=bp_sys, bp_diastolic=bp_dia)
        coverage["bp"] = True
        topics["bp"] = "answered"

    today = today_local_date()
    observations = list(patch.get("medication_observations") or [])
    if not observations and patch.get("medication_status") in MEDICATION_STATUSES:
        observations = [{"date": today, "status": patch["medication_status"],
                         "detail": patch.get("medication_detail", ""),
                         "timing": patch.get("medication_timing", "unknown")}]
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("status") not in MEDICATION_STATUSES:
            continue
        observation_date = observation.get("date", today)
        try:
            date.fromisoformat(observation_date)
        except (TypeError, ValueError):
            continue
        timing = observation.get("timing", "unknown")
        if timing not in {"unknown", "not_due", "overdue"}:
            timing = "unknown"
        previous = state["medication_observations"].get(observation_date) or {}
        if (timing == "unknown" and previous.get("status") == observation["status"]
                and previous.get("timing") in {"not_due", "overdue"}):
            timing = previous["timing"]
        detail = {"status": observation["status"], "detail": observation.get("detail", ""), "timing": timing}
        state["medication_observations"][observation_date] = detail
        if observation_date == today:
            state.update(medication_status=detail["status"], medication_detail=detail["detail"],
                         medication_timing=timing, medication_date=today)
    if patch.get("medication_adherence"):
        state["medication_adherence"] = patch["medication_adherence"]
    status = state.get("medication_status", "unknown")
    coverage["medication"] = status in MEDICATION_STATUSES and (
        status != "not_taken_yet" or state["medication_timing"] != "unknown"
    )
    topics["medication"] = ("answered" if coverage["medication"] else
                            "clarification_needed" if status == "not_taken_yet" else topics["medication"])
    if not coverage["medication"] and topics["medication"] == "answered":
        topics["medication"] = "unanswered"

    positives = patch.get("symptoms") or []
    denials = patch.get("denied_symptoms") or []
    state["symptoms"] = unique_list((state.get("symptoms") or []) + positives)
    state["denied_symptoms"] = [item for item in unique_list(
        (state.get("denied_symptoms") or []) + denials
    ) if item not in state["symptoms"]]
    # 调用原安全规则，但限定每个分句的否定范围，避免前句“没有”吞掉转折后红旗。
    dangers = unique_list([flag for clause in split_answer_clauses(raw_text)
                           for flag in detect_danger_signs(clause)])
    state["red_flags"] = unique_list((state.get("red_flags") or []) + dangers)
    state["denied_red_flags"] = [flag for flag in unique_list(
        state["denied_red_flags"] + (patch.get("denied_red_flags") or [])
    ) if flag in DANGER_PATTERNS and flag not in state["red_flags"]]
    if state["symptoms"] or state["red_flags"]:
        state["symptom_status"] = "reported"
    elif denials or patch.get("topic_status", {}).get("current_symptoms") == "answered":
        state["symptom_status"] = "denied"
    if state["red_flags"]:
        state["red_flag_status"] = "reported"
    elif set(DANGER_PATTERNS).issubset(state["denied_red_flags"]):
        state["red_flag_status"] = "denied"
    else:
        state.setdefault("red_flag_status", "unknown")
    state.setdefault("symptom_status", "unknown")
    coverage["current_symptoms"] = state["symptom_status"] != "unknown"
    coverage["red_flags"] = state["red_flag_status"] != "unknown"
    for topic in ("current_symptoms", "red_flags"):
        if coverage[topic]:
            topics[topic] = "answered"
        elif patch.get("denied_red_flags") and topic == "red_flags":
            topics[topic] = "unanswered"

    for topic, value in (patch.get("topic_status") or {}).items():
        if topic not in DIALOGUE_TOPICS:
            continue
        if value == "clarification_needed" and not coverage[topic]:
            topics[topic] = value
        elif value == "answered" and topic == "measurement_context":
            coverage[topic] = True
            topics[topic] = "answered"
    additional = (patch.get("additional_information") or "").strip()
    if additional:
        previous = state.get("additional_information") or ""
        if additional not in previous.splitlines():
            state["additional_information"] = "\n".join(part for part in (previous, additional) if part)
    question = (patch.get("patient_question") or "").strip()
    state["latest_patient_question"] = question
    _record_answer_progress(state, patch, pending_topics)
    return state

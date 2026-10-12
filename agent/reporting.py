"""随访报告、结束语与高风险升级提示。"""

from backend.utils import now_iso


def build_urgent_message(
    state: dict
):

    red_flags = (
        state.get(
            "red_flags"
        )
        or
        []
    )


    if red_flags:

        reason_text = (
            "您刚才提到的症状中有需要优先处理的情况："
            +
            "、".join(
                red_flags
            )
            +
            "。"
        )


    else:

        reason_text = (
            "当前血压记录达到本系统需要优先升级处理的范围。"
        )


    return (
        reason_text
        +
        "这时不建议继续普通线上随访，"
        "应尽快联系医疗专业人员进行评估。"
        "如果症状突然出现、明显加重或情况严重，"
        "请立即寻求急救或急诊帮助。"
    )


def generate_report(
    state: dict
):

    status = state.get("followup_status", "active")
    if status == "completed" and state.get("report") is not None:
        return state["report"]
    if status in {"interrupted", "failed"}:
        # 中断只持久化尚未确认的状态；不能因循环停止生成完成报告。
        return state.get("report")

    report = {

        "patient": {

            "patient_id":
                state.get(
                    "patient_id"
                ),

            "patient_name":
                state.get(
                    "patient_name"
                ),

            "age":
                state.get(
                    "age"
                ),

            "conditions":
                state.get(
                    "conditions"
                ),
        },


        "trigger": {

            "type":
                state.get(
                    "trigger_type"
                ),

            "reason":
                state.get(
                    "followup_reason"
                ),
        },


        "blood_pressure": {

            "systolic":
                state.get(
                    "bp_systolic"
                ),

            "diastolic":
                state.get(
                    "bp_diastolic"
                ),

            "trend":
                state.get(
                    "bp_trend"
                ),

            "history":
                state.get(
                    "bp_history"
                ),
        },


        "medication": {

            "status":
                state.get(
                    "medication_status"
                ),

            "detail":
                state.get(
                    "medication_detail"
                ),
        },


        "symptoms": {

            "reported":
                state.get(
                    "symptoms"
                ),

            "denied":
                state.get(
                    "denied_symptoms"
                ),

            "red_flags":
                state.get(
                    "red_flags"
                ),
        },


        "risk": {

            "level":
                state.get(
                    "risk_level"
                ),

            "reasons":
                state.get(
                    "risk_reasons"
                ),
        },


        "visit_advice":
            state.get(
                "visit_advice"
            ),


        "medication_guidance":
            state.get(
                "medication_guidance"
            ),


        "long_term_memory": {
            "snapshot_at": (state.get("patient_memory") or {}).get("loaded_at"),
            "summary": state.get("memory_context") or "",
            "focus": (state.get("patient_memory") or {}).get("focus", []),
        },

        "knowledge": {

            "queries":
                state.get(
                    "knowledge_queries"
                ),

            "sources":
                state.get(
                    "knowledge_sources"
                ),
        },


        "generated_at":
            now_iso(),
    }

    required = state.get("required_topics") or [
        "bp", "medication", "current_symptoms", "red_flags",
    ]
    if "required_topics" not in state and state.get("risk_level") == "attention":
        required = required + ["measurement_context"]
    coverage = state.get("coverage") or {}
    errors = state.get("task_errors") or []
    report["task_completion"] = {
        "status": status,
        "required_topics": list(required),
        "confirmed_topics": [topic for topic in required if coverage.get(topic)],
        "unconfirmed_topics": [topic for topic in required if not coverage.get(topic)],
        "deferred_topics": dict(state.get("deferred_topics") or {}),
        "all_required_information_confirmed": all(coverage.get(topic) for topic in required),
        "interruption_reason": state.get("interruption_reason") or "",
        "error_count": len(errors),
        "error_summary": [
            {key: error.get(key) for key in ("tool", "error_type", "recoverable")}
            for error in errors[-3:] if isinstance(error, dict)
        ],
    }


    state[
        "report"
    ] = report


    return report


def build_finish_message(
    state: dict
):

    visit_advice = (
        state.get(
            "visit_advice"
        )
    )


    if visit_advice:

        advice = (
            visit_advice.get(
                "advice"
            )
            or
            ""
        )


        if advice:

            return (
                "这次随访的信息已经整理好了。"
                +
                advice
            )


    return (
        "这次随访的信息已经记录完成。"
        "后续请继续按计划记录家庭血压和服药情况；"
        "如果出现新的明显不适，可以及时告诉我。"
    )

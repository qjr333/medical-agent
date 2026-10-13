"""Grounded V4.2 reports with separate clinician and family views."""

from copy import deepcopy

from agent.completion import _monitoring_plan_confirmed, _patient_messages
from agent.reporting import generate_report
from medical_tools.guidance import execution_fingerprint
from medical_tools.safety import valid_bp_measurement


REPORT_SUMMARY_PROMPT = """你是随访报告的信息组织助手。只从后端提供的facts选择并排序fact_id，不能生成新的医学事实或自由文本。
summary_fact_ids用于医生摘要；doctor_attention_fact_ids用于医生关注项；family_assistance_fact_ids用于家属协助项。
每个id必须存在，且对应事实allowed_sections包含目标字段。不得推断诊断、排除未知风险、修改处方或虚构检索来源。
必须保留所有required_fact_ids中的summary与doctor_attention项，以免遗漏危险或未确认信息。
只返回JSON：{"summary_fact_ids":[],"doctor_attention_fact_ids":[],"family_assistance_fact_ids":[]}。summary_fact_ids不能为空。"""

SECTIONS = ("summary_fact_ids", "doctor_attention_fact_ids", "family_assistance_fact_ids")
TOPIC_NAMES = {
    "bp": "最近一次血压", "medication": "本次服药情况",
    "current_symptoms": "当前不适", "red_flags": "危险症状筛查",
    "measurement_context": "测量时间与条件", "patient_question": "患者提出的问题",
    "monitoring_plan": "恢复血压监测的计划", "medication_details": "药物名称与原医嘱核对",
    "patient_answer": "本次患者回答", "safety_assessment": "安全风险复核",
}
MEDICATION_NAMES = {
    "taken": "已按时服药", "not_taken_yet": "尚未服药",
    "missed": "漏服", "irregular": "服药不规律", "stopped": "已停药",
    "unknown": "尚未确认",
}
TREND_NAMES = {
    "rising": "上升", "falling": "下降", "stable": "稳定",
    "fluctuating": "波动", "unknown": "尚不能确认",
}
TOOL_NAMES = {
    "CHECK_BP_HISTORY": "血压历史分析", "RETRIEVE_MEDICAL_GUIDANCE": "医学依据检索",
    "MEDICATION_GUIDANCE": "用药辅助评估", "GENERATE_VISIT_ADVICE": "就医辅助评估",
    "ASSESS_RISK": "风险复核", "CREATE_ALERT": "安全告警",
}
TOPIC_NAMES.update({"tool:" + action: label + "尚未完成" for action, label in TOOL_NAMES.items()})


def _topic_name(topic):
    return TOPIC_NAMES.get(topic, "相关评估尚未完成" if topic.startswith("tool:") else "其他待确认信息")


def _completion_metadata(state, outcome):
    """Planning metadata is never rendered as a patient medical fact."""
    evaluation = state.get("completion_evaluation") or {}
    metadata = {"outcome": outcome, "pending_items": deepcopy(evaluation.get("pending_items") or []),
                "next_goal": evaluation.get("next_goal") or "",
                "model_status": evaluation.get("model_status") or "not_requested"}
    evidence = (evaluation.get("objective_evidence") or {}).get("monitoring_plan")
    if _monitoring_plan_confirmed(evidence, _patient_messages(state)):
        metadata["objective_evidence"] = {"monitoring_plan": evidence}
    return metadata


def _completion_outcome(state):
    if state.get("followup_status") == "escalated" or state.get("risk_level") == "high" or state.get("red_flags"):
        return "escalated"
    value = (state.get("completion_evaluation") or {}).get("outcome") or state.get("completion_outcome")
    if value in {"completed", "partial", "escalated"}:
        return value
    return "completed" if state.get("followup_status") == "completed" else "partial"


def _current_tools(state):
    """Past executions are excluded unless their input fingerprint still matches."""
    entries = (state.get("v41_tool_results") or []) + (state.get("v42_tool_results") or [])
    tools = {}
    for entry in reversed(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
            continue
        result = entry["result"]
        action = result.get("tool")
        if not action or action in tools or not isinstance(result.get("data"), dict):
            continue
        try:
            current = execution_fingerprint(state, action, entry.get("query"))
        except Exception:
            continue
        if entry.get("fingerprint") == current:
            tools[action] = deepcopy(result)
    return tools


def _sources(tools):
    sources = {}
    for action, result in tools.items():
        if result.get("status") not in {"success", "cached", "fallback", "empty"}:
            continue
        rows = result.get("sources") or result["data"].get("source_blocks") or []
        if action == "RETRIEVE_MEDICAL_GUIDANCE":
            rows = result["data"].get("results") or rows
        for row in rows:
            if not isinstance(row, dict) or not (row.get("block_id") or row.get("source_id")):
                continue
            key = (row.get("source_id"), row.get("block_id"), row.get("source_title"))
            source = sources.setdefault(key, {k: row.get(k) for k in (
                "source_id", "block_id", "source_title", "year", "source_pages")})
            source.setdefault("used_by_tools", [])
            if action not in source["used_by_tools"]:
                source["used_by_tools"].append(action)
            source["usage"] = ("used_by_medical_tool" if any(tool != "RETRIEVE_MEDICAL_GUIDANCE"
                               for tool in source["used_by_tools"]) else "retrieved")
    return list(sources.values())


def _queries(state, tools):
    queries = []
    for entry in (state.get("v41_tool_results") or []) + (state.get("v42_tool_results") or []):
        if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
            continue
        if entry["result"] != tools.get("RETRIEVE_MEDICAL_GUIDANCE"):
            continue
        query = entry.get("query")
        if isinstance(query, str) and query and query not in queries:
            queries.append(query)
    return queries


def _unconfirmed(state, base):
    required = base["task_completion"]["required_topics"]
    topics = [topic for topic in required if not (state.get("coverage") or {}).get(topic)
              or (state.get("topic_status") or {}).get(topic) == "clarification_needed"]
    pending = (state.get("completion_evaluation") or {}).get("pending_items") or []
    for item in pending:
        if isinstance(item, dict):
            item = item.get("topic") or item.get("item") or item.get("name")
        # Free model notes remain planning metadata. Only registered objectives
        # and actual tool failures may become report confirmation items.
        if isinstance(item, str) and item in TOPIC_NAMES and item not in topics:
            topics.append(item)
    return topics


def _fact_catalog(state, outcome, tools, unconfirmed, sources):
    facts = {}

    def add(identifier, render, *, attention=False, family=False, required=False):
        allowed = [SECTIONS[2]] if family else [SECTIONS[0]]
        if attention:
            allowed.append(SECTIONS[1])
        facts[identifier] = {"fact_id": identifier, "render": render,
                             "allowed_sections": allowed, "required": required}

    add("completion", {
        "completed": "本次随访信息已完成整理。",
        "partial": "本次随访部分完成，仍存在待确认信息。",
        "escalated": "本次随访已触发安全升级，应优先由医疗专业人员评估。",
    }[outcome], attention=outcome != "completed", required=True)
    risk = state.get("risk_level")
    if risk in {"normal", "attention", "high"}:
        add("risk_assessment", {
            "normal": "本次安全评估未触发高风险升级；这不能代替完整的临床风险排查。",
            "attention": "本次安全评估标记为需要关注，需结合当前信息由医生判断。",
            "high": "本次安全评估标记为高风险，应优先处理安全升级。",
        }[risk], attention=risk != "normal", required=risk == "high")
    systolic, diastolic = state.get("bp_systolic"), state.get("bp_diastolic")
    if valid_bp_measurement(systolic, diastolic):
        add("blood_pressure", f"本次记录的血压为{systolic}/{diastolic} mmHg。")
    else:
        add("blood_pressure", "本次有效血压读数尚未确认。", attention=True, required=True)
    med_status = state.get("medication_status") or "unknown"
    timing = {"not_due": "，尚未到服药时间", "overdue": "，已超过原定服药时间"}.get(state.get("medication_timing"), "")
    add("medication_status", "本次服药情况为" + MEDICATION_NAMES.get(med_status, "尚未确认")
        + (timing if med_status == "not_taken_yet" else "") + "。",
        attention=med_status != "taken")
    # Medication detail remains in the clinician view; it is never an AI family fact.
    for key, label, status_key in (("symptoms", "当前不适", "symptom_status"),
                                   ("red_flags", "危险表现", "red_flag_status")):
        reported = [str(item) for item in state.get(key) or []]
        denied = [str(item) for item in state.get("denied_" + key) or []]
        if reported:
            add(key, "患者报告" + label + "：" + "、".join(reported) + "。",
                attention=True, required=key == "red_flags")
        elif state.get(status_key) == "denied" and denied:
            add(key, "患者本次明确否认：" + "、".join(denied) + "。")
        elif key == "symptoms" and state.get(status_key) == "denied":
            add(key, "患者本次否认当前不适。")
        else:
            add(key, label + "尚未完整确认，不能将空记录解释为没有症状。",
                attention=True, required=True)
    if unconfirmed:
        add("unconfirmed", "仍待核实：" + "、".join(_topic_name(t) for t in unconfirmed) + "。",
            attention=True, required=True)
    history = tools.get("CHECK_BP_HISTORY") or {}
    data = history.get("data") or {}
    if history.get("status") in {"success", "cached"} and type(data.get("record_count")) is int:
        add("bp_analysis", f"血压历史工具分析了{data['record_count']}次实际记录，趋势为"
            + TREND_NAMES.get(data.get("trend"), "尚不能确认") + "。",
            attention=data.get("trend") in {"rising", "fluctuating"})
    for action, result in tools.items():
        data = result["data"]
        if result.get("status") in {"error", "blocked"}:
            add("tool_" + action.lower(), f"{TOOL_NAMES.get(action, '相关工具')}本次未完成有效评估，不能作为已完成检查的依据。",
                attention=True, required=True)
        elif action == "MEDICATION_GUIDANCE" and data.get("message"):
            add("medication_guidance", "用药工具建议：" + str(data["message"]),
                attention=bool(data.get("needs_followup") or data.get("needs_professional_review")))
        elif action == "GENERATE_VISIT_ADVICE" and data.get("advice"):
            add("visit_advice", "就医工具建议：" + str(data["advice"]),
                attention=data.get("level") != "home_followup")
    if sources:
        used = sum(source["usage"] == "used_by_medical_tool" for source in sources)
        add("retrieval", f"当前有效工具结果涉及{len(sources)}个知识片段，其中{used}个被医学工具采用；"
            "仅检索获得的片段不等同于已支持医学结论。")
    add("family_record", "可协助患者保存实际血压读数与测量时间，补充尚未确认的信息。", family=True)
    add("family_prescription", "可协助患者核对原医嘱，避免自行调整处方，并向医生或药师确认疑问。", family=True)
    if outcome == "escalated":
        add("family_urgent", "请优先协助患者寻求急诊或急救帮助，不要因等待报告延误处理。", family=True)
    return facts


def _selection(result, facts):
    if not isinstance(result, dict) or set(result) != set(SECTIONS):
        raise ValueError("invalid_summary_schema")
    selected = {}
    for section in SECTIONS:
        values = result[section]
        if not isinstance(values, list) or len(values) > len(facts) or any(
                not isinstance(identifier, str) or identifier not in facts
                or section not in facts[identifier]["allowed_sections"] for identifier in values):
            raise ValueError("invalid_summary_fact_ids")
        selected[section] = list(dict.fromkeys(values))
        required = {identifier for identifier, fact in facts.items()
                    if fact["required"] and section in fact["allowed_sections"]}
        if not required.issubset(selected[section]):
            raise ValueError("missing_required_summary_facts")
    if not selected[SECTIONS[0]]:
        raise ValueError("empty_summary")
    return selected


def _family_view(state, outcome, tools, facts, selected, unconfirmed, assistance, ai_status):
    # Family summaries use only this small backend-approved subset. The model's
    # order can be retained, but medication tool prose and doctor notes stay out.
    family_ids = {"completion", "blood_pressure", "medication_status", "symptoms", "red_flags",
                  "risk_assessment", "bp_analysis", "unconfirmed"}
    summary_ids = [identifier for identifier in selected[SECTIONS[0]] if identifier in family_ids]
    for identifier in ("completion", "blood_pressure", "medication_status", "symptoms", "red_flags", "unconfirmed"):
        if identifier in facts and identifier not in summary_ids:
            summary_ids.append(identifier)
    summary = "".join(facts[identifier]["render"] for identifier in summary_ids)
    alert_ids = []
    if state.get("risk_level") in {"attention", "high"}:
        alert_ids.append("risk_assessment")
    if state.get("symptoms"):
        alert_ids.append("symptoms")
    if state.get("red_flags"):
        alert_ids.append("red_flags")
    if state.get("medication_status") in {"missed", "irregular", "stopped"}:
        alert_ids.append("medication_status")
    history = tools.get("CHECK_BP_HISTORY") or {}
    if history.get("status") in {"success", "cached"} and (history.get("data") or {}).get("trend") in {"rising", "fluctuating"}:
        alert_ids.append("bp_analysis")
    return {
        "patient_name": state.get("patient_name"), "completion_outcome": outcome,
        "blood_pressure": {"systolic": state.get("bp_systolic"), "diastolic": state.get("bp_diastolic")},
        "risk_level": state.get("risk_level"), "summary": summary,
        "abnormal_alerts": [facts[identifier]["render"] for identifier in alert_ids if identifier in facts],
        "unconfirmed_items": [_topic_name(t) for t in unconfirmed],
        "assistance": assistance, "ai_summary_status": ai_status,
    }


def build_report_v42(state, *, model_call=None):
    """Create one terminal report. The model only orders backend-grounded facts."""
    outcome = _completion_outcome(state)
    existing = state.get("report")
    if isinstance(existing, dict) and existing.get("schema_version") == "4.2" and existing.get("completion_outcome") == outcome:
        return existing
    shadow = deepcopy(state)
    shadow.update(report=None, followup_status="completed" if outcome == "completed" else
                  "escalated" if outcome == "escalated" else "partially_completed")
    report = generate_report(shadow)
    tools = _current_tools(state)
    sources = _sources(tools)
    unconfirmed = _unconfirmed(state, report)
    report["task_completion"].update(
        unconfirmed_topics=unconfirmed,
        confirmed_topics=[topic for topic in report["task_completion"]["required_topics"] if topic not in unconfirmed],
        all_required_information_confirmed=not unconfirmed,
    )
    report["knowledge"] = {"queries": _queries(state, tools), "sources": sources}
    for action, field in (("MEDICATION_GUIDANCE", "medication_guidance"),
                          ("GENERATE_VISIT_ADVICE", "visit_advice")):
        result = tools.get(action) or {}
        report[field] = deepcopy(result.get("data")) if result.get("status") in {"success", "cached", "fallback"} else None
    history = tools.get("CHECK_BP_HISTORY") or {}
    report["blood_pressure"]["trend"] = (history.get("data") or {}).get("trend", "unknown")
    facts = _fact_catalog(state, outcome, tools, unconfirmed, sources)
    selected = {section: [identifier for identifier, fact in facts.items()
                         if section in fact["allowed_sections"]] for section in SECTIONS}
    ai_status, ai_reason = "not_requested", "model_not_requested"
    if outcome == "escalated":
        ai_reason = "safety_priority"
    elif model_call is not None:
        try:
            result = model_call("REPORT_SUMMARY", REPORT_SUMMARY_PROMPT, {
                "completion_outcome": outcome, "facts": list(facts.values()),
                "required_fact_ids": {section: [identifier for identifier, fact in facts.items()
                    if fact["required"] and section in fact["allowed_sections"]] for section in SECTIONS},
            })
        except Exception:
            ai_status, ai_reason = "failed", "model_call_failed"
        else:
            if result is None or result == {}:
                ai_status, ai_reason = "failed", "model_unavailable"
            else:
                try:
                    selected = _selection(result, facts)
                    ai_status, ai_reason = "success", "grounded_fact_selection"
                except ValueError:
                    ai_status, ai_reason = "invalid", "invalid_summary_fact_selection"
    summary = "".join(facts[identifier]["render"] for identifier in selected[SECTIONS[0]])
    attention = [facts[identifier]["render"] for identifier in selected[SECTIONS[1]]]
    assistance = [facts[identifier]["render"] for identifier in selected[SECTIONS[2]]]
    report.update(schema_version="4.2", completion_outcome=outcome,
                  completion_evaluation=_completion_metadata(state, outcome),
                  tool_analysis=list(tools.values()), unconfirmed_items=unconfirmed,
                  doctor_attention=attention,
                  ai_summary={"status": ai_status, "reason": ai_reason,
                              "summary": summary, "doctor_attention": attention,
                              "family_assistance": assistance,
                              "selected_fact_ids": selected})
    report["doctor_view"] = deepcopy(report)
    report["doctor_view"]["summary_kind"] = "ai_organized_facts" if ai_status == "success" else "recorded_facts"
    report["family_view"] = _family_view(state, outcome, tools, facts, selected, unconfirmed, assistance, ai_status)
    state["report"] = report
    return report

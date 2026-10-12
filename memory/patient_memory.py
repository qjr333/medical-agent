"""Patient Memory V2 with the existing 7/30-day summaries and followup focus."""

import json
from datetime import datetime, timedelta, timezone

from database.repository import (
    get_recent_bp_records,
    get_recent_medication_records,
    read_window_records,
    read_last_followup,
    count_unresolved_followups,
)
from medical_tools.safety import valid_bp_measurement

def parse_storage_time(value):
    """统一按UTC解释数据库时间；兼容SQLite默认时间与ISO-8601格式。"""
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def summarize_bp_window(records: list, reference_time: datetime, days: int):
    """严格使用测量时间计算滑动N天；与最近N次读数区分。"""
    start = reference_time - timedelta(days=days)
    valid = []
    for row in records:
        timestamp = parse_storage_time(row.get("measured_at"))
        if (
            timestamp is not None
            and start <= timestamp <= reference_time
            and valid_bp_measurement(row.get("systolic"), row.get("diastolic"))
        ):
            valid.append((timestamp, row))

    valid.sort(key=lambda item: (item[0], item[1].get("id", 0)))
    values = [item[1] for item in valid]
    result = {
        "window_days": days,
        "measurement_count": len(values),
        "measurement_days": len({item[0].date() for item in valid}),
        "systolic_avg": None,
        "diastolic_avg": None,
        "systolic_min": None,
        "systolic_max": None,
        "diastolic_min": None,
        "diastolic_max": None,
        "consecutive_rise_last_3_readings": False,
        "latest": values[-1] if values else None,
    }
    if values:
        sys_values = [row["systolic"] for row in values]
        dia_values = [row["diastolic"] for row in values]
        result.update({
            "systolic_avg": round(sum(sys_values) / len(values), 1),
            "diastolic_avg": round(sum(dia_values) / len(values), 1),
            "systolic_min": min(sys_values),
            "systolic_max": max(sys_values),
            "diastolic_min": min(dia_values),
            "diastolic_max": max(dia_values),
        })
    if len(values) >= 3:
        last = values[-3:]
        result["consecutive_rise_last_3_readings"] = (
            last[0]["systolic"] < last[1]["systolic"] < last[2]["systolic"]
            or last[0]["diastolic"] < last[1]["diastolic"] < last[2]["diastolic"]
        )
    return result


def summarize_medication_window(records: list, reference_time: datetime, days: int):
    """统计的是患者汇报次数，不能当作漏服天数/漏服药片数。"""
    start = reference_time - timedelta(days=days)
    window = []
    for row in records:
        stamp = parse_storage_time(row.get("recorded_at"))
        if stamp is not None and start <= stamp <= reference_time:
            window.append(row)
    counts = {key: sum(row.get("status") == key for row in window)
              for key in ("taken", "not_taken_yet", "missed", "irregular", "stopped")}
    return {
        "window_days": days,
        "report_count": len(window),
        "status_report_counts": counts,
        "issue_report_count": (
            counts["missed"] + counts["irregular"] + counts["stopped"]
        ),
        "distinct_report_dates": len({str(row.get("recorded_at", ""))[:10] for row in window}),
        "note": "本字段仅统计服药报告条数，不能证明每天都按时服药，也不代表漏服天数。",
    }


def extract_followup_summary(row):
    """只提取历史随访事实；历史症状不能写入本次红旗状态。"""
    if row is None:
        return None
    entry = dict(row)
    try:
        past_state = json.loads(entry.get("state_json") or "{}")
    except (ValueError, TypeError):
        past_state = {}
    try:
        report = json.loads(entry.get("report_json") or "{}")
    except (ValueError, TypeError):
        report = {}
    report_risk = report.get("risk") or {}
    visit = report.get("visit_advice") or past_state.get("visit_advice") or {}
    return {
        "session_id": entry["session_id"],
        "trigger_type": entry["trigger_type"],
        "status": entry["status"],
        "risk_level": entry["risk_level"],
        "updated_at": entry["updated_at"],
        "risk_reasons": report_risk.get("reasons") or past_state.get("risk_reasons") or [],
        "visit_advice_level": visit.get("level"),
        "pending_goal": past_state.get("current_goal") if entry["status"] in {"active", "waiting_for_user", "interrupted", "failed"} else None,
        "last_recorded_medication_status": (
            (report.get("medication") or {}).get("status")
            or past_state.get("medication_status", "unknown")
        ),
    }


def read_window_observations(patient_id: str, reference_time: datetime):
    """按时间筛选过去30天全部事件，不把“最近365条”等同于30天。"""
    oldest_date = (reference_time - timedelta(days=30)).date().isoformat()
    bp_rows, med_rows = read_window_records(patient_id, oldest_date)

    bp_records = [dict(row) for row in bp_rows]
    for row in bp_records:
        row["label"] = str(row["measured_at"])[:10]
    med_records = [dict(row) for row in med_rows]
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    bp_records.sort(key=lambda row: (parse_storage_time(row.get("measured_at")) or minimum, row["id"]))
    med_records.sort(key=lambda row: (parse_storage_time(row.get("recorded_at")) or minimum, row["id"]), reverse=True)
    return bp_records, med_records


def load_patient_memory(patient_id: str, exclude_session_id: str = None):
    """Patient Memory V2: 真实时间窗口 + 历史依从性报告 + 上次随访摘要。"""
    reference_time = datetime.now(timezone.utc)
    bp_for_windows, medication_for_windows = read_window_observations(patient_id, reference_time)
    # 窗口统计会读取所有30天事件，而原始记录仅保留最近30次/14条。
    recent_bp = get_recent_bp_records(patient_id, limit=30)
    recent_medication = get_recent_medication_records(patient_id, limit=14)

    last_row = read_last_followup(patient_id, exclude_session_id)
    unresolved_count = count_unresolved_followups(patient_id, exclude_session_id)

    seven_bp = summarize_bp_window(bp_for_windows, reference_time, 7)
    thirty_bp = summarize_bp_window(bp_for_windows, reference_time, 30)
    seven_med = summarize_medication_window(medication_for_windows, reference_time, 7)
    thirty_med = summarize_medication_window(medication_for_windows, reference_time, 30)
    last_followup = extract_followup_summary(last_row)

    focus = []
    if seven_bp["consecutive_rise_last_3_readings"]:
        focus.append("recent_bp_rise")
    if seven_med["issue_report_count"]:
        focus.append("recent_medication_issue")
    if last_followup and last_followup.get("risk_level") in ("attention", "high"):
        focus.append("previous_risk_attention")
    if last_followup and last_followup.get("pending_goal"):
        focus.append("previous_unfinished_question")

    return {
        "version": "2.0",
        "patient_id": patient_id,
        "loaded_at": reference_time.isoformat(),
        "recent_bp_records": recent_bp,
        "recent_medication_records": recent_medication,
        "last_followup": last_followup,
        "unresolved_followup_count": unresolved_count,
        "blood_pressure": {
            "last_7_days": seven_bp,
            "last_30_days": thirty_bp,
            "latest_recorded": (get_recent_bp_records(patient_id, limit=1) or [None])[-1],
        },
        "medication": {
            "last_7_days": seven_med,
            "last_30_days": thirty_med,
            "last_report": (recent_medication or [None])[0],
        },
        "recent_missed_reports": seven_med["issue_report_count"],
        "focus": focus,
        "notes": [
            "历史记录不代表当前症状、当前服药状态或当前风险判断。",
            "7/30天统计按测量/汇报时间窗口计算，不是最近7/30条记录。",
            "服药汇报次数不等于实际漏服天数，缺少报告不代表按时服药。",
        ],
    }


def memory_context_text(state: dict):
    """面向RAG和建议生成的少量、已核实历史信息。"""
    memory = state.get("patient_memory") or {}
    if not memory:
        return "暂无可用的跨随访历史。"
    parts = []
    bp = (memory.get("blood_pressure") or {}).get("last_7_days") or {}
    if bp.get("measurement_count"):
        parts.append(
            "历史近7天实际测量记录共{}条（{}个日期），均值{}/{} mmHg".format(
                bp["measurement_count"], bp["measurement_days"],
                bp["systolic_avg"], bp["diastolic_avg"],
            )
        )
    med = (memory.get("medication") or {}).get("last_7_days") or {}
    if med.get("issue_report_count"):
        parts.append("历史近7天有{}条漏服/服药不规律/停药报告".format(med["issue_report_count"]))
    past = memory.get("last_followup") or {}
    if past:
        parts.append("前次随访状态{}，当时风险等级{}".format(
            past.get("status", "unknown"), past.get("risk_level", "unknown")
        ))
    if not parts:
        return "没有足够的历史记录支持跨随访摘要。"
    return "；".join(parts) + "。以上均为历史事实，不代表本次服药或症状状态。"


def refresh_patient_memory(state: dict, session_id: str = None):
    """每轮从SQLite刷新长期历史，独立保留本轮信息。"""
    memory = load_patient_memory(state["patient_id"], exclude_session_id=session_id)
    state["patient_memory"] = memory
    # 用数据库读数分析历史；当前触发血压仍保留在本轮字段中。
    state["bp_history"] = memory["recent_bp_records"]
    state["memory_context"] = memory_context_text(state)
    return memory


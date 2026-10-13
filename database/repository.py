"""Existing patient observations, sessions, trace, alert and report persistence."""

import json
from datetime import datetime, timedelta, timezone
from threading import RLock
from collections.abc import MutableMapping

from backend.utils import now_iso, to_json
from medical_tools.safety import valid_bp_measurement
from .connection import get_connection, current_database_path

# SQLite state_json remains authoritative; memory is a process-local cache.
class ScopedSessions(MutableMapping):
    """Caches are isolated even when two databases contain the same session ID."""
    def __init__(self):
        self._data = {}
        self._lock = RLock()
    def _key(self, key):
        return (str(current_database_path()), key)
    def __getitem__(self, key):
        with self._lock:
            return self._data[self._key(key)]
    def __setitem__(self, key, value):
        with self._lock:
            self._data[self._key(key)] = value
    def __delitem__(self, key):
        with self._lock:
            del self._data[self._key(key)]
    def __iter__(self):
        scope = str(current_database_path())
        with self._lock:
            return iter([key for database, key in self._data if database == scope])
    def __len__(self):
        return sum(1 for _ in self)


sessions = ScopedSessions()
SESSION_LOCK = RLock()
_session_locks = {}
_session_locks_guard = RLock()


def session_lock(session_id):
    key = (str(current_database_path()), session_id)
    with _session_locks_guard:
        return _session_locks.setdefault(key, RLock())

def seed_demo_patient(patient: dict, bp_history: list):
    """只在首次初始化时补入3条演示历史，不污染已有数据。"""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """INSERT INTO patients
                   (patient_id, patient_name, age, conditions_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(patient_id) DO UPDATE SET
                       patient_name = excluded.patient_name,
                       age = excluded.age,
                       conditions_json = excluded.conditions_json""",
                (
                    patient["patient_id"],
                    patient["patient_name"],
                    patient["age"],
                    to_json(patient["conditions"]),
                ),
            )
            existing = conn.execute(
                "SELECT COUNT(*) FROM bp_records WHERE patient_id = ?",
                (patient["patient_id"],),
            ).fetchone()[0]
            if existing == 0:
                for offset, item in zip((3, 2, 1), bp_history):
                    measured_at = (
                        datetime.now(timezone.utc) - timedelta(days=offset)
                    ).replace(hour=8, minute=0, second=0, microsecond=0).isoformat()
                    conn.execute(
                        """INSERT INTO bp_records
                           (patient_id, systolic, diastolic, measured_at, source)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            patient["patient_id"],
                            item["systolic"], item["diastolic"],
                            measured_at, "demo_seed",
                        ),
                    )
    finally:
        conn.close()


def get_recent_bp_records(patient_id: str, limit: int = 30):
    """按真实时间顺序返回最近 N 次读数；N次不等于N天。"""
    limit = max(1, min(int(limit), 365))
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, systolic, diastolic, measured_at, source
               FROM bp_records WHERE patient_id = ?
               ORDER BY measured_at DESC, id DESC LIMIT ?""",
            (patient_id, limit),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "label": row["measured_at"][:10],
                "systolic": row["systolic"],
                "diastolic": row["diastolic"],
                "measured_at": row["measured_at"],
                "source": row["source"],
            }
            for row in reversed(rows)
        ]
    finally:
        conn.close()


def get_recent_medication_records(patient_id: str, limit: int = 14):
    limit = max(1, min(int(limit), 100))
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, medication_name, status, detail, recorded_at
               FROM medication_records WHERE patient_id = ?
               ORDER BY recorded_at DESC, id DESC LIMIT ?""",
            (patient_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def read_session(session_id: str):
    """先看缓存，缓存失效时从SQLite恢复，支持程序重启。"""
    if session_id in sessions:
        return sessions[session_id]
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT state_json FROM followup_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        state = json.loads(row["state_json"])
    except (ValueError, TypeError):
        return None
    state.setdefault("symptom_status", "unknown")
    state.setdefault("red_flag_status", "unknown")
    sessions[session_id] = state
    return state


def persist_session(
    session_id: str,
    state: dict,
    new_bp: dict = None,
    new_medication=None,
):
    """一次SQLite事务保存状态、新观察值、增量Trace、告警草稿及报告。"""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """INSERT INTO followup_sessions
                   (session_id, patient_id, trigger_type, followup_reason,
                    status, risk_level, state_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                    status = excluded.status,
                    risk_level = excluded.risk_level,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at""",
                (
                    session_id,
                    state["patient_id"],
                    state.get("trigger_type"),
                    state.get("followup_reason"),
                    state.get("followup_status", "active"),
                    state.get("risk_level", "normal"),
                    to_json(state), now_iso(),
                ),
            )
            if new_bp and valid_bp_measurement(
                new_bp.get("systolic"), new_bp.get("diastolic")
            ):
                source = new_bp.get("source", "patient_chat")
                should_insert = True
                if source == "demo_trigger":
                    # 仅演示事件去重：同一天、同一模拟读数最多保存一次。
                    # 真实设备读数应使用设备事件ID做幂等，而不能仅依赖数值。
                    should_insert = conn.execute(
                        """SELECT id FROM bp_records
                           WHERE patient_id = ? AND systolic = ? AND diastolic = ?
                             AND source = 'demo_trigger'
                             AND substr(measured_at, 1, 10) = ? LIMIT 1""",
                        (state["patient_id"], new_bp["systolic"],
                         new_bp["diastolic"], now_iso()[:10]),
                    ).fetchone() is None
                if should_insert:
                    conn.execute(
                        """INSERT INTO bp_records
                           (patient_id, systolic, diastolic, measured_at, source)
                           VALUES (?, ?, ?, ?, ?)""",
                        (state["patient_id"], new_bp["systolic"],
                         new_bp["diastolic"], now_iso(), source),
                    )
            medication_records = ([new_medication] if isinstance(new_medication, dict)
                                  else new_medication or [])
            for medication in medication_records:
                if not isinstance(medication, dict) or medication.get("status") not in {
                    "taken", "not_taken_yet", "missed", "irregular", "stopped"
                }:
                    continue
                conn.execute(
                    """INSERT INTO medication_records
                       (patient_id, medication_name, status, detail, recorded_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        state["patient_id"],
                        medication.get("medication_name"),
                        medication["status"],
                        medication.get("detail", ""), now_iso(),
                    ),
                )
            persisted_events = conn.execute(
                "SELECT COUNT(*) FROM agent_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            trace = state.get("trace", [])
            for item in trace[persisted_events:]:
                conn.execute(
                    """INSERT INTO agent_events
                       (session_id, event_name, detail_json, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        session_id,
                        item.get("event", "UNKNOWN"),
                        to_json(item.get("detail")),
                        item.get("time") or now_iso(),
                    ),
                )
            alerts = state.get("alerts") or {}
            for target, key in (
                ("family", "family_alert_content"),
                ("clinician", "clinician_alert_content"),
            ):
                content = alerts.get(key)
                if content:
                    existing = conn.execute(
                        """SELECT id FROM alerts
                           WHERE session_id = ? AND target_type = ? LIMIT 1""",
                        (session_id, target),
                    ).fetchone()
                    if existing:
                        conn.execute(
                            "UPDATE alerts SET content = ? WHERE id = ?",
                            (content, existing["id"]),
                        )
                    else:
                        conn.execute(
                            """INSERT INTO alerts
                               (session_id, target_type, content, status)
                               VALUES (?, ?, ?, 'draft')""",
                            (session_id, target, content),
                        )
            report = state.get("report")
            if report:
                existing = conn.execute(
                    "SELECT id FROM reports WHERE session_id = ? ORDER BY id LIMIT 1",
                    (session_id,),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE reports SET report_json = ? WHERE id = ?",
                        (to_json(report), existing["id"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO reports (session_id, report_json) VALUES (?, ?)",
                        (session_id, to_json(report)),
                    )
                if report.get("schema_version") == "4.2":
                    # The session upsert already holds the SQLite writer lock;
                    # report identity and unique recipient messages commit together.
                    from database.collaboration import persist_report_delivery
                    report_id = conn.execute(
                        "SELECT id FROM reports WHERE session_id = ? ORDER BY id LIMIT 1",
                        (session_id,),
                    ).fetchone()["id"]
                    persist_report_delivery(conn, report_id, state["patient_id"], report)
    finally:
        conn.close()
    sessions[session_id] = state


def get_database_status():
    """Counts used by the existing local-debug API."""
    conn = get_connection()
    try:
        tables = (
            "patients", "bp_records", "medication_records",
            "followup_sessions", "agent_events", "alerts", "reports",
        )
        return {
            "status": "ok",
            "database": "sqlite",
            "counts": {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in tables
            },
        }
    finally:
        conn.close()


def patient_exists(patient_id: str):
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT patient_id FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone() is not None
    finally:
        conn.close()


def get_patient_history(patient_id: str, limit: int = 30):
    """Return the existing history response, or None for an unknown patient."""
    limit = max(1, min(limit, 100))
    conn = get_connection()
    try:
        patient = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        if patient is None:
            return None
        sessions_data = conn.execute(
            """SELECT session_id, trigger_type, followup_reason,
                      status, risk_level, created_at, updated_at
               FROM followup_sessions WHERE patient_id = ?
               ORDER BY updated_at DESC LIMIT ?""",
            (patient_id, limit),
        ).fetchall()
        return {
            "patient": {
                "patient_id": patient["patient_id"],
                "patient_name": patient["patient_name"],
                "age": patient["age"],
                "conditions": json.loads(patient["conditions_json"] or "[]"),
            },
            "bp_records": get_recent_bp_records(patient_id, limit),
            "medication_records": get_recent_medication_records(patient_id, limit),
            "followup_sessions": [dict(row) for row in sessions_data],
        }
    finally:
        conn.close()


def read_window_records(patient_id: str, oldest_date: str):
    """Read all observations since a date; memory applies exact UTC windows."""
    conn = get_connection()
    try:
        bp_rows = conn.execute(
            """SELECT id, systolic, diastolic, measured_at, source
               FROM bp_records WHERE patient_id = ?
                 AND substr(measured_at, 1, 10) >= ?""",
            (patient_id, oldest_date),
        ).fetchall()
        med_rows = conn.execute(
            """SELECT id, medication_name, status, detail, recorded_at
               FROM medication_records WHERE patient_id = ?
                 AND substr(recorded_at, 1, 10) >= ?""",
            (patient_id, oldest_date),
        ).fetchall()
        return [dict(row) for row in bp_rows], [dict(row) for row in med_rows]
    finally:
        conn.close()


def read_last_followup(patient_id: str, exclude_session_id: str = None):
    conn = get_connection()
    try:
        query = """SELECT f.session_id, f.trigger_type, f.status, f.risk_level,
                          f.updated_at, f.state_json, r.report_json
                   FROM followup_sessions AS f
                   LEFT JOIN reports AS r ON r.session_id = f.session_id
                   WHERE f.patient_id = ?"""
        params = [patient_id]
        if exclude_session_id:
            query += " AND f.session_id != ?"
            params.append(exclude_session_id)
        query += " ORDER BY f.updated_at DESC, f.created_at DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def count_unresolved_followups(patient_id: str, exclude_session_id: str = None):
    conn = get_connection()
    try:
        query = """SELECT COUNT(*) FROM followup_sessions
                   WHERE patient_id = ? AND status IN ('active','waiting_for_user','interrupted','failed')"""
        params = [patient_id]
        if exclude_session_id:
            query += " AND session_id != ?"
            params.append(exclude_session_id)
        return conn.execute(query, params).fetchone()[0]
    finally:
        conn.close()

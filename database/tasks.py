"""Durable proactive events, UTC plans and leased task execution.

Only station records are created here. No external delivery adapter is configured.
All functions use the active database scope selected by connection.using_database.
"""

import json
import secrets
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from database.connection import get_connection
from medical_tools.safety import ATTENTION_BP_DIA, ATTENTION_BP_SYS, valid_bp_measurement

TASK_TYPES = {"scheduled", "high_bp", "missed_measurement"}
TASK_STATUSES = {"queued", "running", "waiting_for_user", "completed", "failed", "retry"}
SESSION_TERMINAL_STATUSES = {"completed", "partially_completed", "escalated"}


def utc_datetime(value=None):
    """Normalize an ISO timestamp or datetime to UTC; legacy naive values are UTC."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Timestamp must be an ISO timestamp or datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def utc_iso(value=None):
    return utc_datetime(value).isoformat(timespec="microseconds")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    return value


def _scope_ids(patient_ids):
    """None means an internal global worker; an empty explicit scope means none."""
    if patient_ids is None:
        return None
    if not isinstance(patient_ids, (list, tuple, set, frozenset)):
        raise ValueError("patient_ids must be a collection of patient identifiers")
    return tuple(dict.fromkeys(_text(patient_id, "patient_id") for patient_id in patient_ids))


def _scope_clause(patient_ids, column="patient_id"):
    identifiers = _scope_ids(patient_ids)
    if identifiers is None:
        return "", ()
    if not identifiers:
        return " AND 0", ()
    return f" AND {column} IN ({','.join('?' for _ in identifiers)})", identifiers


def init_task_schema(conn):
    """Add tables and indexes idempotently, without committing the caller's work."""
    statements = [
        """CREATE TABLE IF NOT EXISTS proactive_tasks (
            task_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL,
            event_id TEXT NOT NULL, trigger_type TEXT NOT NULL
                CHECK(trigger_type IN ('scheduled','high_bp','missed_measurement')),
            payload_json TEXT NOT NULL DEFAULT '{}', due_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK(status IN ('queued','running','waiting_for_user','completed','failed','retry')),
            attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
            lease_until TEXT, claim_token TEXT, session_id TEXT NOT NULL UNIQUE,
            last_error TEXT, result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            delivery_status TEXT NOT NULL DEFAULT 'not_configured',
            station_delivery_status TEXT NOT NULL DEFAULT 'pending',
            report_generated INTEGER NOT NULL DEFAULT 0, station_messages INTEGER NOT NULL DEFAULT 0,
            UNIQUE(patient_id,event_id), FOREIGN KEY(patient_id) REFERENCES patients(patient_id))""",
        """CREATE TABLE IF NOT EXISTS proactive_events (
            patient_id TEXT NOT NULL, event_id TEXT NOT NULL,
            event_type TEXT NOT NULL, request_json TEXT NOT NULL,
            task_id TEXT, bp_record_id INTEGER, created_at TEXT NOT NULL,
            PRIMARY KEY(patient_id,event_id),
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY(task_id) REFERENCES proactive_tasks(task_id),
            FOREIGN KEY(bp_record_id) REFERENCES bp_records(id))""",
        """CREATE TABLE IF NOT EXISTS proactive_plans (
            plan_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL,
            plan_type TEXT NOT NULL CHECK(plan_type IN ('scheduled','missed_measurement')),
            due_at TEXT NOT NULL, interval_seconds INTEGER NOT NULL,
            window_seconds INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id))""",
        """CREATE TABLE IF NOT EXISTS proactive_inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL UNIQUE,
            patient_id TEXT NOT NULL, session_id TEXT NOT NULL,
            content TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES proactive_tasks(task_id),
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id))""",
        "CREATE INDEX IF NOT EXISTS idx_proactive_tasks_due ON proactive_tasks(status,due_at)",
        "CREATE INDEX IF NOT EXISTS idx_proactive_tasks_patient ON proactive_tasks(patient_id,created_at)",
        "CREATE INDEX IF NOT EXISTS idx_proactive_plans_due ON proactive_plans(enabled,due_at)",
    ]
    for statement in statements:
        conn.execute(statement)


@contextmanager
def _connection(*, write=False):
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _patient(conn, patient_id):
    _text(patient_id, "patient_id")
    if conn.execute("SELECT 1 FROM patients WHERE patient_id=?", (patient_id,)).fetchone() is None:
        raise ValueError("Unknown patient")


def _task(row):
    if row is None:
        return None
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json"))
    result_json = result.pop("result_json")
    result["result"] = json.loads(result_json) if result_json else None
    if result["trigger_type"] == "high_bp":
        result["measurement"] = result["payload"].get("measurement")
    return result


def _find_task(conn, task_id):
    return _task(conn.execute("SELECT * FROM proactive_tasks WHERE task_id=?", (task_id,)).fetchone())


def _register_event(conn, patient_id, event_id, event_type, request, now):
    _text(event_id, "event_id")
    request_json = _json(request)
    row = conn.execute("SELECT * FROM proactive_events WHERE patient_id=? AND event_id=?",
                       (patient_id, event_id)).fetchone()
    if row:
        if row["event_type"] != event_type or row["request_json"] != request_json:
            raise ValueError("Event ID already exists with a different payload")
        return dict(row)
    conn.execute("""INSERT INTO proactive_events
        (patient_id,event_id,event_type,request_json,created_at) VALUES(?,?,?,?,?)""",
                 (patient_id, event_id, event_type, request_json, now))
    return None


def _verified_measurement(conn, patient_id, payload):
    reference = payload.get("bp_record_id")
    if not isinstance(reference, int) or isinstance(reference, bool):
        raise ValueError("high_bp requires a real bp_record_id; use record_bp_event")
    row = conn.execute("SELECT * FROM bp_records WHERE id=? AND patient_id=?", (reference, patient_id)).fetchone()
    if row is None or not valid_bp_measurement(row["systolic"], row["diastolic"]):
        raise ValueError("Blood pressure evidence is missing or invalid")
    if row["systolic"] < ATTENTION_BP_SYS and row["diastolic"] < ATTENTION_BP_DIA:
        raise ValueError("Blood pressure is below the existing attention threshold")
    given = payload.get("measurement", payload)
    if not isinstance(given, dict):
        raise ValueError("Measurement payload must be an object")
    for key in ("systolic", "diastolic"):
        if key in given and given[key] != row[key]:
            raise ValueError("Blood pressure payload conflicts with the persisted measurement")
    return {key: row[key] for key in ("id", "systolic", "diastolic", "measured_at", "source")}


def _measurement_count(conn, patient_id, start, end):
    rows = conn.execute("""SELECT systolic,diastolic FROM bp_records
        WHERE patient_id=? AND julianday(measured_at)>=julianday(?)
            AND julianday(measured_at)<julianday(?)""", (patient_id, utc_iso(start), utc_iso(end))).fetchall()
    return sum(valid_bp_measurement(row["systolic"], row["diastolic"]) for row in rows)


def _verified_missing_window(conn, patient_id, payload, due_at):
    if not payload.get("window_start") or not payload.get("window_end"):
        raise ValueError("missed_measurement requires a real window_start and window_end; use create_plan")
    start, end = utc_datetime(payload["window_start"]), utc_datetime(payload["window_end"])
    if start >= end or end > utc_datetime(due_at):
        raise ValueError("Measurement window must end by the task due_at")
    count = _measurement_count(conn, patient_id, start, end)
    if count:
        raise ValueError("The measurement window already contains valid blood pressure records")
    return {"window_start": utc_iso(start), "window_end": utc_iso(end), "measurement_count": count}


def _create_task(conn, patient_id, event_id, trigger_type, payload, due_at, max_attempts, now,
                 *, registered=False, requested_due=None):
    _patient(conn, patient_id)
    if trigger_type not in TASK_TYPES:
        raise ValueError("Unsupported trigger_type")
    if not isinstance(payload, dict):
        raise ValueError("Task payload must be an object")
    _positive(max_attempts, "max_attempts", 10)
    request = {"trigger_type": trigger_type, "payload": payload,
               "due_at": requested_due, "max_attempts": max_attempts}
    if not registered:
        existing = _register_event(conn, patient_id, event_id, "task", request, now)
        if existing:
            return _find_task(conn, existing["task_id"])
    payload = dict(payload)
    if trigger_type == "high_bp":
        payload["measurement"] = _verified_measurement(conn, patient_id, payload)
    elif trigger_type == "missed_measurement":
        payload.update(_verified_missing_window(conn, patient_id, payload, due_at))
    task_id = str(uuid.uuid4())
    session_id = "task_" + task_id.replace("-", "")
    conn.execute("""INSERT INTO proactive_tasks
        (task_id,patient_id,event_id,trigger_type,payload_json,due_at,max_attempts,session_id,created_at,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
                 (task_id, patient_id, event_id, trigger_type, _json(payload), due_at,
                  max_attempts, session_id, now, now))
    conn.execute("UPDATE proactive_events SET task_id=? WHERE patient_id=? AND event_id=?",
                 (task_id, patient_id, event_id))
    return _find_task(conn, task_id)


def create_task(patient_id, event_id, trigger_type, payload=None, due_at=None, *, max_attempts=3):
    now = utc_iso()
    due = utc_iso(due_at) if due_at is not None else now
    with _connection(write=True) as conn:
        return _create_task(conn, patient_id, event_id, trigger_type, {} if payload is None else payload, due,
                            max_attempts, now, requested_due=due if due_at is not None else None)


def record_bp_event(patient_id, event_id, systolic, diastolic, measured_at=None, source="device"):
    """Atomically deduplicate an actual reading, and queue attention-level events."""
    if not valid_bp_measurement(systolic, diastolic):
        raise ValueError("Invalid blood pressure measurement")
    _text(source, "source")
    requested_time = utc_iso(measured_at) if measured_at is not None else None
    now = utc_iso()
    request = {"systolic": systolic, "diastolic": diastolic,
               "measured_at": requested_time, "source": source}
    with _connection(write=True) as conn:
        _patient(conn, patient_id)
        existing = _register_event(conn, patient_id, event_id, "bp_measurement", request, now)
        if existing:
            measurement = conn.execute("SELECT * FROM bp_records WHERE id=?", (existing["bp_record_id"],)).fetchone()
            return {"event_id": event_id, "measurement": dict(measurement) if measurement else None,
                    "task": _find_task(conn, existing["task_id"]), "duplicate": True}
        cursor = conn.execute("""INSERT INTO bp_records(patient_id,systolic,diastolic,measured_at,source)
            VALUES(?,?,?,?,?)""", (patient_id, systolic, diastolic, requested_time or now, source))
        record_id = cursor.lastrowid
        conn.execute("UPDATE proactive_events SET bp_record_id=? WHERE patient_id=? AND event_id=?",
                     (record_id, patient_id, event_id))
        task = None
        if systolic >= ATTENTION_BP_SYS or diastolic >= ATTENTION_BP_DIA:
            task = _create_task(conn, patient_id, event_id, "high_bp", {"bp_record_id": record_id},
                                now, 3, now, registered=True)
        measurement = dict(conn.execute("SELECT * FROM bp_records WHERE id=?", (record_id,)).fetchone())
        return {"event_id": event_id, "measurement": measurement, "task": task, "duplicate": False}


def get_task(task_id):
    with _connection() as conn:
        return _find_task(conn, task_id)


def list_tasks(patient_id=None, limit=100):
    _positive(limit, "limit", 1000)
    with _connection() as conn:
        query, parameters = "SELECT * FROM proactive_tasks", []
        if patient_id is not None:
            query += " WHERE patient_id=?"
            parameters.append(patient_id)
        query += " ORDER BY created_at DESC,task_id LIMIT ?"
        return [_task(row) for row in conn.execute(query, parameters + [limit]).fetchall()]


def create_plan(patient_id, plan_type, due_at, interval_seconds=86400, *, plan_id=None, window_seconds=None):
    if plan_type not in {"scheduled", "missed_measurement"}:
        raise ValueError("Unsupported plan_type")
    interval_seconds = _positive(interval_seconds, "interval_seconds")
    window_seconds = _positive(window_seconds if window_seconds is not None else interval_seconds, "window_seconds")
    due_at, now = utc_iso(due_at), utc_iso()
    plan_id = _text(plan_id, "plan_id") if plan_id is not None else str(uuid.uuid4())
    with _connection(write=True) as conn:
        _patient(conn, patient_id)
        row = conn.execute("SELECT * FROM proactive_plans WHERE plan_id=?", (plan_id,)).fetchone()
        if row:
            if any(row[key] != value for key, value in {
                "patient_id": patient_id, "plan_type": plan_type, "due_at": due_at,
                "interval_seconds": interval_seconds, "window_seconds": window_seconds}.items()):
                raise ValueError("Plan ID already exists with different settings")
            return dict(row)
        conn.execute("""INSERT INTO proactive_plans
            (plan_id,patient_id,plan_type,due_at,interval_seconds,window_seconds,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?)""", (plan_id, patient_id, plan_type, due_at,
                                      interval_seconds, window_seconds, now, now))
        return dict(conn.execute("SELECT * FROM proactive_plans WHERE plan_id=?", (plan_id,)).fetchone())


def get_plan(plan_id):
    with _connection() as conn:
        row = conn.execute("SELECT * FROM proactive_plans WHERE plan_id=?", (plan_id,)).fetchone()
        return dict(row) if row else None


def list_plans(patient_id=None):
    with _connection() as conn:
        if patient_id is not None:
            rows = conn.execute("SELECT * FROM proactive_plans WHERE patient_id=? ORDER BY due_at", (patient_id,))
        else:
            rows = conn.execute("SELECT * FROM proactive_plans ORDER BY due_at")
        return [dict(row) for row in rows.fetchall()]


def update_plan(plan_id, *, enabled=None, due_at=None, interval_seconds=None, window_seconds=None):
    values = {}
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        values["enabled"] = int(enabled)
    if due_at is not None:
        values["due_at"] = utc_iso(due_at)
    for key, value in (("interval_seconds", interval_seconds), ("window_seconds", window_seconds)):
        if value is not None:
            values[key] = _positive(value, key)
    with _connection(write=True) as conn:
        if conn.execute("SELECT 1 FROM proactive_plans WHERE plan_id=?", (plan_id,)).fetchone() is None:
            return None
        if values:
            values["updated_at"] = utc_iso()
            assignments = ",".join(f"{key}=?" for key in values)
            conn.execute(f"UPDATE proactive_plans SET {assignments} WHERE plan_id=?", [*values.values(), plan_id])
        return dict(conn.execute("SELECT * FROM proactive_plans WHERE plan_id=?", (plan_id,)).fetchone())


def generate_due_tasks(now=None, *, max_windows=100, patient_ids=None):
    """Advance durable plans and deduplicate each UTC window in one transaction."""
    _positive(max_windows, "max_windows", 10000)
    clock, generated, checked = utc_datetime(now), [], 0
    stamp = utc_iso(clock)
    scope, scope_parameters = _scope_clause(patient_ids)
    if scope == " AND 0":
        return []
    with _connection(write=True) as conn:
        plans = conn.execute("SELECT * FROM proactive_plans WHERE enabled=1 AND due_at<=?" + scope + " ORDER BY due_at,plan_id",
                             (stamp, *scope_parameters)).fetchall()
        for plan in plans:
            due = utc_datetime(plan["due_at"])
            while due <= clock and checked < max_windows:
                checked += 1
                start = due - timedelta(seconds=plan["window_seconds"])
                payload = {"plan_id": plan["plan_id"], "window_start": utc_iso(start), "window_end": utc_iso(due)}
                missing = True
                if plan["plan_type"] == "missed_measurement":
                    count = _measurement_count(conn, plan["patient_id"], start, due)
                    payload["measurement_count"] = count
                    missing = count == 0
                if missing:
                    event_id = f"plan:{plan['plan_id']}:{utc_iso(due)}"
                    task = _create_task(conn, plan["patient_id"], event_id, plan["plan_type"], payload,
                                        utc_iso(due), 3, stamp, requested_due=utc_iso(due))
                    generated.append(task)
                due += timedelta(seconds=plan["interval_seconds"])
            conn.execute("UPDATE proactive_plans SET due_at=?,updated_at=? WHERE plan_id=?",
                         (utc_iso(due), stamp, plan["plan_id"]))
            if checked >= max_windows:
                break
    return generated


def _backoff(attempts):
    return min(300, 5 * (2 ** max(0, attempts - 1)))


def recover_expired_leases(now=None, *, patient_ids=None):
    clock = utc_datetime(now)
    stamp = utc_iso(clock)
    scope, scope_parameters = _scope_clause(patient_ids)
    if scope == " AND 0":
        return 0
    with _connection(write=True) as conn:
        rows = conn.execute("SELECT * FROM proactive_tasks WHERE status='running' AND lease_until<=?" + scope,
                            (stamp, *scope_parameters)).fetchall()
        for row in rows:
            exhausted = row["attempts"] >= row["max_attempts"]
            conn.execute("""UPDATE proactive_tasks SET status=?,due_at=?,claim_token=NULL,lease_until=NULL,
                last_error='Worker lease expired',updated_at=? WHERE task_id=? AND claim_token=?""",
                         ("failed" if exhausted else "retry", utc_iso(clock + timedelta(seconds=_backoff(row["attempts"]))),
                          stamp, row["task_id"], row["claim_token"]))
        return len(rows)


def claim_task(now=None, *, lease_seconds=300, patient_ids=None):
    _positive(lease_seconds, "lease_seconds")
    clock = utc_datetime(now)
    stamp, lease = utc_iso(clock), utc_iso(clock + timedelta(seconds=lease_seconds))
    scope, scope_parameters = _scope_clause(patient_ids)
    if scope == " AND 0":
        return None
    with _connection(write=True) as conn:
        row = conn.execute("""SELECT task_id FROM proactive_tasks
            WHERE status IN ('queued','retry') AND due_at<=? AND attempts<max_attempts
            """ + scope + " ORDER BY due_at,created_at,task_id LIMIT 1", (stamp, *scope_parameters)).fetchone()
        if row is None:
            return None
        token = secrets.token_urlsafe(24)
        conn.execute("""UPDATE proactive_tasks SET status='running',attempts=attempts+1,
            claim_token=?,lease_until=?,updated_at=? WHERE task_id=? AND status IN ('queued','retry')""",
                     (token, lease, stamp, row["task_id"]))
        return _find_task(conn, row["task_id"])


def renew_task_lease(task_id, claim_token, now=None, *, lease_seconds=300):
    _positive(lease_seconds, "lease_seconds")
    clock = utc_datetime(now)
    with _connection(write=True) as conn:
        cursor = conn.execute("""UPDATE proactive_tasks SET lease_until=?,updated_at=?
            WHERE task_id=? AND status='running' AND claim_token=? AND lease_until>?""",
                              (utc_iso(clock + timedelta(seconds=lease_seconds)), utc_iso(clock),
                               task_id, claim_token, utc_iso(clock)))
        return cursor.rowcount == 1


def finish_task(task_id, claim_token, result=None, error=None, now=None):
    """Only the current lease owner can record an execution result."""
    clock, result = utc_datetime(now), result or {}
    stamp = utc_iso(clock)
    with _connection(write=True) as conn:
        task = _find_task(conn, task_id)
        if not task or task["status"] != "running" or task["claim_token"] != claim_token:
            return None
        result_json = None
        if isinstance(result, dict):
            try:
                result_json = _json(result)
            except (ValueError, TypeError):
                error = error or "Execution returned a result that cannot be persisted as JSON"
        if error is None:
            if not isinstance(result, dict) or result.get("session_id") != task["session_id"]:
                error = "Execution did not return the preallocated session_id"
            elif not isinstance(result.get("status"), str) or result["status"] not in SESSION_TERMINAL_STATUSES | {"waiting_for_user"}:
                error = result.get("error") or f"Execution returned nonterminal status: {result.get('status')}"
        if error is not None:
            exhausted = task["attempts"] >= task["max_attempts"]
            status = "failed" if exhausted else "retry"
            due = utc_iso(clock + timedelta(seconds=_backoff(task["attempts"])))
            conn.execute("""UPDATE proactive_tasks SET status=?,due_at=?,last_error=?,result_json=?,
                claim_token=NULL,lease_until=NULL,updated_at=? WHERE task_id=? AND claim_token=?""",
                         (status, due, str(error)[:2000], result_json, stamp, task_id, claim_token))
        else:
            report = bool(result.get("report_generated", False))
            count = result.get("station_messages", 0)
            count = count if type(count) is int and count >= 0 else 0
            station = "delivered" if report and count else "no_recipients" if report else "pending"
            conn.execute("""UPDATE proactive_tasks SET status=?,last_error=NULL,result_json=?,
                claim_token=NULL,lease_until=NULL,updated_at=?,station_delivery_status=?,
                report_generated=?,station_messages=? WHERE task_id=? AND claim_token=?""",
                         ("completed" if result["status"] in SESSION_TERMINAL_STATUSES else "waiting_for_user",
                          result_json, stamp, station, int(report), count, task_id, claim_token))
            content = result.get("message") or "已创建随访会话，请打开站内会话继续。"
            conn.execute("""INSERT INTO proactive_inbox(task_id,patient_id,session_id,content,created_at)
                VALUES(?,?,?,?,?) ON CONFLICT(task_id) DO NOTHING""",
                         (task_id, task["patient_id"], task["session_id"], str(content), stamp))
        return _find_task(conn, task_id)


def sync_session_status(session_id, status, *, report_generated=False, station_messages=0):
    """Reconcile a saved session after patient replies; running leases stay owned."""
    if status not in SESSION_TERMINAL_STATUSES | {"waiting_for_user", "failed", "interrupted"}:
        return 0
    if type(station_messages) is not int or station_messages < 0:
        raise ValueError("station_messages must be a nonnegative integer")
    target = "completed" if status in SESSION_TERMINAL_STATUSES else "failed" if status in {"failed", "interrupted"} else status
    station = "delivered" if report_generated and station_messages else "no_recipients" if report_generated else "pending"
    with _connection(write=True) as conn:
        cursor = conn.execute("""UPDATE proactive_tasks
            SET status=CASE WHEN status='completed' THEN 'completed' ELSE ? END,
            report_generated=?,station_messages=?,
            station_delivery_status=?,updated_at=?
            WHERE session_id=? AND status IN ('waiting_for_user','retry','completed','failed')""",
                              (target, int(bool(report_generated)), station_messages, station, utc_iso(), session_id))
        return cursor.rowcount


def reconcile_waiting_sessions(*, patient_ids=None):
    """Recover completion already persisted before a server restart."""
    scope, scope_parameters = _scope_clause(patient_ids, "t.patient_id")
    if scope == " AND 0":
        return 0
    with _connection(write=True) as conn:
        rows = conn.execute("""SELECT t.task_id,f.status,f.state_json FROM proactive_tasks t
            JOIN followup_sessions f ON f.session_id=t.session_id
            WHERE t.status='waiting_for_user'
            AND f.status IN ('completed','partially_completed','escalated','failed','interrupted')""" + scope,
                            scope_parameters).fetchall()
        for row in rows:
            try:
                state = json.loads(row["state_json"] or "{}")
            except (ValueError, TypeError):
                state = {}
            state = state if isinstance(state, dict) else {}
            report = bool(state.get("report"))
            count = conn.execute("""SELECT COUNT(*) FROM report_messages m JOIN reports r ON r.id=m.report_id
                JOIN proactive_tasks t ON t.session_id=r.session_id WHERE t.task_id=?""", (row["task_id"],)).fetchone()[0]
            conn.execute("""UPDATE proactive_tasks SET status=?,report_generated=?,station_messages=?,
                station_delivery_status=?,updated_at=? WHERE task_id=? AND status='waiting_for_user'""",
                         ("completed" if row["status"] in SESSION_TERMINAL_STATUSES else "failed", int(report), count,
                          "delivered" if report and count else "no_recipients" if report else "pending",
                          utc_iso(), row["task_id"]))
        return len(rows)

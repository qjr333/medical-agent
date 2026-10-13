"""Authenticated report collaboration over the existing SQLite reports.

Provisioning is an internal administrative operation. Public callers may only
use an opaque bearer token; identities, roles and patient grants are reread from
SQLite on every operation, including previously delivered messages.
"""

import hashlib
import json
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from backend.utils import now_iso, to_json
from .connection import get_connection

ROLES = {"doctor", "family"}
REPORT_STATUSES = {"new", "reviewed", "followup_required", "closed"}


def init_collaboration_schema(conn):
    """Idempotent additive migration; never commit the caller's transaction."""
    statements = [
        """CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('doctor','family')),
            active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS access_tokens (
            token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            created_at TEXT NOT NULL, expires_at TEXT NOT NULL, revoked_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id))""",
        """CREATE TABLE IF NOT EXISTS patient_access (
            user_id TEXT NOT NULL, patient_id TEXT NOT NULL,
            authorized INTEGER NOT NULL DEFAULT 1 CHECK(authorized IN (0,1)),
            updated_at TEXT NOT NULL, PRIMARY KEY(user_id,patient_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id),
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id))""",
        """CREATE TABLE IF NOT EXISTS report_workflow (
            report_id INTEGER NOT NULL, doctor_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new'
                CHECK(status IN ('new','reviewed','followup_required','closed')),
            notes TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL, updated_by TEXT,
            PRIMARY KEY(report_id,doctor_id),
            FOREIGN KEY(report_id) REFERENCES reports(id),
            FOREIGN KEY(doctor_id) REFERENCES users(user_id),
            FOREIGN KEY(updated_by) REFERENCES users(user_id))""",
        """CREATE TABLE IF NOT EXISTS report_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER NOT NULL,
            recipient_id TEXT NOT NULL, recipient_role TEXT NOT NULL
                CHECK(recipient_role IN ('doctor','family')),
            content_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'unread'
                CHECK(status IN ('unread','read')),
            created_at TEXT NOT NULL, read_at TEXT,
            UNIQUE(report_id,recipient_id),
            FOREIGN KEY(report_id) REFERENCES reports(id),
            FOREIGN KEY(recipient_id) REFERENCES users(user_id))""",
        "CREATE INDEX IF NOT EXISTS idx_access_tokens_user ON access_tokens(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_patient_access_patient ON patient_access(patient_id,authorized)",
        "CREATE INDEX IF NOT EXISTS idx_report_messages_recipient ON report_messages(recipient_id,status)",
    ]
    for statement in statements:
        conn.execute(statement)


@contextmanager
def _connection(db_path=None, *, write=False):
    conn = get_connection(db_path)
    try:
        # Keep identity and grant checks in one snapshot. Writes serialize with
        # revocation, so a check cannot be followed by an unauthorized mutation.
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _identity(conn, user):
    user_id = user.get("user_id") if isinstance(user, dict) else user
    if not isinstance(user_id, str) or not user_id:
        raise PermissionError("Authentication required")
    row = conn.execute("SELECT user_id,role FROM users WHERE user_id=? AND active=1", (user_id,)).fetchone()
    if row is None or row["role"] not in ROLES:
        raise PermissionError("Account is not active")
    return dict(row)


def _require_patient(conn, user, patient_id, roles=("doctor", "family")):
    identity = _identity(conn, user)
    if identity["role"] not in roles:
        raise PermissionError("Role is not permitted")
    allowed = conn.execute("""SELECT 1 FROM patient_access
        WHERE user_id=? AND patient_id=? AND authorized=1""", (identity["user_id"], patient_id)).fetchone()
    if allowed is None:
        raise PermissionError("Patient access is not authorized")
    return identity


def require_patient_access(user, patient_id, *, roles=("doctor", "family"), db_path=None):
    with _connection(db_path) as conn:
        return _require_patient(conn, user, patient_id, roles)


def current_user(user, *, db_path=None):
    with _connection(db_path) as conn:
        return _identity(conn, user)


def provision_user(user_id, role, *, active=True, db_path=None):
    """Administrative provisioning; never exposed as an ordinary HTTP API."""
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 128:
        raise ValueError("Invalid user_id")
    if role not in ROLES or type(active) is not bool:
        raise ValueError("Invalid account role or active flag")
    with _connection(db_path, write=True) as conn:
        stamp = now_iso()
        previous = conn.execute("SELECT role FROM users WHERE user_id=?", (user_id,)).fetchone()
        conn.execute("""INSERT INTO users(user_id,role,active,created_at,updated_at)
            VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
            role=excluded.role,active=excluded.active,updated_at=excluded.updated_at""",
            (user_id, role, int(active), stamp, stamp))
        if previous is not None and previous["role"] != role:
            # Doctor assignment is not family consent, and family consent is
            # not doctor assignment. Provision the new role, then regrant it.
            conn.execute("UPDATE patient_access SET authorized=0,updated_at=? WHERE user_id=?",
                         (stamp, user_id))
    return {"user_id": user_id, "role": role, "active": active}


def set_user_active(user_id, active, *, db_path=None):
    if type(active) is not bool:
        raise ValueError("active must be boolean")
    with _connection(db_path, write=True) as conn:
        changed = conn.execute("UPDATE users SET active=?,updated_at=? WHERE user_id=?",
                               (int(active), now_iso(), user_id)).rowcount
        if not changed:
            raise LookupError("Account does not exist")


def _token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_token(user_id, *, expires_in_hours=24, db_path=None):
    if isinstance(expires_in_hours, bool) or not isinstance(expires_in_hours, (int, float)):
        raise ValueError("Invalid token lifetime")
    if not 0 < expires_in_hours <= 8760:
        raise ValueError("Token lifetime must be positive and at most one year")
    token = secrets.token_urlsafe(48)
    with _connection(db_path, write=True) as conn:
        _identity(conn, user_id)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)).isoformat()
        conn.execute("""INSERT INTO access_tokens(token_hash,user_id,created_at,expires_at)
            VALUES (?,?,?,?)""", (_token_hash(token), user_id, now_iso(), expires_at))
    return token


def authenticate_token(token, *, db_path=None):
    if not isinstance(token, str) or not 32 <= len(token) <= 512:
        return None
    with _connection(db_path) as conn:
        row = conn.execute("""SELECT u.user_id,u.role,t.expires_at
            FROM access_tokens t JOIN users u ON u.user_id=t.user_id
            WHERE t.token_hash=? AND t.revoked_at IS NULL AND u.active=1""", (_token_hash(token),)).fetchone()
        if row is None or row["role"] not in ROLES:
            return None
        try:
            expires = datetime.fromisoformat(row["expires_at"])
            if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
                return None
        except (TypeError, ValueError):
            return None
        return {"user_id": row["user_id"], "role": row["role"]}


def revoke_token(token, *, db_path=None):
    with _connection(db_path, write=True) as conn:
        return bool(conn.execute("UPDATE access_tokens SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                                 (now_iso(), _token_hash(token))).rowcount)


def link_patient(user_id, patient_id, authorized=True, *, db_path=None):
    if type(authorized) is not bool:
        raise ValueError("authorized must be boolean")
    with _connection(db_path, write=True) as conn:
        _identity(conn, user_id)
        if conn.execute("SELECT 1 FROM patients WHERE patient_id=?", (patient_id,)).fetchone() is None:
            raise LookupError("Patient does not exist")
        conn.execute("""INSERT INTO patient_access(user_id,patient_id,authorized,updated_at)
            VALUES (?,?,?,?) ON CONFLICT(user_id,patient_id) DO UPDATE SET
            authorized=excluded.authorized,updated_at=excluded.updated_at""",
            (user_id, patient_id, int(authorized), now_iso()))


def _report_row(conn, report_id):
    row = conn.execute("""SELECT r.id,r.session_id,r.report_json,r.created_at,
        f.patient_id,p.patient_name FROM reports r
        JOIN followup_sessions f ON f.session_id=r.session_id
        JOIN patients p ON p.patient_id=f.patient_id WHERE r.id=?""", (report_id,)).fetchone()
    if row is None:
        raise LookupError("Report does not exist")
    return row


def _report_json(row):
    value = json.loads(row["report_json"])
    if not isinstance(value, dict):
        raise ValueError("Invalid stored report")
    return value


def _family_view(report):
    view = report.get("family_view")
    if not isinstance(view, dict):
        return {"available": False, "message": "家属版摘要尚未生成，请联系医疗人员。"}
    # This is a separate generated view; no doctor notes or raw report fallback.
    return view


def _message_content(role, report):
    if role == "family":
        return {"title": "家属随访摘要已生成", "family_view": _family_view(report)}
    return {"title": "患者随访报告已生成", "doctor_view": report.get("doctor_view") or {}}


def persist_report_delivery(conn, report_id, patient_id, report):
    """Deliver to current grants in the caller's report-saving transaction.

    The persisted report and its session own patient identity. The argument is
    retained for the repository integration but cannot override stored content.
    """
    row = _report_row(conn, report_id)
    if row["patient_id"] != patient_id:
        raise ValueError("Report patient does not match its session")
    stored_report = _report_json(row)
    recipients = conn.execute("""SELECT u.user_id,u.role FROM users u
        JOIN patient_access a ON a.user_id=u.user_id
        WHERE a.patient_id=? AND a.authorized=1 AND u.active=1 ORDER BY u.user_id""", (patient_id,)).fetchall()
    created = 0
    stamp = now_iso()
    for recipient in recipients:
        user_id, role = recipient["user_id"], recipient["role"]
        if role not in ROLES:
            continue
        existing = conn.execute("SELECT id FROM report_messages WHERE report_id=? AND recipient_id=?",
                                (report_id, user_id)).fetchone()
        conn.execute("""INSERT INTO report_messages
            (report_id,recipient_id,recipient_role,content_json,created_at)
            VALUES (?,?,?,?,?) ON CONFLICT(report_id,recipient_id) DO UPDATE SET
            recipient_role=excluded.recipient_role,content_json=excluded.content_json""",
            (report_id, user_id, role, to_json(_message_content(role, stored_report)), stamp))
        created += int(existing is None)
        if role == "doctor":
            conn.execute("""INSERT INTO report_workflow(report_id,doctor_id,updated_at)
                VALUES (?,?,?) ON CONFLICT(report_id,doctor_id) DO NOTHING""", (report_id, user_id, stamp))
    return {"created_messages": created, "recipients": [dict(row) for row in recipients]}


def _report_view(conn, row, identity):
    report = _report_json(row)
    result = {"id": row["id"], "report_id": row["id"], "session_id": row["session_id"],
              "patient_id": row["patient_id"], "patient_name": row["patient_name"], "created_at": row["created_at"]}
    if identity["role"] == "family":
        view = _family_view(report)
        return {**result, "report": view, "family_view": view}
    workflow = conn.execute("SELECT status,notes,updated_at,updated_by FROM report_workflow WHERE report_id=? AND doctor_id=?",
                            (row["id"], identity["user_id"])).fetchone()
    return {**result, "report": report, "doctor_view": report.get("doctor_view") or {},
            **(dict(workflow) if workflow else {"status": "new", "notes": "", "updated_at": None, "updated_by": None})}


def list_reports(user, *, patient_id=None, limit=50, db_path=None):
    limit = max(1, min(int(limit), 200))
    with _connection(db_path) as conn:
        identity = _identity(conn, user)
        if patient_id is not None:
            _require_patient(conn, identity, patient_id)
        rows = conn.execute("""SELECT r.id,r.session_id,r.report_json,r.created_at,f.patient_id,p.patient_name
            FROM reports r JOIN followup_sessions f ON f.session_id=r.session_id
            JOIN patients p ON p.patient_id=f.patient_id
            JOIN patient_access a ON a.patient_id=f.patient_id AND a.user_id=? AND a.authorized=1
            JOIN users u ON u.user_id=a.user_id AND u.active=1 AND u.role=?
            WHERE (? IS NULL OR f.patient_id=?) ORDER BY r.created_at DESC,r.id DESC LIMIT ?""",
            (identity["user_id"], identity["role"], patient_id, patient_id, limit)).fetchall()
        return [_report_view(conn, row, identity) for row in rows]


def get_report(user, report_id, *, db_path=None):
    with _connection(db_path) as conn:
        row = _report_row(conn, report_id)
        identity = _require_patient(conn, user, row["patient_id"])
        return _report_view(conn, row, identity)


def _message_row(conn, user, message_id):
    identity = _identity(conn, user)
    row = conn.execute("""SELECT m.*,f.patient_id FROM report_messages m
        JOIN reports r ON r.id=m.report_id JOIN followup_sessions f ON f.session_id=r.session_id
        JOIN patient_access a ON a.patient_id=f.patient_id AND a.user_id=m.recipient_id AND a.authorized=1
        JOIN users u ON u.user_id=m.recipient_id AND u.active=1 AND u.role=m.recipient_role
        WHERE m.id=? AND m.recipient_id=?""", (message_id, identity["user_id"])).fetchone()
    if row is None:
        raise PermissionError("Message access is not authorized")
    return row, identity


def _message_view(conn, row, identity):
    # Rederive role-specific content from authoritative report; stale stored
    # payloads cannot disclose a former doctor view after an account role change.
    report = _report_json(_report_row(conn, row["report_id"]))
    return {"id": row["id"], "message_id": row["id"], "report_id": row["report_id"],
            "patient_id": row["patient_id"], "recipient_id": row["recipient_id"],
            "recipient_role": identity["role"], "status": row["status"],
            "created_at": row["created_at"], "read_at": row["read_at"],
            "content": _message_content(identity["role"], report)}


def list_messages(user, *, limit=50, unread_only=False, db_path=None):
    limit = max(1, min(int(limit), 200))
    with _connection(db_path) as conn:
        identity = _identity(conn, user)
        rows = conn.execute("""SELECT m.*,f.patient_id FROM report_messages m
            JOIN reports r ON r.id=m.report_id JOIN followup_sessions f ON f.session_id=r.session_id
            JOIN patient_access a ON a.patient_id=f.patient_id AND a.user_id=m.recipient_id AND a.authorized=1
            JOIN users u ON u.user_id=m.recipient_id AND u.active=1 AND u.role=m.recipient_role
            WHERE m.recipient_id=? AND (?=0 OR m.status='unread')
            ORDER BY m.created_at DESC,m.id DESC LIMIT ?""", (identity["user_id"], int(bool(unread_only)), limit)).fetchall()
        return [_message_view(conn, row, identity) for row in rows]


def get_message(user, message_id, *, db_path=None):
    with _connection(db_path) as conn:
        row, identity = _message_row(conn, user, message_id)
        return _message_view(conn, row, identity)


def mark_read(user, message_id, *, db_path=None):
    with _connection(db_path, write=True) as conn:
        row, identity = _message_row(conn, user, message_id)
        conn.execute("UPDATE report_messages SET status='read',read_at=COALESCE(read_at,?) WHERE id=?",
                     (now_iso(), row["id"]))
        row, identity = _message_row(conn, identity, message_id)
        return _message_view(conn, row, identity)


def update_report_status(user, report_id, status, notes="", *, db_path=None):
    if status not in REPORT_STATUSES or not isinstance(notes, str) or len(notes) > 4000:
        raise ValueError("Invalid report status or notes")
    with _connection(db_path, write=True) as conn:
        row = _report_row(conn, report_id)
        identity = _require_patient(conn, user, row["patient_id"], ("doctor",))
        conn.execute("""INSERT INTO report_workflow(report_id,doctor_id,status,notes,updated_at,updated_by)
            VALUES (?,?,?,?,?,?) ON CONFLICT(report_id,doctor_id) DO UPDATE SET
            status=excluded.status,notes=excluded.notes,updated_at=excluded.updated_at,updated_by=excluded.updated_by""",
            (report_id, identity["user_id"], status, notes, now_iso(), identity["user_id"]))
        return _report_view(conn, row, identity)


def repush_report(user, report_id, *, db_path=None):
    with _connection(db_path, write=True) as conn:
        row = _report_row(conn, report_id)
        _require_patient(conn, user, row["patient_id"], ("doctor",))
        return persist_report_delivery(conn, report_id, row["patient_id"], _report_json(row))

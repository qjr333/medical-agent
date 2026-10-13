"""SQLite connection and additive, idempotent initialization/migrations."""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "medical_agent.db"


def get_connection(db_path=None):
    """Open SQLite without initializing or replacing existing data."""
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_db(db_path=None):
    conn = get_connection(db_path)

    try:
        conn.executescript("""
        -- 1. 患者基本信息
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            patient_name TEXT NOT NULL,
            age INTEGER,
            conditions_json TEXT DEFAULT '[]',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- 2. 血压历史记录
        CREATE TABLE IF NOT EXISTS bp_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            systolic INTEGER NOT NULL,
            diastolic INTEGER NOT NULL,
            measured_at TEXT NOT NULL,
            source TEXT DEFAULT 'home',
            FOREIGN KEY (patient_id)
                REFERENCES patients(patient_id)
        );

        -- 3. 服药记录
        CREATE TABLE IF NOT EXISTS medication_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            medication_name TEXT,
            status TEXT,
            detail TEXT,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id)
                REFERENCES patients(patient_id)
        );

        -- 4. 随访会话
        CREATE TABLE IF NOT EXISTS followup_sessions (
            session_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            trigger_type TEXT,
            followup_reason TEXT,
            status TEXT DEFAULT 'active',
            risk_level TEXT DEFAULT 'normal',
            state_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id)
                REFERENCES patients(patient_id)
        );

        -- 5. Agent执行记录
        CREATE TABLE IF NOT EXISTS agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            detail_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id)
                REFERENCES followup_sessions(session_id)
        );

        -- 6. 告警记录
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            target_type TEXT,
            content TEXT,
            status TEXT DEFAULT 'draft',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id)
                REFERENCES followup_sessions(session_id)
        );

        -- 7. 随访报告
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id)
                REFERENCES followup_sessions(session_id)
        );

        -- 常用查询索引
        CREATE INDEX IF NOT EXISTS idx_bp_patient_time
        ON bp_records(patient_id, measured_at);

        CREATE INDEX IF NOT EXISTS idx_med_patient_time
        ON medication_records(patient_id, recorded_at);

        CREATE INDEX IF NOT EXISTS idx_events_session
        ON agent_events(session_id);
        """)

        from database.collaboration import init_collaboration_schema
        init_collaboration_schema(conn)
        conn.commit()

    finally:
        conn.close()


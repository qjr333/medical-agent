"""Persistent simulation registry, physically separate from clinical records."""
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.utils import now_iso, to_json
from database.connection import get_connection


class SimulationStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = self.root / "simulation_registry.db"
        with get_connection(self.registry) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS simulation_runs (
                run_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL, owner TEXT, lease_until TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")

    def create(self, scenario_id, options):
        run_id, token = str(uuid.uuid4()), secrets.token_urlsafe(40)
        payload = {"run_id": run_id, "scenario_id": scenario_id, "status": "queued",
                   "is_simulation": True, "options": options, "transcript": [],
                   "evaluation": None, "created_at": now_iso()}
        with get_connection(self.registry) as conn:
            conn.execute("INSERT INTO simulation_runs(run_id,token_hash,payload_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                         (run_id, hashlib.sha256(token.encode()).hexdigest(), to_json(payload), now_iso(), now_iso()))
        return payload, token

    def get(self, run_id, token=None, *, trusted=False):
        try:
            if str(uuid.UUID(run_id)) != run_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise LookupError("Unknown simulation") from None
        with get_connection(self.registry) as conn:
            row = conn.execute("SELECT * FROM simulation_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise LookupError("Unknown simulation")
        if not trusted and (not isinstance(token, str) or not secrets.compare_digest(
                row["token_hash"], hashlib.sha256(token.encode()).hexdigest())):
            raise PermissionError("Simulation access required")
        return json.loads(row["payload_json"])

    def database(self, run_id):
        self.get(run_id, trusted=True)
        return self.root / run_id / "medical_agent.db"

    def claim(self, run_id, owner):
        now = now_iso()
        lease = (datetime.now(timezone.utc) + timedelta(seconds=180)).isoformat()
        with get_connection(self.registry) as conn:
            result = conn.execute("""UPDATE simulation_runs SET owner=?,lease_until=?
                WHERE run_id=? AND (owner IS NULL OR lease_until<?)""", (owner, lease, run_id, now))
            return bool(result.rowcount)

    def save(self, payload, *, owner=None, release=False):
        payload["updated_at"] = now_iso()
        lease = None if release else (datetime.now(timezone.utc) + timedelta(seconds=180)).isoformat()
        with get_connection(self.registry) as conn:
            if owner is None:
                conn.execute("UPDATE simulation_runs SET payload_json=?,updated_at=? WHERE run_id=?",
                             (to_json(payload), now_iso(), payload["run_id"]))
            else:
                changed = conn.execute("""UPDATE simulation_runs SET payload_json=?,updated_at=?,
                    lease_until=?,owner=? WHERE run_id=? AND owner=?""",
                    (to_json(payload), now_iso(), lease, None if release else owner, payload["run_id"], owner))
                if not changed.rowcount:
                    raise RuntimeError("Simulation lease lost")

    def renew(self, run_id, owner):
        lease = (datetime.now(timezone.utc) + timedelta(seconds=180)).isoformat()
        with get_connection(self.registry) as conn:
            return bool(conn.execute("UPDATE simulation_runs SET lease_until=? WHERE run_id=? AND owner=?",
                                     (lease, run_id, owner)).rowcount)

    def pending(self):
        with get_connection(self.registry) as conn:
            rows = conn.execute("SELECT payload_json FROM simulation_runs").fetchall()
        return [json.loads(row[0]) for row in rows if json.loads(row[0]).get("status") in {"queued", "running"}]

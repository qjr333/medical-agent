"""V4.2 authorization and delivery regressions on isolated temporary SQLite."""

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.access import authenticated_user, require_patient_access, require_role
from database import collaboration as store
from database import connection


class CollaborationV42Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "collaboration.db"
        self.db_override = patch.object(connection, "DB_PATH", self.db_path)
        self.db_override.start()
        self.addCleanup(self.db_override.stop)
        connection.init_db(self.db_path)
        conn = connection.get_connection(self.db_path)
        try:
            with conn:
                store.init_collaboration_schema(conn)
                for patient_id in ("P1", "P2"):
                    conn.execute("INSERT INTO patients(patient_id,patient_name) VALUES (?,?)",
                                 (patient_id, "测试患者" + patient_id))
        finally:
            conn.close()
        self.doctor = store.provision_user("doctor-1", "doctor")
        self.other = store.provision_user("doctor-2", "doctor")
        self.family = store.provision_user("family-1", "family")
        store.link_patient(self.doctor["user_id"], "P1")
        store.link_patient(self.other["user_id"], "P2")
        store.link_patient(self.family["user_id"], "P1")
        self.report1 = self.create_report("P1", "S1", {
            "schema_version": "4.2", "patient": {"patient_id": "P1"},
            "doctor_view": {"summary": "医生完整记录", "private_detail": "doctor-secret"},
            "family_view": {"summary": "家属提醒：协助记录血压"},
            "internal_information": "full-report-secret",
        })
        self.report2 = self.create_report("P2", "S2", {
            "schema_version": "4.2", "doctor_view": {"summary": "其他患者记录"},
            "family_view": {"summary": "其他家属摘要"},
        })

    def query(self, sql, parameters=()):
        conn = connection.get_connection(self.db_path)
        try:
            return [dict(row) for row in conn.execute(sql, parameters).fetchall()]
        finally:
            conn.close()

    def create_report(self, patient_id, session_id, report):
        conn = connection.get_connection(self.db_path)
        try:
            with conn:
                conn.execute("INSERT INTO followup_sessions(session_id,patient_id) VALUES (?,?)",
                             (session_id, patient_id))
                report_id = conn.execute("INSERT INTO reports(session_id,report_json) VALUES (?,?)",
                                         (session_id, json.dumps(report))).lastrowid
                store.persist_report_delivery(conn, report_id, patient_id, report)
            return report_id
        finally:
            conn.close()

    def test_additive_schema_is_idempotent_and_provisions_no_accounts(self):
        conn = connection.get_connection(self.db_path)
        try:
            with conn:
                store.init_collaboration_schema(conn)
                store.init_collaboration_schema(conn)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0], 3)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0], 2)
        finally:
            conn.close()
        clean_path = self.db_path.parent / "clean.db"
        connection.init_db(clean_path)
        clean = connection.get_connection(clean_path)
        try:
            with clean:
                store.init_collaboration_schema(clean)
            self.assertEqual(clean.execute("SELECT COUNT(*) FROM users").fetchone()[0], 0)
            self.assertEqual(clean.execute("SELECT COUNT(*) FROM access_tokens").fetchone()[0], 0)
        finally:
            clean.close()

    def test_opaque_tokens_are_hashed_unique_and_expire_after_24_hours(self):
        start = datetime.now(timezone.utc)
        first = store.issue_token("doctor-1")
        second = store.issue_token("doctor-1")
        self.assertGreaterEqual(len(first), 48)
        self.assertNotEqual(first, second)
        rows = self.query("SELECT token_hash,expires_at FROM access_tokens")
        self.assertEqual(len(rows), 2)
        self.assertIn(hashlib.sha256(first.encode()).hexdigest(), [row["token_hash"] for row in rows])
        for row in rows:
            self.assertNotEqual(row["token_hash"], first)
            self.assertTrue(start + timedelta(hours=23.99) < datetime.fromisoformat(row["expires_at"])
                            < start + timedelta(hours=24.01))
        self.assertEqual(store.authenticate_token(first), {"user_id": "doctor-1", "role": "doctor"})
        self.assertIsNone(store.authenticate_token("doctor-1"))
        self.assertIsNone(store.authenticate_token("x" * 64))

    def test_expired_revoked_and_inactive_credentials_fail_immediately(self):
        expired = store.issue_token("doctor-1")
        conn = connection.get_connection(self.db_path)
        try:
            with conn:
                conn.execute("UPDATE access_tokens SET expires_at=? WHERE token_hash=?",
                             ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                              hashlib.sha256(expired.encode()).hexdigest()))
        finally:
            conn.close()
        self.assertIsNone(store.authenticate_token(expired))
        revoked = store.issue_token("doctor-1")
        self.assertTrue(store.revoke_token(revoked))
        self.assertFalse(store.revoke_token(revoked))
        self.assertIsNone(store.authenticate_token(revoked))
        live = store.issue_token("doctor-1")
        store.set_user_active("doctor-1", False)
        self.assertIsNone(store.authenticate_token(live))
        with self.assertRaises(PermissionError):
            store.get_report(self.doctor, self.report1)
        with self.assertRaises(PermissionError):
            store.issue_token("doctor-1")

    def test_fastapi_ignores_claimed_identity_headers_and_requires_bearer(self):
        app = FastAPI()
        @app.get("/test-identity")
        def identity(user=Depends(authenticated_user)):
            return user
        @app.get("/test-doctor")
        def doctor(user=Depends(authenticated_user)):
            return require_role(user, "doctor")
        @app.get("/test-patient/{patient_id}")
        def patient(patient_id: str, user=Depends(authenticated_user)):
            return require_patient_access(user, patient_id, "doctor")
        token = store.issue_token("family-1")
        with TestClient(app) as http:
            spoof = {"X-User-Id": "doctor-1", "X-Role": "doctor"}
            self.assertEqual(http.get("/test-identity", headers=spoof).status_code, 401)
            headers = {**spoof, "Authorization": "Bearer " + token}
            self.assertEqual(http.get("/test-identity", headers=headers).json(),
                             {"user_id": "family-1", "role": "family"})
            self.assertEqual(http.get("/test-doctor", headers=headers).status_code, 403)
            doctor_token = store.issue_token("doctor-1")
            doctor_headers = {"Authorization": "Bearer " + doctor_token}
            self.assertEqual(http.get("/test-patient/P1", headers=doctor_headers).status_code, 200)
            self.assertEqual(http.get("/test-patient/P2", headers=doctor_headers).status_code, 403)
            store.revoke_token(doctor_token)
            self.assertEqual(http.get("/test-patient/P1", headers=doctor_headers).status_code, 401)

    def test_reports_and_messages_never_cross_patient_or_recipient(self):
        self.assertEqual([row["report_id"] for row in store.list_reports(self.doctor)], [self.report1])
        self.assertEqual([row["report_id"] for row in store.list_reports(self.other)], [self.report2])
        message = store.list_messages(self.doctor)[0]
        for operation in [
            lambda: store.get_report(self.other, self.report1),
            lambda: store.list_reports(self.other, patient_id="P1"),
            lambda: store.get_message(self.other, message["id"]),
            lambda: store.mark_read(self.other, message["id"]),
            lambda: store.update_report_status(self.other, self.report1, "reviewed"),
            lambda: store.repush_report(self.other, self.report1),
            lambda: store.update_report_status(self.family, self.report1, "reviewed"),
            lambda: store.repush_report(self.family, self.report1),
        ]:
            with self.assertRaises(PermissionError):
                operation()

    def test_family_receives_only_generated_family_view_without_notes(self):
        store.update_report_status(self.doctor, self.report1, "reviewed", "private-doctor-note")
        family_report = store.get_report(self.family, self.report1)
        self.assertEqual(family_report["report"], {"summary": "家属提醒：协助记录血压"})
        serialized = json.dumps([family_report, store.list_reports(self.family), store.list_messages(self.family)])
        for forbidden in ("private-doctor-note", "doctor-secret", "full-report-secret", "doctor_view", "updated_by"):
            self.assertNotIn(forbidden, serialized)
        self.assertNotIn("notes", family_report)

    def test_legacy_report_without_family_view_has_no_full_report_fallback(self):
        legacy_id = self.create_report("P1", "legacy", {
            "patient": {"patient_id": "P1"}, "risk": {"private_detail": "legacy-secret"},
        })
        self.assertFalse(store.get_report(self.family, legacy_id)["family_view"]["available"])
        serialized = json.dumps([store.get_report(self.family, legacy_id), store.list_messages(self.family)])
        self.assertNotIn("legacy-secret", serialized)
        self.assertNotIn('"risk"', serialized)

    def test_revocation_hides_old_reports_and_messages_and_blocks_mutations(self):
        message_id = store.list_messages(self.doctor)[0]["id"]
        family_message_id = store.list_messages(self.family)[0]["id"]
        store.link_patient("doctor-1", "P1", authorized=False)
        self.assertEqual(store.list_reports(self.doctor), [])
        self.assertEqual(store.list_messages(self.doctor), [])
        for operation in [
            lambda: store.get_report(self.doctor, self.report1),
            lambda: store.get_message(self.doctor, message_id),
            lambda: store.mark_read(self.doctor, message_id),
            lambda: store.update_report_status(self.doctor, self.report1, "closed"),
            lambda: store.repush_report(self.doctor, self.report1),
        ]:
            with self.assertRaises(PermissionError):
                operation()
        store.link_patient("family-1", "P1", authorized=False)
        self.assertEqual(store.list_messages(self.family), [])
        with self.assertRaises(PermissionError):
            store.get_message(self.family, family_message_id)
        with self.assertRaises(PermissionError):
            store.get_report(self.family, self.report1)

    def test_role_change_does_not_disclose_previous_doctor_messages(self):
        token = store.issue_token("doctor-1")
        message_id = store.list_messages(self.doctor)[0]["id"]
        store.update_report_status(self.doctor, self.report1, "reviewed", "former-doctor-note")
        store.provision_user("doctor-1", "family")
        self.assertEqual(store.authenticate_token(token)["role"], "family")
        self.assertEqual(store.list_messages(self.doctor), [])
        with self.assertRaises(PermissionError):
            store.get_message(self.doctor, message_id)
        with self.assertRaises(PermissionError):
            store.update_report_status(self.doctor, self.report1, "closed")
        with self.assertRaises(PermissionError):
            store.get_report(self.doctor, self.report1)
        store.link_patient("doctor-1", "P1")
        self.assertNotIn("doctor-secret", json.dumps(store.get_report(self.doctor, self.report1)))
        store.provision_user("doctor-3", "doctor")
        store.link_patient("doctor-3", "P1")
        store.repush_report({"user_id": "doctor-3"}, self.report1)
        safe_message = store.get_message(self.doctor, message_id)
        self.assertEqual(safe_message["recipient_role"], "family")
        self.assertIn("family_view", safe_message["content"])
        self.assertNotIn("doctor_view", safe_message["content"])
        self.assertNotIn("former-doctor-note", json.dumps(safe_message))

    def test_doctors_keep_individual_notes_and_updated_by_audit(self):
        third = store.provision_user("doctor-3", "doctor")
        store.link_patient("doctor-3", "P1")
        for status in ("new", "reviewed", "followup_required", "closed"):
            result = store.update_report_status(self.doctor, self.report1, status, "doctor-one-note")
            self.assertEqual(result["status"], status)
            self.assertEqual(result["updated_by"], "doctor-1")
        other_view = store.get_report(third, self.report1)
        self.assertEqual(other_view["status"], "new")
        self.assertEqual(other_view["notes"], "")
        store.update_report_status(third, self.report1, "reviewed", "doctor-three-note")
        self.assertEqual(store.get_report(self.doctor, self.report1)["notes"], "doctor-one-note")
        self.assertEqual(store.get_report(third, self.report1)["notes"], "doctor-three-note")
        for status in ("anything", "NEW", "delete"):
            with self.assertRaises(ValueError):
                store.update_report_status(self.doctor, self.report1, status)

    def test_repeated_delivery_and_repush_are_idempotent_and_preserve_read_state(self):
        self.assertEqual(len(self.query("SELECT id FROM report_messages WHERE report_id=?", (self.report1,))), 2)
        message = store.list_messages(self.doctor)[0]
        read = store.mark_read(self.doctor, message["id"])
        self.assertEqual(read["status"], "read")
        self.assertTrue(read["read_at"])
        self.assertEqual(store.list_messages(self.doctor, unread_only=True), [])
        for _ in range(3):
            pushed = store.repush_report(self.doctor, self.report1)
            self.assertEqual(pushed["created_messages"], 0)
        reread = store.get_message(self.doctor, message["id"])
        self.assertEqual(reread["read_at"], read["read_at"])
        self.assertEqual(reread["status"], "read")
        self.assertEqual(len(self.query("SELECT id FROM report_messages WHERE report_id=?", (self.report1,))), 2)
        store.provision_user("family-2", "family")
        store.link_patient("family-2", "P1")
        self.assertEqual(store.repush_report(self.doctor, self.report1)["created_messages"], 1)
        self.assertEqual(len(store.list_messages({"user_id": "family-2"})), 1)

    def test_delivery_patient_identity_and_transaction_rollback(self):
        before = len(self.query("SELECT id FROM reports"))
        conn = connection.get_connection(self.db_path)
        try:
            with self.assertRaises(ValueError):
                with conn:
                    conn.execute("INSERT INTO followup_sessions(session_id,patient_id) VALUES ('bad','P1')")
                    report_id = conn.execute("INSERT INTO reports(session_id,report_json) VALUES ('bad','{}')").lastrowid
                    store.persist_report_delivery(conn, report_id, "P2", {})
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0], before)
            self.assertIsNone(conn.execute("SELECT session_id FROM followup_sessions WHERE session_id='bad'").fetchone())
        finally:
            conn.close()

    def test_token_notes_and_read_state_survive_new_connections(self):
        token = store.issue_token("doctor-1")
        message_id = store.list_messages(self.doctor)[0]["id"]
        read = store.mark_read(self.doctor, message_id)
        store.update_report_status(self.doctor, self.report1, "followup_required", "durable-note")
        connection.init_db(self.db_path)
        restored_user = store.authenticate_token(token)
        self.assertEqual(restored_user, {"user_id": "doctor-1", "role": "doctor"})
        restored = store.get_report(restored_user, self.report1)
        self.assertEqual(restored["status"], "followup_required")
        self.assertEqual(restored["notes"], "durable-note")
        self.assertEqual(store.get_message(restored_user, message_id)["read_at"], read["read_at"])


if __name__ == "__main__":
    unittest.main()

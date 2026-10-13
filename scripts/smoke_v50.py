"""Reproducible isolated business-chain validation, with explicit model mode."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("stable", "high_bp", "nonadherent", "unclear", "danger", "refusing"), default="stable")
    parser.add_argument("--mode", choices=("live", "scripted"), default="live")
    parser.add_argument("--env-file")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    os.environ.setdefault("RAG_DISABLE_EMBEDDING", "1")
    from backend.simulation_v5 import SimulationManager
    from database.connection import using_database
    from database import repository, collaboration
    with tempfile.TemporaryDirectory(prefix="medical-v50-smoke-") as directory:
        manager = SimulationManager(directory)
        created = manager.create(args.scenario, model_mode=args.mode, delay_seconds=0, background=False)
        print(json.dumps({"event": "started", "scenario": args.scenario, "model_mode": args.mode}), flush=True)
        result = manager.execute(created["run_id"])
        view = manager.view(created["run_id"], created["run_token"])
        summary = {key: result.get(key) for key in ("scenario_id", "status", "followup_status", "evaluation", "error_type", "error")}
        summary.update(model_mode=args.mode, real_model_requests=view["real_model_requests"], trace=view.get("trace", []),
                       transcript=result["transcript"], task=result.get("task"), doctor_handled=False, family_read=False)
        with using_database(manager.store.database(created["run_id"])):
            doctor = collaboration.authenticate_token(created["doctor_token"])
            family = collaboration.authenticate_token(created["family_token"])
            reports = collaboration.list_reports(doctor)
            if reports:
                report = reports[0]
                handled = collaboration.update_report_status(doctor, report["id"], "closed", "虚构患者演示报告已核对，按原医嘱与实际就医安排处理。")
                summary["doctor_handled"] = handled["status"] == "closed"
                summary["report"] = report["report"]
                duplicate = collaboration.repush_report(doctor, report["id"])
                summary["duplicate_push_created_messages"] = duplicate["created_messages"]
            messages = collaboration.list_messages(family)
            if messages:
                summary["family_read"] = collaboration.mark_read(family, messages[0]["id"])["status"] == "read"
        # Record only fictional facts and allowlisted trace; credentials are never exported.
        if args.output:
            Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        brief = {key: summary[key] for key in ("scenario_id", "model_mode", "status", "followup_status", "real_model_requests", "doctor_handled", "family_read")}
        brief["score"] = (summary["evaluation"] or {}).get("total_score")
        brief["evaluation_pass"] = (summary["evaluation"] or {}).get("pass")
        print(json.dumps(brief, ensure_ascii=False), flush=True)
        return 0 if summary["status"] == "completed" and summary["doctor_handled"] and summary["family_read"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

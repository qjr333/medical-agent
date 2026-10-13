"""Allowlisted execution events for UI; no prompts, secrets or reasoning text."""
from copy import deepcopy

STAGES = {"USER_MESSAGE": "patient_input", "STATE_UPDATE": "information_extraction",
          "LLM_DECISION": "planner", "V4_TOOL_CALL": "medical_tool", "TOOL_CACHE": "medical_tool",
          "RAG_RETRIEVAL": "rag", "AGENT_RESPONSE": "agent_reply", "TASK_COMPLETION": "completion",
          "COMPLETION_EVALUATION": "completion", "TASK_COMPLETION_EVALUATED": "completion", "REPORT_GENERATED": "report", "MESSAGE_DELIVERY": "notification",
          "SAFETY_OVERRIDE": "safety", "TRIGGER_ENGINE": "trigger", "EXECUTION_RESUMED": "resume"}
STAGES["LLM_VALIDATION_ERROR"] = "planner"
STAGES["TASK_INTERRUPTED"] = "execution_error"
KEYS = {"action", "topic", "tool", "status", "purpose", "ok", "called", "deepseek_called", "duration_ms",
        "risk_level", "next_status", "outcome", "ai_status", "recipient_count", "channel", "external_delivery",
        "trigger_type", "automatic", "cache_hit", "executed", "count", "source_count", "retrieved_count"}


def present_trace(state, *, fictional=False):
    result = []
    for i, entry in enumerate(state.get("trace") or []):
        event = entry.get("event", "")
        detail = entry.get("detail") or {}
        if not isinstance(detail, dict):
            continue
        stage = STAGES.get(event)
        if event == "LLM_CALL":
            stage = {"EXTRACT": "information_extraction", "PLAN": "planner", "COMPLETE": "completion",
                     "REPORT_SUMMARY": "report", "SIMULATED_PATIENT": "simulated_patient"}.get(detail.get("purpose"), "model")
        if stage is None:
            continue
        safe = {k: deepcopy(v) for k, v in detail.items() if k in KEYS and isinstance(v, (str, int, float, bool, type(None)))}
        if event == "LLM_CALL":
            safe["provider_request_count"] = sum(1 for p in detail.get("provider_events", [])
                                                 if isinstance(p, dict) and p.get("status") == "started")
        if fictional and event in {"USER_MESSAGE", "AGENT_RESPONSE"}:
            safe["message"] = str(detail.get("message", ""))[:3000]
        # Retrieval sources are pulled from actual persisted evidence only.
        if stage == "rag":
            safe["sources"] = []
            for source in detail.get("sources", []):
                if not isinstance(source, dict):
                    continue
                item = {k: v[:200] if isinstance(v, str) else v for k, v in source.items()
                        if k in {"source_id", "block_id", "source_title", "year", "source_pages"}
                        and isinstance(v, (str, int, float, bool, type(None)))}
                pages = source.get("source_pages")
                if isinstance(pages, list):
                    item["source_pages"] = [p for p in pages[:20] if isinstance(p, int) or
                                            isinstance(p, str) and len(p) <= 30 and all(c in "0123456789-–—,页 pP" for c in p)]
                safe["sources"].append(item)
        result.append({"id": i, "time": entry.get("time"), "event": event, "stage": stage, "detail": safe})
    return result

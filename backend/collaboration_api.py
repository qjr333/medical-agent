"""Authenticated station report/message APIs, independent of patient chat UI."""

from copy import deepcopy
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict

from backend.access import authenticated_user, require_patient_access, require_role
from database import collaboration, repository

router = APIRouter(prefix="/api", tags=["Reports and collaboration"])


def _operation(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except PermissionError:
        raise HTTPException(status_code=403, detail="未获授权执行该操作") from None
    except LookupError:
        raise HTTPException(status_code=404, detail="报告或消息不存在") from None
    except ValueError:
        raise HTTPException(status_code=422, detail="数据或处理状态无效") from None


class HandlingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["new", "reviewed", "followup_required", "closed"]
    notes: str = Field(default="", max_length=4000)


@router.get("/me")
def me(user=Depends(authenticated_user)):
    return user


@router.get("/reports")
def reports(patient_id: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200),
            user=Depends(authenticated_user)):
    return {"items": _operation(collaboration.list_reports, user, patient_id=patient_id, limit=limit)}


@router.get("/reports/{report_id}")
def report(report_id: int, user=Depends(authenticated_user)):
    return _operation(collaboration.get_report, user, report_id)


@router.patch("/reports/{report_id}/handling")
def handle_report(report_id: int, request: HandlingRequest, user=Depends(authenticated_user)):
    require_role(user, "doctor")
    return _operation(collaboration.update_report_status, user, report_id, request.status, request.notes)


@router.post("/reports/{report_id}/push")
def push_report(report_id: int, user=Depends(authenticated_user)):
    require_role(user, "doctor")
    return _operation(collaboration.repush_report, user, report_id)


@router.get("/messages")
def messages(unread_only: bool = False, limit: int = Query(default=50, ge=1, le=200),
             user=Depends(authenticated_user)):
    return {"items": _operation(collaboration.list_messages, user, unread_only=unread_only, limit=limit)}


@router.get("/messages/{message_id}")
def message(message_id: int, user=Depends(authenticated_user)):
    return _operation(collaboration.get_message, user, message_id)


@router.post("/messages/{message_id}/read")
def read_message(message_id: int, user=Depends(authenticated_user)):
    return _operation(collaboration.mark_read, user, message_id)


@router.post("/reports/session/{session_id}/generate")
def retry_report(session_id: str, user=Depends(authenticated_user)):
    """Recover a pending background report without reopening a closed follow-up."""
    require_role(user, "doctor")
    with repository.SESSION_LOCK:
        state = repository.read_session(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        require_patient_access(user, state["patient_id"], "doctor")
        if state.get("agent_version") != "V4.2" or state.get("followup_status") not in {
                "completed", "partially_completed", "escalated"}:
            raise HTTPException(status_code=409, detail="随访尚未达到报告生成条件")
        from agent.reporting_v42 import build_report_v42
        state = deepcopy(state)
        # Retries recover missing fact reports; they never rerun an urgent LLM.
        # Existing generated reports remain identical and delivery is idempotent.
        build_report_v42(state)
        state["report_generation"] = {"status": "generated"}
        repository.persist_session(session_id, state)
        rows = _operation(collaboration.list_reports, user, patient_id=state["patient_id"], limit=200)
        return next(row for row in rows if row["session_id"] == session_id)

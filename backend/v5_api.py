"""V5 demo, trace and proactive APIs; existing collaboration routes are reused."""
import os
import threading
from pathlib import Path
from typing import Literal, Optional, List

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from agent.trace_view import present_trace
from agent.state_manager import add_trace
from backend.access import authenticated_user, require_patient_access, require_role
from backend.proactive import ProactiveEngine
from backend.runtime_v5 import AgentRuntime
from backend.simulation_v5 import SimulationManager
from database import tasks, repository, collaboration
from database.connection import current_database_path, using_database, get_connection
from simulation.patient import list_scenarios

router = APIRouter(prefix="/api/v5", tags=["V5 proactive and simulation"])
_manager = None
_manager_lock = threading.Lock()
_stop = threading.Event()
_workers = []
_jobs = set()
_job_lock = threading.Lock()


def manager():
    global _manager
    with _manager_lock:
        if _manager is None:
            root = os.getenv("SIMULATION_DATA_DIR") or str(Path(__file__).resolve().parent.parent / "data" / "simulations")
            _manager = SimulationManager(root)
        return _manager


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulationRequest(StrictRequest):
    scenario_id: str
    model_mode: Literal["live", "scripted"] = "live"
    trigger_type: Optional[Literal["scheduled", "high_bp", "missed_measurement"]] = None
    max_turns: int = Field(default=20, ge=1, le=30)
    delay_seconds: int = Field(default=1, ge=0, le=60)


class PlanRequest(StrictRequest):
    patient_id: str
    plan_type: Literal["scheduled", "missed_measurement"]
    due_at: str
    interval_seconds: int = Field(default=86400, ge=1)
    window_seconds: Optional[int] = Field(default=None, ge=1)


class PlanUpdate(StrictRequest):
    enabled: Optional[bool] = None
    due_at: Optional[str] = None
    interval_seconds: Optional[int] = Field(default=None, ge=1)
    window_seconds: Optional[int] = Field(default=None, ge=1)


class BPEventRequest(StrictRequest):
    patient_id: str
    event_id: str = Field(min_length=1, max_length=200)
    systolic: int
    diastolic: int
    measured_at: Optional[str] = None


class MessageRequest(StrictRequest):
    message: str = Field(min_length=1, max_length=4000)


class RunAccess(StrictRequest):
    run_id: str
    run_token: str


class CompareRequest(StrictRequest):
    runs: List[RunAccess] = Field(min_length=1, max_length=20)


def operation(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except PermissionError:
        raise HTTPException(403, "无权访问此运行") from None
    except LookupError:
        raise HTTPException(404, "数据不存在") from None
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


@router.get("/scenarios")
def scenarios():
    return {"items": list_scenarios(), "is_simulation": True}


@router.post("/simulations", status_code=202)
def simulate(request: SimulationRequest):
    if len(manager().store.pending()) >= 8:
        raise HTTPException(429, "演示运行繁忙，请等待当前任务完成")
    return operation(manager().create, **request.model_dump())


@router.get("/simulations/{run_id}")
def simulation(run_id: str, x_demo_token: Optional[str] = Header(default=None)):
    return operation(manager().view, run_id, x_demo_token)


@router.post("/simulations/{run_id}/resume", status_code=202)
def resume_simulation(run_id: str, x_demo_token: Optional[str] = Header(default=None)):
    payload = operation(manager().store.get, run_id, x_demo_token)
    if payload["status"] in {"completed", "budget_exhausted", "failed"}:
        raise HTTPException(409, "此运行已经结束，请启动新运行")
    manager().launch(run_id)
    return {"run_id": run_id, "status": "queued"}


@router.post("/evaluations/compare")
def compare(request: CompareRequest):
    items = []
    for run in request.runs:
        payload = operation(manager().view, run.run_id, run.run_token)
        items.append({key: payload.get(key) for key in
                      ("run_id", "scenario_id", "status", "options", "followup_status", "real_model_requests", "evaluation")})
    return {"items": items}


def allowed_session(sid, user):
    state = repository.read_session(sid)
    if state is None:
        raise HTTPException(404, "会话不存在")
    require_patient_access(user, state["patient_id"], "doctor")
    return state


@router.get("/sessions/{sid}/trace")
def trace(sid: str, user=Depends(authenticated_user)):
    state = allowed_session(sid, user)
    return {"session_id": sid, "items": present_trace(state), "followup_status": state["followup_status"]}


@router.get("/sessions/{sid}")
def session(sid: str, user=Depends(authenticated_user)):
    state = allowed_session(sid, user)
    return {"session_id": sid, "followup_status": state["followup_status"], "message": state.get("last_agent_response", ""),
            "question_context": state.get("question_context", {}), "report_generation": state.get("report_generation", {})}


def dispatch(function, key):
    path = current_database_path()
    key = (str(path), key)
    with _job_lock:
        if key in _jobs:
            return False
        _jobs.add(key)
    def execute():
        try:
            with using_database(path):
                function()
        except Exception as error:
            # A 202 acknowledgement is never converted to a silent successful turn.
            with using_database(path):
                sid = key[1].split(":", 1)[1] if key[1].startswith(("turn:", "recover:")) else None
                if sid:
                    try:
                        with repository.session_lock(sid):
                            state = repository.read_session(sid)
                            if state:
                                if state.get("followup_status") in {"completed", "partially_completed", "escalated"}:
                                    state["report_generation"] = {"status": "failed", "error_type": type(error).__name__}
                                else:
                                    state.update(followup_status="interrupted", interruption_reason="background_execution_failed",
                                                 last_agent_response="本次处理未完成，当前记录已保留，请稍后重试。")
                                add_trace(state, "TASK_INTERRUPTED", {"reason": "background_execution_failed", "error_type": type(error).__name__})
                                repository.persist_session(sid, state)
                    except Exception:
                        pass
            print("[V5] Background execution:", type(error).__name__)
        finally:
            with _job_lock:
                _jobs.discard(key)
    threading.Thread(target=execute, daemon=True, name="v5-background").start()
    return True


@router.post("/sessions/{sid}/messages", status_code=202)
def patient_message(sid: str, request: MessageRequest, user=Depends(authenticated_user)):
    state = allowed_session(sid, user)
    if state["followup_status"] not in {"waiting_for_user", "interrupted"}:
        raise HTTPException(409, "会话未等待新输入")
    if not dispatch(lambda: AgentRuntime().turn(sid, request.message), "turn:" + sid):
        raise HTTPException(409, "本会话正在处理上一条输入，请等待结果")
    return {"session_id": sid, "status": "processing"}


@router.post("/plans")
def create_plan(request: PlanRequest, user=Depends(authenticated_user)):
    require_patient_access(user, request.patient_id, "doctor")
    return operation(tasks.create_plan, **request.model_dump())


@router.patch("/plans/{plan_id}")
def update_plan(plan_id: str, request: PlanUpdate, user=Depends(authenticated_user)):
    plan = tasks.get_plan(plan_id)
    if plan is None:
        raise HTTPException(404, "计划不存在")
    require_patient_access(user, plan["patient_id"], "doctor")
    return operation(tasks.update_plan, plan_id, **request.model_dump(exclude_none=True))


@router.get("/plans")
def plans(patient_id: str, user=Depends(authenticated_user)):
    require_patient_access(user, patient_id, "doctor")
    return {"items": tasks.list_plans(patient_id)}


@router.post("/bp-events")
def bp_event(request: BPEventRequest, user=Depends(authenticated_user)):
    require_patient_access(user, request.patient_id, "doctor")
    return operation(tasks.record_bp_event, **request.model_dump())


@router.get("/tasks")
def task_list(patient_id: str, user=Depends(authenticated_user)):
    require_patient_access(user, patient_id, "doctor")
    return {"items": tasks.list_tasks(patient_id)}


@router.post("/engine/tick", status_code=202)
def tick(user=Depends(authenticated_user)):
    require_role(user, "doctor")
    conn = get_connection()
    try:
        patients = [row[0] for row in conn.execute("SELECT patient_id FROM patient_access WHERE user_id=? AND authorized=1", (user["user_id"],))]
    finally:
        conn.close()
    # Only tasks belonging to the authenticated doctor's patient grants may be executed by this request.
    def execute_task(task):
        collaboration.require_patient_access(user, task["patient_id"], roles=("doctor",))
        return AgentRuntime().start_task(task)
    # A manual tick is an administrative demo operation, not a clock override.
    dispatch(lambda: ProactiveEngine(execute_task, max_tasks_per_tick=1, patient_ids=patients).tick(), "tick:" + user["user_id"])
    return {"status": "processing", "external_delivery": "not_configured"}


async def simulation_scope(request, call_next):
    run_id = request.headers.get("X-Simulation-Run-ID")
    if not run_id:
        return await call_next(request)
    try:
        path = manager().store.database(run_id)
        with using_database(path):
            auth = request.headers.get("Authorization", "")
            user = collaboration.authenticate_token(auth[7:]) if auth.startswith("Bearer ") else None
            if user is None:
                manager().store.get(run_id, request.headers.get("X-Demo-Token"))
            return await call_next(request)
    except PermissionError:
        return JSONResponse(status_code=403, content={"detail": "模拟运行授权无效"})
    except LookupError:
        return JSONResponse(status_code=404, content={"detail": "模拟运行不存在"})


def start_workers():
    global _stop
    if os.getenv("LLM_AGENT_V50", "0") != "1":
        return
    if _workers:
        return
    path = current_database_path()
    stop = threading.Event()
    _stop = stop
    manager().recover()
    def work():
        with using_database(path):
            engine = ProactiveEngine(AgentRuntime().start_task, max_tasks_per_tick=1)
            while not stop.is_set():
                try:
                    engine.tick()
                    recover_pending_sessions()
                except Exception as error:
                    print("[V5] Task worker:", type(error).__name__)
                stop.wait(1)
    for _ in range(4):
        thread = threading.Thread(target=work, daemon=True, name="proactive-followup")
        _workers.append(thread)
        thread.start()


def recover_pending_sessions():
    """Resume durable execution/report work after crashes; no synthetic patient input."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT session_id,state_json FROM followup_sessions WHERE status IN ('processing','interrupted','completed','partially_completed','escalated') ORDER BY updated_at DESC LIMIT 200").fetchall()
    finally:
        conn.close()
    import json
    for row in rows:
        state = json.loads(row["state_json"])
        if state.get("agent_version") != "V5.0":
            continue
        verification_pending = (state.get("followup_status") == "interrupted"
                                and (state.get("extraction_pending") or {}).get("status") == "awaiting_verification")
        pending = verification_pending or state.get("followup_status") == "processing" or (state.get("report_generation") or {}).get("status") in {"pending", "summary_pending", "failed"}
        if not pending or state.get("background_recovery_attempts", 0) >= 3:
            continue
        sid = row["session_id"]
        def recover(session_id=sid):
            with repository.session_lock(session_id):
                fresh = repository.read_session(session_id)
                if not fresh or fresh.get("background_recovery_attempts", 0) >= 3:
                    return
                verifying = (fresh.get("followup_status") == "interrupted"
                             and (fresh.get("extraction_pending") or {}).get("status") == "awaiting_verification")
                if verifying or fresh.get("followup_status") == "processing" or (fresh.get("report_generation") or {}).get("status") in {"pending", "summary_pending", "failed"}:
                    fresh["background_recovery_attempts"] = fresh.get("background_recovery_attempts", 0) + 1
                    repository.persist_session(session_id, fresh)
                    runtime = AgentRuntime()
                    if verifying or fresh["followup_status"] == "processing":
                        runtime.turn(session_id)
                    else:
                        runtime.finalize(session_id)
                    result = runtime.task_result(session_id)
                    tasks.sync_session_status(session_id, result["status"], report_generated=result["report_generated"], station_messages=result["station_messages"])
        dispatch(recover, "recover:" + sid)


def stop_workers():
    _stop.set()
    for thread in _workers:
        thread.join(timeout=1)
    _workers.clear()

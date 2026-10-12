"""FastAPI 入口：应用初始化、兼容适配和原有 API 路由。"""

import os
import uuid
import copy
import sqlite3
from time import perf_counter
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import (
    BASE_DIR, STATIC_DIR, INDEX_FILE, DEMO_DATA_FILE,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    load_demo_data as _load_demo_data,
)
from backend import model_client
from backend.model_client import parse_json_text
from backend.utils import now_iso, to_json, unique_list
from database.connection import get_connection, init_db
from database import repository
from database.repository import (
    sessions, SESSION_LOCK, get_recent_bp_records, get_recent_medication_records,
    read_session, persist_session,
)
from memory.patient_memory import (
    parse_storage_time, summarize_bp_window, summarize_medication_window,
    extract_followup_summary, read_window_observations, load_patient_memory,
    memory_context_text, refresh_patient_memory,
)
from agent.state_manager import (
    create_patient_state as _create_patient_state, merge_user_patch, add_trace, ensure_dialogue_state,
)
from agent.extractor import (
    COMMON_SYMPTOMS, extract_bp_from_text, detect_medication_status,
    detect_symptoms, looks_like_patient_question,
    split_answer_clauses,
    extract_user_patch as _extract_user_patch,
)
from agent.planner import get_available_actions, select_next_action
from agent.question_planner import generate_question
from agent.reporting import build_urgent_message, generate_report, build_finish_message
from agent.runner import run_agent as _run_agent
from medical_tools.safety import (
    HIGH_BP_SYS, HIGH_BP_DIA, ATTENTION_BP_SYS, ATTENTION_BP_DIA, DANGER_PATTERNS,
    phrase_is_negated, detect_danger_signs, contains_red_flag_language,
    valid_bp_measurement, tool_assess_risk,
)
from medical_tools.bp_trend import tool_check_bp_history
from medical_tools.medication import (
    is_medication_question, contains_unsafe_medication_instruction,
    tool_medication_guidance as _tool_medication_guidance,
)
from medical_tools.visit_advice import tool_generate_visit_advice as _tool_generate_visit_advice
from medical_tools.guidance import build_guidance_query, tool_retrieve_medical_guidance, execution_fingerprint
from medical_tools.alerts import tool_create_alert
from medical_tools.registry import TOOL_REGISTRY
from rag import retrieve_medical_guidance, get_knowledge_status, rebuild_index


def load_demo_data():
    return _load_demo_data(DEMO_DATA_FILE)


DEMO_DATA = load_demo_data()
DEMO_PATIENT = DEMO_DATA["patient"]
DEMO_BP_HISTORY = DEMO_DATA["bp_history"]
TRIGGER_CONFIGS = DEMO_DATA["trigger_configs"]
client = model_client.client


def call_json_model(system_prompt: str, user_prompt: str, temperature: float = 0.1):
    """保留原模型调用入口，业务实现集中在 backend.model_client。"""
    return model_client.call_json_model(
        system_prompt, user_prompt, temperature,
        client_override=client, model_name=DEEPSEEK_MODEL,
    )


def create_patient_state(trigger_type: str):
    return _create_patient_state(
        trigger_type, DEMO_PATIENT=DEMO_PATIENT,
        DEMO_BP_HISTORY=DEMO_BP_HISTORY, TRIGGER_CONFIGS=TRIGGER_CONFIGS,
    )


def extract_user_patch(text: str, patient_state=None):
    return _extract_user_patch(
        text, patient_state, call_json_model=call_json_model,
        contains_red_flag_language=contains_red_flag_language,
    )


def tool_medication_guidance(state: dict):
    return _tool_medication_guidance(state, call_json_model=call_json_model)


def tool_generate_visit_advice(state: dict):
    return _tool_generate_visit_advice(state, call_json_model=call_json_model)


def run_agent(state: dict):
    return _run_agent(state, call_json_model=call_json_model)


def seed_demo_patient():
    return repository.seed_demo_patient(DEMO_PATIENT, DEMO_BP_HISTORY)




app = FastAPI(
    title="Elderly Hypertension Follow-up Agent",
    version="1.6.1-demo-json",
    description="老年高血压主动随访 Agent Demo",
)


@app.exception_handler(sqlite3.Error)
async def database_unavailable(request, error):
    return JSONResponse(status_code=503, content={
        "detail": "数据库暂时不可用，尚未确认本次数据保存成功，请稍后重试。",
    })


def _refresh_memory_safely(state, session_id=None):
    start = perf_counter()
    try:
        return refresh_patient_memory(state, session_id)
    except sqlite3.Error as error:
        state["patient_memory"] = {"version": "2.0", "status": "unavailable"}
        state["memory_context"] = "本次历史记录读取失败，不能据此判断既往没有问题。"
        state["bp_history"] = []
        entry = {"tool": "PATIENT_MEMORY", "error_type": type(error).__name__, "recoverable": True}
        if entry not in state.setdefault("task_errors", []):
            state["task_errors"].append(entry)
        state["task_errors"] = state["task_errors"][-8:]
        add_trace(state, "TOOL_ERROR", {**entry, "duration_ms": round((perf_counter() - start) * 1000, 3), "next_status": "active"})
        return state["patient_memory"]


def _persist_safely(session_id, state, **observations):
    start = perf_counter()
    try:
        return persist_session(session_id, state, **observations)
    except sqlite3.Error as error:
        entry = {"tool": "SQLITE_PERSIST", "error_type": type(error).__name__, "recoverable": False}
        if entry not in state.setdefault("task_errors", []):
            state["task_errors"].append(entry)
        state["task_errors"] = state["task_errors"][-8:]
        add_trace(state, "TOOL_ERROR", {**entry, "duration_ms": round((perf_counter() - start) * 1000, 3), "next_status": "interrupted"})
        state["report"] = None
        state["interruption_reason"] = "database_write_failed"
        tool_assess_risk(state)
        sessions.pop(session_id, None)
        detail = "数据库暂时不可用，尚未确认本次数据保存成功，请稍后重试。"
        if state.get("risk_level") == "high":
            state["followup_status"] = "escalated"
            # 写失败也保留进程内安全封闭状态，避免下一轮重置为普通随访。
            sessions[session_id] = state
            detail = build_urgent_message(state) + detail
        else:
            state["followup_status"] = "interrupted"
        raise HTTPException(status_code=503, detail=detail) from None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if os.path.exists(
    STATIC_DIR
):

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR
        ),
        name="static",
    )


class TriggerRequest(
    BaseModel
):

    trigger_type: str = "scheduled"


class ChatRequest(
    BaseModel
):

    session_id: str

    message: str


@app.on_event("startup")
def initialize_local_database():
    init_db()
    seed_demo_patient()
    print("[DB] SQLite已初始化；会话、血压、用药及Agent Trace可持久化。")


@app.get("/")
def root():

    if os.path.exists(
        INDEX_FILE
    ):

        return FileResponse(
            INDEX_FILE
        )


    return HTMLResponse(
        """
        <html>
            <body>
                <h2>Elderly Hypertension Follow-up Agent</h2>
                <p>Frontend index.html not found.</p>
                <p>Open <a href="/docs">/docs</a> to test APIs.</p>
            </body>
        </html>
        """
    )


@app.get("/api/demo")
def get_demo():

    return {

        "patient":
            DEMO_PATIENT,

        "bp_history":
            DEMO_BP_HISTORY,

        "triggers": [

            {

                "type":
                    key,

                "label":
                    value[
                        "label"
                    ],

                "description":
                    value[
                        "followup_reason"
                    ],
            }

            for key, value
            in TRIGGER_CONFIGS.items()
        ],
    }


@app.post("/api/trigger")
def trigger_followup(request: TriggerRequest):
    trigger_type = request.trigger_type
    if trigger_type not in TRIGGER_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown trigger_type: {trigger_type}")

    with SESSION_LOCK:
        session_id = str(uuid.uuid4())
        state = create_patient_state(trigger_type)
        memory = _refresh_memory_safely(state)
        add_trace(state, "MEMORY_LOADED", {
            "version": memory.get("version"),
            "bp_records_7d": (memory.get("blood_pressure") or {}).get("last_7_days", {}).get("measurement_count", 0),
            "medication_issue_reports_7d": (memory.get("medication") or {}).get("last_7_days", {}).get("issue_report_count", 0),
            "prior_followup": (memory.get("last_followup") or {}).get("session_id"),
            "focus": memory.get("focus"),
        })
        add_trace(state, "TRIGGER_ENGINE", {
            "trigger_type": trigger_type,
            "reason": state["followup_reason"],
        })

        # 在运行Agent前保存初始会话，防止中间意外时完全丢失。
        initial_bp = None
        if valid_bp_measurement(state.get("bp_systolic"), state.get("bp_diastolic")):
            initial_bp = {
                "systolic": state["bp_systolic"],
                "diastolic": state["bp_diastolic"],
                "source": "demo_trigger",
            }
        _persist_safely(session_id, state, new_bp=initial_bp)
        _refresh_memory_safely(state, session_id)

        response = run_agent(state)
        add_trace(state, "AGENT_RESPONSE", {
            "message": response.get("message", ""),
            "followup_status": state.get("followup_status"),
        })
        _persist_safely(session_id, state)
        response["session_id"] = session_id
        return response


@app.post("/api/chat")
def chat(request: ChatRequest):
    session_id = request.session_id
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    with SESSION_LOCK:
        cached = sessions.get(session_id)
        if cached and cached.get("followup_status") in {"completed", "escalated"}:
            raise HTTPException(status_code=409, detail="该随访已结束或转入高风险升级流程，请发起新随访。")
        try:
            state = read_session(session_id)
        except sqlite3.Error as error:
            # 恢复会话失败也先用原规则核验新输入的危险表现，不能只让患者稍后重试。
            systolic, diastolic = extract_bp_from_text(message)
            if not valid_bp_measurement(systolic, diastolic):
                systolic, diastolic = None, None
            danger = unique_list([label for clause in split_answer_clauses(message)
                                  for label in detect_danger_signs(clause)])
            emergency_state = {"bp_systolic": systolic, "bp_diastolic": diastolic,
                               "red_flags": danger, "symptoms": [], "trace": []}
            tool_assess_risk(emergency_state)
            detail = "数据库暂时不可用，尚未确认本次数据保存成功，请稍后重试。"
            if emergency_state.get("risk_level") == "high":
                held_state = copy.deepcopy(sessions.get(session_id) or emergency_state)
                held_state.update({key: value for key, value in emergency_state.items()
                                   if key != "trace" and value is not None})
                held_state.update(followup_status="escalated", current_goal="safety_escalation", report=None)
                add_trace(held_state, "TOOL_ERROR", {"tool": "SQLITE_READ", "error_type": type(error).__name__,
                                                   "recoverable": False, "duration_ms": 0, "next_status": "escalated"})
                sessions[session_id] = held_state
                detail = build_urgent_message(emergency_state) + detail
            raise HTTPException(status_code=503, detail=detail) from None
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if state.get("followup_status") in {"completed", "escalated"}:
            raise HTTPException(
                status_code=409,
                detail="该随访已结束或转入高风险升级流程，请发起新随访。",
            )

        state = copy.deepcopy(state)
        state["followup_status"] = "active"
        ensure_dialogue_state(state)
        previous_facts = execution_fingerprint(state, "PATIENT_FACTS", query="")
        add_trace(state, "USER_MESSAGE", {"message": message})
        try:
            patch = extract_user_patch(message, state)
        except Exception as error:
            entry = {"tool": "EXTRACTOR", "error_type": type(error).__name__, "recoverable": True}
            state.setdefault("task_errors", []).append(entry)
            add_trace(state, "TOOL_ERROR", {**entry, "duration_ms": 0, "next_status": "active"})
            patch = _extract_user_patch(message, state, call_json_model=lambda **kwargs: {})

        # 拦截明显错误的测量数值；避免把无效读数写入状态或数据库。
        bp_sys = patch.get("bp_systolic")
        bp_dia = patch.get("bp_diastolic")
        new_bp = None
        if bp_sys is not None or bp_dia is not None:
            if valid_bp_measurement(bp_sys, bp_dia):
                new_bp = {
                    "systolic": bp_sys,
                    "diastolic": bp_dia,
                    "source": "patient_chat",
                }
            else:
                patch["bp_systolic"] = None
                patch["bp_diastolic"] = None
                add_trace(state, "INVALID_BP_INPUT", {"message": message})

        merge_user_patch(state, patch, message)
        new_medication = [
            {"status": item["status"], "medication_name": None,
             "detail": f"服药日期：{item['date']}；{item.get('detail') or message}"}
            for item in patch.get("medication_observations", [])
            if item.get("status") in {"taken", "not_taken_yet", "missed", "irregular", "stopped"}
        ]

        # 新的观察结果会改变原有Advice；不能沿用上一轮的建议。
        if execution_fingerprint(state, "PATIENT_FACTS", query="") != previous_facts:
            state["visit_advice"] = None
            state["medication_guidance"] = None
        add_trace(state, "STATE_UPDATE", {
            "patch": patch,
            "coverage": state.get("coverage"),
        })

        # 新测量/用药记录与本轮文本、State一并写入数据库。
        _persist_safely(session_id, state, new_bp=new_bp, new_medication=new_medication)
        _refresh_memory_safely(state, session_id)
        response = run_agent(state)
        add_trace(state, "AGENT_RESPONSE", {
            "message": response.get("message", ""),
            "followup_status": state.get("followup_status"),
        })
        _persist_safely(session_id, state)
        response["session_id"] = session_id
        return response


@app.get("/api/test-rag")
def test_rag(
    query: str,
    age: Optional[int] = 72,
    topic: Optional[str] = None,
):

    test_state = {

        "age":
            age,

        "medication_status":
            "unknown",

        "red_flags":
            [],
    }


    results = (
        retrieve_medical_guidance(

            query,

            patient_state=
                test_state,

            topic_hint=
                topic,

            top_k=
                5,
        )
    )


    return {

        "query":
            query,

        "patient_age":
            age,

        "topic_hint":
            topic,

        "results":
            results,
    }


@app.get(
    "/api/test-medication-guidance"
)
def test_medication_guidance():

    state = {

        "age":
            72,

        "bp_systolic":
            162,

        "bp_diastolic":
            96,

        "medication_status":
            "missed",

        "medication_detail":
            "患者表示今天早上的降压药忘记服用",

        "symptoms":
            [],

        "red_flags":
            [],

        "latest_patient_question":
            (
                "我今天早上的降压药忘了吃，"
                "现在能不能自己多吃一片？"
            ),

        "retrieved_guidance":
            [],

        "knowledge_queries":
            [],

        "knowledge_sources":
            [],

        "last_guidance_query":
            "",
    }


    query = (
        "72岁高血压老人今天忘记服用降压药，"
        "询问现在是否可以自行多吃一片。"
    )


    rag_result = (
        tool_retrieve_medical_guidance(

            state,

            query,
        )
    )


    medication_result = (
        tool_medication_guidance(
            state
        )
    )


    return {

        "query":
            query,

        "rag_results":
            rag_result.get(
                "results"
            ),

        "medication_guidance":
            medication_result,
    }


@app.get(
    "/api/knowledge-status"
)
def knowledge_status():

    return (
        get_knowledge_status()
    )


@app.post(
    "/api/rebuild-knowledge"
)
def rebuild_knowledge():

    rebuild_index()


    return {

        "status":
            "rebuilt",

        "knowledge":
            get_knowledge_status(),
    }


@app.get("/api/debug/session/{session_id}")
def debug_session(session_id: str):
    state = read_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@app.get("/api/db-status")
def db_status():
    """仅用于本地演示/调试；上线前必须增加鉴权。"""
    return repository.get_database_status()


@app.get("/api/patients/{patient_id}/history")
def patient_history(patient_id: str, limit: int = 30):
    """返回数据库中的长期记录。仅用于本地演示，真实患者信息需鉴权。"""
    result = repository.get_patient_history(patient_id, limit)
    if result is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return result


@app.get("/api/patients/{patient_id}/memory")
def patient_memory_api(patient_id: str):
    """只读历史记忆预览。比赛本地演示用；部署时需患者权限校验。"""
    if not repository.patient_exists(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return load_patient_memory(patient_id)


@app.get("/api/health")
def health():

    return {

        "status":
            "ok",

        "version":
            "1.6.1-demo-json",

        "model_configured":
            bool(
                DEEPSEEK_API_KEY
            ),

        "knowledge":
            get_knowledge_status(),
    }


@app.get("/api/test-bp-trend")
def test_bp_trend():

    state = create_patient_state(
        "high_bp"
    )


    result = (
        tool_check_bp_history(
            state
        )
    )


    return {

        "input_history":
            state.get(
                "bp_history"
            ),

        "analysis":
            result,
    }


@app.get("/api/test-visit-advice")
def test_visit_advice():

    state = create_patient_state(
        "high_bp"
    )

    # 1. 先分析血压趋势
    bp_analysis = tool_check_bp_history(
        state
    )

    # 2. 根据趋势重新判断风险
    risk_result = tool_assess_risk(
        state
    )

    # 3. 获取指南证据
    query = (
        "72岁高血压患者，家庭血压近期连续升高，"
        "当前162/96 mmHg，应该如何继续随访以及何时需要联系医生或线下评估"
    )

    rag_result = tool_retrieve_medical_guidance(
        state,
        query,
    )

    # 4. 生成就医建议
    visit_result = tool_generate_visit_advice(
        state
    )

    return {
        "bp_analysis": bp_analysis,
        "risk": risk_result,
        "rag_results": rag_result.get(
            "results",
            []
        ),
        "visit_advice": visit_result,
    }


@app.get("/api/test-symptom-status")
def test_symptom_status():

    cases = [
        ("未回答", "", ""),
        ("明确否认", "都没有", "check_symptoms"),
        ("普通症状", "我今天有点头晕", "check_symptoms"),
        ("危险症状", "我现在剧烈胸痛", "check_symptoms"),
    ]

    results = {}

    for name, message, goal in cases:

        state = create_patient_state("scheduled")
        state["current_goal"] = goal

        if message:
            merge_user_patch(
                state,
                {"answered_topics": []},
                message
            )

        risk = tool_assess_risk(state)

        results[name] = {
            "symptom_status": state["symptom_status"],
            "red_flag_status": state["red_flag_status"],
            "coverage": {
                "current_symptoms": state["coverage"]["current_symptoms"],
                "red_flags": state["coverage"]["red_flags"]
            },
            "risk_level": risk["risk_level"]
        }

    return results

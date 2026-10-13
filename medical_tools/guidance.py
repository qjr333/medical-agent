"""RAG queries and retrieval with the existing per-session result cache."""

import hashlib
import json
from pathlib import Path

import rag
from backend.utils import unique_list
from memory.patient_memory import memory_context_text
from rag import retrieve_medical_guidance


def build_guidance_query(
    state: dict
):

    patient_question = (
        state.get(
            "latest_patient_question"
        )
        or
        ""
    ).strip()


    if patient_question:

        return patient_question


    age = state.get(
        "age"
    )


    sys_bp = state.get(
        "bp_systolic"
    )


    dia_bp = state.get(
        "bp_diastolic"
    )


    medication_status = (
        state.get(
            "medication_status"
        )
    )


    symptoms = (
        state.get(
            "symptoms"
        )
        or
        []
    )


    trend = state.get(
        "bp_trend"
    )


    return (
        f"{age}岁高血压患者，"
        f"当前家庭血压 {sys_bp}/{dia_bp} mmHg，"
        f"近期血压趋势 {trend}，"
        f"服药状态 {medication_status}，"
        f"当前症状 {symptoms}。"
        f"跨随访历史摘要：{memory_context_text(state)}"
        f"请检索适用于老年患者的家庭血压监测、"
        f"用药依从性、随访和就医指导依据。"
    )



def knowledge_revision():
    """会话检索缓存的版本；不加载Embedding，也不改磁盘向量索引。"""
    metadata = getattr(rag, "_cache_metadata", None) or {}
    sources = []
    directory = Path(rag.KNOWLEDGE_DIR)
    paths = sorted(directory.glob("*.md")) + [Path(rag.SOURCE_REGISTRY_PATH)]
    for path in paths:
        try:
            sources.append((str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
        except OSError:
            sources.append((str(path), "unavailable"))
    return _fingerprint({"index": metadata.get("fingerprint"), "sources": sources,
                         "model": rag.EMBEDDING_MODEL_NAME,
                         "disabled": rag.DISABLE_EMBEDDING, "min_score": rag.MIN_SEMANTIC_SCORE})


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    default=str).encode("utf-8")).hexdigest()


def execution_fingerprint(state, action, query=None):
    """只比较影响工具结果的输入，排除Trace/时间/问题次数等易变字段。"""
    if action == "CHECK_BP_HISTORY":
        readings = [(row.get("systolic"), row.get("diastolic"))
                    for row in state.get("bp_history", []) if isinstance(row, dict)]
        current = (state.get("bp_systolic"), state.get("bp_diastolic"))
        if all(type(value) is int for value in current) and (not readings or readings[-1] != current):
            readings.append(current)
        return _fingerprint({"action": action, "readings": readings[-7:]})
    clinical = {key: state.get(key) for key in (
        "age", "conditions", "bp_systolic", "bp_diastolic", "bp_trend",
        "medication_status", "medication_timing", "medication_date", "medication_detail",
        "symptoms", "denied_symptoms", "red_flags", "denied_red_flags",
        "symptom_status", "red_flag_status", "coverage", "additional_information", "measurement_context",
    )}
    memory = state.get("patient_memory") or {}
    memory_facts = {key: memory.get(key) for key in ("blood_pressure", "medication", "last_followup")}
    clinical["memory"] = memory_context_text({"patient_memory": memory_facts})
    clinical["memory_status"] = memory.get("status", "ok")
    clinical["query"] = build_guidance_query(state) if query is None else query
    clinical["knowledge"] = knowledge_revision()
    if action in {"MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE", "ANSWER_PATIENT_QUESTION"}:
        clinical["evidence"] = state.get("retrieved_guidance") or []
    clinical["action"] = action
    return _fingerprint(clinical)


def tool_is_current(state, action, query=None):
    entry = (state.get("tool_execution") or {}).get(action) or {}
    return (entry.get("status") in {"ok", "empty", "cached", "fallback"}
            and entry.get("fingerprint") == execution_fingerprint(state, action, query))


def tool_retrieve_medical_guidance(state: dict, query: str):
    query = (query or "").strip()
    if not query:
        return {"status": "empty_query", "query": "", "results": []}
    signature = execution_fingerprint(state, "RETRIEVE_MEDICAL_GUIDANCE", query)
    if state.get("guidance_fingerprint") == signature:
        return {"status": "cached", "query": query,
                "results": state.get("retrieved_guidance") or []}
    # 新状态/知识版本不能沿用旧证据；查询失败也不能留下上一问题的结果。
    state["retrieved_guidance"] = []
    state["knowledge_sources"] = []
    results = retrieve_medical_guidance(query, patient_state=state, top_k=4)
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        raise ValueError("Invalid retrieval result")
    state["knowledge_queries"] = unique_list((state.get("knowledge_queries") or []) + [query])
    state["retrieved_guidance"] = results
    state["last_guidance_query"] = query
    state["guidance_fingerprint"] = signature
    state["knowledge_sources"] = unique_list([
        {key: item.get(key) for key in ("source_id", "source_title", "year", "source_pages", "block_id")}
        for item in results
    ])
    return {"status": "ok", "query": query, "results": results}

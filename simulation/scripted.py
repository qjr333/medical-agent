"""Explicit offline test/demo model. Never used as a live-model fallback."""
import json

from medical_tools.safety import DANGER_PATTERNS


class ScriptedDemoModel:
    def __call__(self, *, system_prompt, user_prompt, temperature=0.1):
        payload = json.loads(user_prompt)
        if "candidate_facts" in payload:
            return {"response_fact_ids": [f["fact_id"] for f in payload["candidate_facts"]]}
        if "facts" in payload:
            return {s: [f["fact_id"] for f in payload["facts"] if s in f["allowed_sections"]]
                    for s in ("summary_fact_ids", "doctor_attention_fact_ids", "family_assistance_fact_ids")}
        if "evaluation_context" in payload:
            deferred = payload.get("deferred_topics") or {}
            pending = payload.get("backend_pending_items") or []
            messages = payload.get("patient_statements") or []
            plan = next((m for m in reversed(messages) if "明天" in m and "血压" in m), "")
            if plan:
                pending = [p for p in pending if p != "monitoring_plan"]
            partial = payload.get("patient_requested_end") or (pending and all(p in deferred for p in pending))
            result = {"outcome": "partial" if partial else "continue" if pending else "completed",
                      "reason": "离线脚本基于可用事实校验", "pending_items": [], "next_goal": ""}
            if plan:
                result["objective_evidence"] = {"monitoring_plan": plan}
            return result
        if "tool_input" in payload:
            if "用药管理辅助模块" in system_prompt:
                has_details = "氨氯地平" in payload["tool_input"]
                return {"patient_message": "请核对原医嘱，具体漏服处理向医生或药师确认，不要自行调整处方。",
                        "needs_followup": not has_details,
                        "followup_question": "请说明药物名称和医生原来的服用方案。" if not has_details else "",
                        "needs_professional_review": True, "reason": "离线测试回复"}
            return {"level": "contact_clinician", "advice": "请保存近期血压记录，并联系医生评估。"}
        if "trigger_event" not in payload:
            return {"answer_status": "unable" if "记不清" in payload.get("patient_message", "") else
                    "refused" if "不想继续" in payload.get("patient_message", "") else "answered"}
        state = payload["state"]["today_state"]
        coverage = payload.get("coverage") or {}
        deferred = payload.get("deferred_topics") or state.get("deferred_topics") or {}
        tools = {r.get("tool"): r for r in payload.get("previous_tool_results", []) + payload.get("tool_results_this_turn", []) if r.get("tool")}
        def ask(topic, question, goal=None):
            return {"action": "ASK_USER", "topic": topic, "question": question, "goal": goal or topic}
        def tool(action, query=None):
            return {"action": action, "arguments": {"query": query} if query else {}}
        if not coverage.get("red_flags") and "red_flags" not in deferred:
            return ask("red_flags", "现在有没有胸痛、呼吸困难、晕厥、意识异常、突发肢体无力或突发言语异常？")
        for topic, question in (("bp", "今天最近一次血压是多少？"),
                                ("medication", "今天降压药服用了吗，有没有漏服？"),
                                ("current_symptoms", "您现在有没有头晕或其他不舒服？")):
            if not coverage.get(topic) and topic not in deferred and (topic != "bp" or payload["trigger_event"]["type"] != "missed_measurement"):
                return ask(topic, question)
        if payload["trigger_event"]["type"] == "high_bp" and not coverage.get("measurement_context"):
            return ask("measurement_context", "今天测血压前有没有坐着休息，什么时候测量的？")
        if payload["trigger_event"]["type"] == "missed_measurement" and not coverage.get("monitoring_plan"):
            if not any("明天" in r.get("message", "") for r in payload.get("recent_dialogue", [])):
                return ask("additional_information", "接下来计划什么时候恢复测量、记录或上传血压？")
        if payload.get("patient_question") or state.get("medication_status") in {"missed", "irregular"}:
            if "RETRIEVE_MEDICAL_GUIDANCE" not in tools:
                return tool("RETRIEVE_MEDICAL_GUIDANCE", "老年高血压漏服药应核对原医嘱与用药依从性")
            if "MEDICATION_GUIDANCE" not in tools:
                return tool("MEDICATION_GUIDANCE")
            if tools["MEDICATION_GUIDANCE"].get("data", {}).get("needs_followup"):
                return ask("medication", "请说明药物名称与医生原来的服用方案。", "collect_medication_details")
        if coverage.get("bp") and "CHECK_BP_HISTORY" not in tools:
            return tool("CHECK_BP_HISTORY")
        if payload.get("risk_level") == "attention":
            if "RETRIEVE_MEDICAL_GUIDANCE" not in tools:
                return tool("RETRIEVE_MEDICAL_GUIDANCE", "老年高血压家庭监测血压持续偏高就医管理")
            if "GENERATE_VISIT_ADVICE" not in tools:
                return tool("GENERATE_VISIT_ADVICE")
        return {"action": "FINISH", "reply": "本次事实已记录；相关疑问以原医嘱和医生确认意见为准。"}

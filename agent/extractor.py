"""信息提取兼容入口。V5 使用语义提取与证据复核；旧 Agent 保留规则路径。"""

import json
import re
from datetime import date, timedelta

from backend.utils import unique_list


COMMON_SYMPTOMS = [
    "头晕", "头痛", "胸闷", "心慌", "恶心", "乏力", "视物模糊", "耳鸣",
    "胸痛", "呼吸困难", "气短",
]

# 别名只用于信息提取，危险分级仍由原有 Safety Layer 决定。
SYMPTOM_ALIASES = {"呼吸困难": ["喘不上气", "无法呼吸", "呼吸很困难"]}
MEDICATION_STATUSES = {"unknown", "taken", "not_taken_yet", "missed", "irregular", "stopped"}
TOPICS = {"bp", "medication", "current_symptoms", "red_flags", "measurement_context"}


def split_answer_clauses(text: str):
    """否定只在当前分句内生效，不能遮蔽转折后的危险症状。"""
    return [part.strip() for part in re.split(r"[，,。.;；!?！？\n]|但是|不过|然而|可是|只有|只是|现在|当前|此刻|而且|并且|但|却|(?<!没)(?=有(?:头晕|胸痛|呼吸困难|喘不上气|无法呼吸))", text or "") if part.strip()]


def extract_bp_from_text(text: str):
    match = re.search(r"(?<!\d)(\d{2,3})\s*[/／]\s*(\d{2,3})(?!\d)", text or "")
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def _is_negated(clause, start):
    """使用最近的肯定/否定词，兼容一串被否认的症状。"""
    prefix = clause[:start]
    markers = list(re.finditer(r"没有出现|没有|并无|否认|未出现|不伴|无|没|(?<![没无])有|出现|伴随", prefix))
    return bool(markers and markers[-1].group() in {"没有出现", "没有", "并无", "否认", "未出现", "不伴", "无", "没"})


def _term_is_negated(clause, match):
    prefix = clause[:match.start()]
    return bool(re.search(r"(?:没有|并无|否认|不是|没|不曾|未)\s*.{0,4}$", prefix))


def _medication_timing(text):
    # 短答“没到/还没到”允许由本轮服药时间问题的语境消歧。
    # 下游 medication_context 会阻止非服药话题写入药物观察记录。
    compact = re.sub(r"[\s，,。.!！?？]", "", text or "")
    if re.fullmatch(r"(?:还)?没(?:有)?到(?:了|呢|啊|呀|哦)?", compact):
        return "not_due"
    if re.fullmatch(r"(?:已经|都)?(?:过了|超过了|错过了)(?:呢|啊|呀|哦)?", compact):
        return "overdue"
    if re.search(r"不确定|不知道|记不清|不记得|好像|可能|也许", text):
        return "unknown"
    if re.search(r"(?:还没|尚未|没有|没)到(?:.{0,8}(?:吃药|服药|用药).{0,4}|(?:规定|原定)的?)?(?:时间|时候|点)", text):
        return "not_due"
    if re.search(r"(?:已经|已|都)?(?:超过|过了|错过)(?:.{0,8}(?:吃药|服药|用药).{0,4}|(?:规定|原定)的?)?(?:时间|时候|点)|^(?:已经|已)过了(?:呢|啊|了)?$", text):
        return "overdue"
    return "unknown"


def detect_medication_status(text: str):
    """尚未服用与明确漏服分别记录，明确否认漏服不能当作未服药。"""
    found = []
    for clause in split_answer_clauses(text):
        if re.search(r"不确定|记不清|不记得|不知道|好像|可能|也许", clause):
            continue
        timing = _medication_timing(clause)
        missed = re.search(r"忘记|忘了|漏服|漏吃", clause)
        if missed and (missed.group() in {"漏服", "漏吃"} or re.search(r"药|吃|服", clause)) and not re.search(r"吃(?:早饭|午饭|晚饭|饭|零食)", clause) and not _term_is_negated(clause, missed):
            found.append("missed")
            continue
        irregular = re.search(r"不规律|间断服|断断续续", clause)
        if irregular and not _term_is_negated(clause, irregular):
            found.append("irregular")
            continue
        stopped = re.search(r"停药|停止服药|不吃了", clause)
        if stopped and not _term_is_negated(clause, stopped):
            found.append("stopped")
            continue
        # 饭、零食等不能充当降压药的服用状态。
        not_yet = re.search(r"还没(?:有)?(?:吃|服)|尚未(?:吃|服)|没有(?:吃|服)|没(?:吃|服)|未服", clause)
        if (not_yet and not re.search(r"(?:吃|服)(?:早饭|午饭|晚饭|饭|零食)", clause)) or (timing == "not_due" and re.search(r"药|服用", clause)):
            found.append("not_taken_yet")
            continue
        taken = re.search(r"吃了|服了|吃过了|服用过|服用了|按时(?:吃|服药)|正常服药|照常服药", clause)
        if taken and not _term_is_negated(clause, taken) and not re.search(r"吃(?:了|过了)(?:早饭|午饭|晚饭|饭|零食)", clause):
            found.append("taken")
    # 同日先报告漏服再补服，仍保留漏服报告；不同日期由 observations 分开。
    for status in ("missed", "stopped", "irregular", "not_taken_yet", "taken"):
        if status in found:
            return status
    return "unknown"


def detect_symptoms(text: str):
    positive, denied = [], []
    for clause in split_answer_clauses(text):
        if re.search(r"不确定|记不清|不知道|不记得|不能确定", clause):
            continue
        for symptom in COMMON_SYMPTOMS:
            for term in [symptom] + SYMPTOM_ALIASES.get(symptom, []):
                for match in re.finditer(re.escape(term), clause):
                    (denied if _is_negated(clause, match.start()) else positive).append(symptom)
    positive = unique_list(positive)
    return positive, [symptom for symptom in unique_list(denied) if symptom not in positive]


def looks_like_patient_question(text: str):
    return any(marker in (text or "") for marker in ["?", "？", "能不能", "可不可以", "可以吗", "要不要", "怎么办", "怎么做", "应该", "是否", "为什么", "怎么", "如何"])


def _question_context(state):
    context = dict(state.get("question_context") or {})
    goal = context.get("goal") or state.get("current_goal") or ""
    question = context.get("question") or state.get("last_agent_question") or ""
    context.setdefault("goal", goal)
    context.setdefault("question", question)
    if not context.get("kind"):
        if goal == "check_medication":
            if re.search(r"超过|还没到|服药时间", question):
                context["kind"] = "timing"
            elif re.search(r"漏服|停药|忘记", question) and not re.search(r"是否.*服用|按.*安排.*服|服用了", question):
                context["kind"] = "adherence"
            else:
                context["kind"] = "medication_taken"
        else:
            context["kind"] = {"collect_bp": "bp", "check_symptoms": "symptoms", "check_measurement_context": "measurement_context"}.get(goal, "")
    if not context.get("topic"):
        context["topic"] = {"bp": "bp", "symptoms": "current_symptoms", "measurement_context": "measurement_context"}.get(context["kind"], "medication" if context["kind"].startswith("medication") or context["kind"] in {"adherence", "timing", "taken_clarify"} else "")
    return context


def _short_answer(text):
    clean = re.sub(r"[\s，。,.!?？！]", "", text or "")
    if re.fullmatch(r"(?:没有不舒服|都没有不舒服|没有不适|没什么不舒服)(?:呢|啊|呀|哦|了|的)?", clean):
        return "no_symptoms"
    if re.fullmatch(r"(?:没有|没|无|都没有)(?:呢|啊|呀|哦|了|的)?", clean):
        return "no"
    if re.fullmatch(r"(?:是|是的|有|有的|对|对的|嗯|是啊|是呀|没错|是这样)(?:呢|啊|呀|哦)?", clean):
        return "yes"
    if re.search(r"不确定|不知道|记不清|不记得|说不清", clean):
        return "uncertain"
    return ""


def _answer_availability(text):
    """拒答和无法作答描述对话意愿，不能作为症状/服药的否认。"""
    # 患者明确“记不清/不知道”时，立即标记无法提供；不把它当成普通模糊回答反复澄清。
    # 仅匹配独立、无其他事实的短答，避免吞掉“我不知道，但现在喘不上气”等危险症状。
    compact = re.sub(r"[\s，,。.!！?？]", "", text or "")
    if re.fullmatch(
        r"(?:我)?(?:真的|实在|确实|也)?"
        r"(?:不知道|不清楚|不太清楚|记不清(?:楚)?|不记得|记不得|想不起来|忘记了|忘了)"
        r"(?:了|呢|啊|呀|哦)?",
        compact,
    ):
        return "unable"
    if re.search(r"拒绝回答|不想回答|不愿(?:意)?回答|不想说|不愿(?:意)?说|不告诉你|先不回答|先不说|别(?:再)?问了|不要再问", text or ""):
        return "refused"
    if re.search(r"无法回答|不能回答|回答不了|没法回答|无法(?:测|量)|不能(?:测|量)|不会(?:测|量)|(?:测|量)不了", text or ""):
        return "unable"
    return ""


def _today():
    # 日期使用项目统一的本地日期；支持旧兼容环境中单独导入提取模块。
    from backend import utils
    helper = getattr(utils, "today_local_date", None)
    return date.fromisoformat(helper()) if helper else date.today()


def _medication_observations(text, today, medication_context=False):
    observations = []
    current_date = today
    for clause in split_answer_clauses(text):
        # 同一分句出现新的日期时切开，避免昨天漏服覆盖今天已服。
        segments = re.split(r"(?=今天|昨天|前天|明天|\d{4}-\d{2}-\d{2})", clause)
        for segment in segments:
            if not segment:
                continue
            explicit = re.search(r"\d{4}-\d{2}-\d{2}", segment)
            if explicit:
                try:
                    current_date = date.fromisoformat(explicit.group())
                except ValueError:
                    continue
            else:
                relative = re.search(r"今天|昨天|前天|明天", segment)
                if relative:
                    current_date = today + timedelta(days={"今天": 0, "昨天": -1, "前天": -2, "明天": 1}[relative.group()])
            status = detect_medication_status(segment)
            segment_timing = _medication_timing(segment)
            prior_not_taken = any(item["date"] == current_date.isoformat() and item["status"] == "not_taken_yet" for item in observations)
            # 同一原话先回答未服用、后补充明确时间，合并到同日期观察。
            if status == "unknown" and prior_not_taken and segment_timing != "unknown":
                status = "not_taken_yet"
            has_medication_evidence = bool(re.search(r"药|服药|服用|漏服|未服", segment))
            # 没有用药问题/词语时，“吃了”可能指吃饭；不将日常活动写成药物记录。
            if not has_medication_evidence:
                short = re.sub(r"^(?:今天|昨天|前天|明天|\d{4}-\d{2}-\d{2})", "", segment).strip()
                generic_answer = bool(re.fullmatch(
                    r"(?:我)?(?:已经|刚才|刚|按时|正常|照常)?(?:吃了|服了|吃过了|按时吃|不吃了|还没吃|还没有吃|没吃|没有吃|尚未服|还没服|不规律|断断续续)(?:了|呢|啊|呀|哦)?", short
                ))
                if not (medication_context or (prior_not_taken and segment_timing != "unknown")) or not (generic_answer or _medication_timing(short) != "unknown"):
                    continue
            if medication_context and status == "unknown" and _medication_timing(segment) == "not_due":
                status = "not_taken_yet"
            if status != "unknown":
                observations.append({"date": current_date.isoformat(), "status": status, "detail": segment.strip(), "timing": _medication_timing(segment)})
    return observations


def _denied_red_flags(text):
    from medical_tools.safety import DANGER_PATTERNS
    denied = []
    aliases = {
        "晕厥": ["晕倒"], "意识异常": ["意识异常"],
        "突发肢体无力": ["肢体无力"], "突发言语异常": ["说话不清", "言语异常"],
    }
    for clause in split_answer_clauses(text):
        if re.search(r"不确定|记不清|不知道|不记得|不能确定", clause):
            continue
        for label, phrases in DANGER_PATTERNS.items():
            for term in unique_list([label] + phrases + aliases.get(label, [])):
                if any(_is_negated(clause, match.start()) for match in re.finditer(re.escape(term), clause)):
                    denied.append(label)
    return unique_list(denied)


def _asked_symptoms(context):
    from medical_tools.safety import DANGER_PATTERNS
    question = context.get("question") or ""
    if "asked_symptoms" in context:
        symptoms = list(context.get("asked_symptoms") or [])
    elif question:
        positive, negative = detect_symptoms(question)
        symptoms = unique_list(positive + negative)
    else:
        symptoms = list(COMMON_SYMPTOMS)
    if "asked_red_flags" in context:
        red_flags = list(context.get("asked_red_flags") or [])
    elif question:
        aliases = {"晕厥": ["晕倒"], "意识异常": ["意识不清", "意识异常"], "突发肢体无力": ["肢体无力"], "突发言语异常": ["说话不清", "言语异常"]}
        red_flags = [label for label, terms in DANGER_PATTERNS.items() if any(term in question for term in [label] + terms + aliases.get(label, []))]
    else:
        red_flags = list(DANGER_PATTERNS)
    return symptoms, red_flags



def _verified_model_bp(text, model_patch):
    """非斜线血压仅接受标明高/低压、可定位且未否定的原文数字证据。"""
    evidence = model_patch.get("evidence") or {}
    if not isinstance(evidence, dict):
        return None, None
    fragment = evidence.get("bp")
    if not isinstance(fragment, str) or not fragment or fragment not in text:
        return None, None
    values = model_patch.get("bp_systolic"), model_patch.get("bp_diastolic")
    if any(type(value) is not int for value in values):
        return None, None
    labels = (r"高压|收缩压", r"低压|舒张压")
    for labels_pattern, value in zip(labels, values):
        match = re.search(r"(?:" + labels_pattern + r")[\s:：是为]*(\d{2,3})(?!\d)", fragment)
        if not match or int(match.group(1)) != value:
            return None, None
    original_clauses = split_answer_clauses(text)
    for piece in split_answer_clauses(fragment):
        source_clauses = [clause for clause in original_clauses if piece in clause]
        if not source_clauses:
            return None, None
        clause = source_clauses[0]
        if re.search(r"不确定|不知道|记不清|不记得|好像|可能|也许", clause):
            return None, None
        if any(_is_negated(clause, match.start()) or _term_is_negated(clause, match)
               for match in re.finditer(r"高压|低压|收缩压|舒张压", clause)):
            return None, None
    return values


def extract_user_patch(text: str, patient_state=None, *, call_json_model=None, contains_red_flag_language=None):
    """按已选 Agent 模式分派，保留旧版本调用接口。"""
    if (patient_state or {}).get("agent_version") == "V5.0":
        from agent.llm_v4 import extract_user_patch_v4
        return extract_user_patch_v4(text, patient_state, call_json_model=call_json_model, enable_v50=True)
    if contains_red_flag_language is None:
        from medical_tools.safety import contains_red_flag_language
    state = patient_state or {}
    text = text or ""
    context = _question_context(state)
    kind, topic = context.get("kind"), context.get("topic")
    answer = _short_answer(text)
    availability = _answer_availability(text)
    clause_answers = [_short_answer(clause) for clause in split_answer_clauses(text)]
    today = _today()
    bp_sys, bp_dia = extract_bp_from_text(text)
    symptoms, denied = detect_symptoms(text)
    medication_context = kind in {"medication_taken", "taken_clarify", "adherence", "timing", "medication_details"}
    observations = _medication_observations(text, today, medication_context)
    current = [item for item in observations if item["date"] == today.isoformat()]
    status = next((status for status in ("missed", "stopped", "irregular", "not_taken_yet", "taken") if any(item["status"] == status for item in current)), "unknown")
    timing = next((item["timing"] for item in reversed(current) if item["timing"] != "unknown"), "unknown")
    denied_flags = _denied_red_flags(text)
    patch = {
        "bp_systolic": bp_sys, "bp_diastolic": bp_dia,
        "medication_status": status, "medication_detail": text if current else "",
        "medication_timing": timing, "medication_observations": observations,
        "symptoms": symptoms, "denied_symptoms": denied, "denied_red_flags": denied_flags,
        "additional_information": "", "answered_topics": [], "topic_status": {},
        "patient_question": text if looks_like_patient_question(text) else "",
        "medication_adherence": "",
        "answer_status": availability,
    }

    def mark(name, value):
        if name in TOPICS:
            patch["topic_status"][name] = value

    if bp_sys is not None:
        mark("bp", "answered")
    if symptoms or denied:
        mark("current_symptoms", "answered")
    has_red_flag_language = contains_red_flag_language(text)
    if has_red_flag_language and answer != "uncertain":
        # 兼容旧 answered_topics 标记，state_manager 根据逐项否认决定全项 Coverage。
        mark("red_flags", "answered")
    if any(term in text for term in ["休息", "刚运动", "刚活动", "坐着", "坐位", "袖带", "刚吃饭", "刚喝咖啡", "情绪激动"]):
        mark("measurement_context", "answered")
    if status != "unknown":
        mark("medication", "clarification_needed" if status == "not_taken_yet" and timing == "unknown" else "answered")

    # 上一题的语义决定“没有/是的”的含义，历史记录不提供当前答案。
    contextual_medication = False
    if medication_context and status == "unknown" and (not observations or current):
        if answer in {"yes", "no"}:
            if kind in {"medication_taken", "taken_clarify"}:
                # 显式kind由Question Planner保证单问；旧长问句回退才检测双问。
                compound = not (state.get("question_context") or {}).get("kind") and bool(re.search(r"漏服|停药|忘记|间断", context.get("question") or ""))
                if compound:
                    mark("medication", "clarification_needed")
                else:
                    patch["medication_status"] = "taken" if answer == "yes" else "not_taken_yet"
                    mark("medication", "answered" if answer == "yes" else "clarification_needed")
                    contextual_medication = True
            else:
                mark("medication", "clarification_needed")
                if kind == "adherence" and answer == "no":
                    patch["medication_adherence"] = "no_missed_or_stopped"
        elif kind == "timing" or (kind == "medication_details" and context.get("goal") == "clarify_medication_timing"):
            answer_timing = _medication_timing(text)
            if answer_timing != "unknown" and state.get("medication_status") == "not_taken_yet":
                patch["medication_timing"] = answer_timing
                patch["medication_status"] = "not_taken_yet"
                mark("medication", "answered")
                contextual_medication = True
        elif re.search(r"(?:没有|没|未|不曾).{0,4}(?:漏服|漏吃|忘记|忘了|停药)|记得.{0,4}(?:服药|吃药)", text):
            patch["medication_adherence"] = "no_missed_or_stopped"
            mark("medication", "clarification_needed")
        if kind == "medication_details" and text.strip():
            patch["additional_information"] = text
            if patch["topic_status"].get("medication") != "answered":
                mark("medication", "clarification_needed")
    if observations and not current and topic == "medication":
        mark("medication", "clarification_needed")
    if not observations and re.search(r"(?:没有|没|未|不曾).{0,4}(?:漏服|漏吃|忘记|忘了|停药)", text):
        patch["medication_adherence"] = "no_missed_or_stopped"
        mark("medication", "clarification_needed")
    if contextual_medication:
        patch["medication_detail"] = text
        patch["medication_observations"].append({"date": today.isoformat(), "status": patch["medication_status"], "detail": text, "timing": patch["medication_timing"]})

    if kind == "symptoms" and any(value in {"no", "no_symptoms"} for value in clause_answers):
        asked_symptoms, asked_flags = _asked_symptoms(context)
        from medical_tools.safety import detect_danger_signs
        positive_flags = unique_list([label for clause in split_answer_clauses(text) for label in detect_danger_signs(clause)])
        patch["denied_symptoms"] = [label for label in unique_list(denied + asked_symptoms) if label not in symptoms]
        patch["denied_red_flags"] = [label for label in unique_list(denied_flags + asked_flags) if label not in symptoms and label not in positive_flags]
        if asked_symptoms:
            mark("current_symptoms", "answered")
        if asked_flags:
            mark("red_flags", "answered")
    elif kind == "symptoms" and answer == "yes" and not symptoms:
        mark("current_symptoms", "clarification_needed")
        if context.get("asked_red_flags"):
            mark("red_flags", "clarification_needed")
    elif kind == "measurement_context" and answer in {"yes", "no"}:
        question = context.get("question") or ""
        if len(re.findall(r"有没有|是否|还是", question)) > 1:
            mark("measurement_context", "clarification_needed")
        else:
            mark("measurement_context", "answered")
            patch["additional_information"] = text
    elif kind == "bp" and answer == "no" and bp_sys is None:
        mark("bp", "clarification_needed")
        patch["additional_information"] = text

    if "no_symptoms" in clause_answers and kind != "symptoms":
        mark("current_symptoms", "answered")
        if topic and topic != "current_symptoms":
            mark(topic, "clarification_needed")
    if answer == "uncertain" and topic and patch["topic_status"].get(topic) != "answered":
        mark(topic, "clarification_needed")
        if kind == "symptoms" and (context.get("asked_red_flags") or has_red_flag_language):
            mark("red_flags", "clarification_needed")

    if availability and topic:
        # 已经主动提供的明确事实保留；拒绝剩余问题不能让未知项变成否认。
        if patch["topic_status"].get(topic) != "answered":
            mark(topic, "clarification_needed")
        if kind == "symptoms" and context.get("asked_red_flags"):
            mark("red_flags", "clarification_needed")

    # 普通规则已解释的短答/数值/症状/用药不调用模型。复杂、模糊文本允许辅助提取，
    # 但模型的状态、否认、Coverage、数值不能直接写入未经原文证明的事实。
    explained = bool(bp_sys is not None or symptoms or denied or observations or patch["topic_status"] or answer in {"yes", "no", "no_symptoms"})
    use_model = not availability and (answer == "uncertain" or (not explained and bool(text.strip())))
    if use_model:
        if call_json_model is None:
            from backend.model_client import call_json_model
        model_patch = call_json_model(
            system_prompt=(
                "你是高血压随访的信息提取助手。只提取患者本次原话明确提供的信息，禁止猜测、诊断、医疗建议。"
                "状态已有事实与历史记忆不能当作患者本轮回答。回答不确定时保持未知。"
                "只返回JSON，保持bp_systolic、bp_diastolic、medication_status、symptoms、denied_symptoms、additional_information字段。"
                "additional_information必须是本次原话中的连续片段。增加evidence对象：bp为包含高压/低压或收缩压/舒张压"
                "及对应阿拉伯数字的原文连续片段，medication为服药原文片段，symptoms和denied_symptoms为症状到原文片段的映射。"
                "没有明确原文证据时返回null或空数组；禁止推断Coverage、未经询问的否认及未提供事实。"
            ),
            user_prompt=json.dumps({"message": text, "question_context": context, "confirmed_this_followup": {
                "medication_status": state.get("medication_status", "unknown"),
                "symptoms": state.get("symptoms") or [], "denied_symptoms": state.get("denied_symptoms") or [],
                "topic_status": state.get("topic_status") or {},
            }}, ensure_ascii=False),
        )
        if isinstance(model_patch, dict):
            additional = model_patch.get("additional_information")
            if isinstance(additional, str) and additional.strip() and additional.strip() in text:
                patch["additional_information"] = additional.strip()
            # 已有症状/用药规则是证据门禁，模型不能推翻或增加未核实事实。
            # 血压补充只覆盖原有斜线规则未解析的明确高压/低压表达。
            if patch["bp_systolic"] is None:
                model_sys, model_dia = _verified_model_bp(text, model_patch)
                if model_sys is not None:
                    patch["bp_systolic"], patch["bp_diastolic"] = model_sys, model_dia
                    mark("bp", "answered")
        if topic and topic not in patch["topic_status"]:
            mark(topic, "clarification_needed")
    topic_fact = {
        "bp": patch["bp_systolic"] is not None,
        "medication": any(item["date"] == today.isoformat() for item in patch["medication_observations"]) or bool(patch["medication_adherence"]),
        "current_symptoms": bool(patch["symptoms"] or patch["denied_symptoms"]),
        "red_flags": bool(patch["denied_red_flags"] or has_red_flag_language),
        "measurement_context": patch["topic_status"].get("measurement_context") == "answered",
    }
    if not availability and topic and patch["topic_status"].get(topic) != "answered" and not topic_fact.get(topic):
        mark(topic, "clarification_needed")
        patch["answer_status"] = "unclear"
    patch["answered_topics"] = [name for name in ("bp", "medication", "current_symptoms", "red_flags", "measurement_context") if patch["topic_status"].get(name) == "answered"]
    return patch

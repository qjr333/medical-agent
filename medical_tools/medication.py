"""Medication questions and the existing prescription output safety guard."""

import re


def is_medication_question(
    text: str
):

    if not text:

        return False


    patterns = [

        r"(药|降压药)",
        r"(漏服|忘记吃|忘记服|漏吃)",
        r"(多吃|少吃|补吃|补服)",
        r"(加药|减药|停药|换药)",
        r"(副作用|不良反应)",
    ]


    return any(
        re.search(
            pattern,
            text
        )
        for pattern
        in patterns
    )



def contains_unsafe_medication_instruction(
    text: str
):

    if not text:

        return False


    safe_negations = [

        "不要自行",
        "不能自行",
        "不应自行",
        "请勿自行",
        "避免自行",
        "不可自行",
    ]


    if any(
        phrase in text
        for phrase
        in safe_negations
    ):

        # 有安全否定不能直接认为违规。
        # 后面继续检查更明确的剂量指令。
        pass


    dangerous_patterns = [

        r"(增加|提高|减少|降低)"
        r".{0,6}"
        r"(到|为)"
        r".{0,6}"
        r"\d+\s*(mg|毫克|片|粒)",

        r"(现在|今天|今晚|明天)"
        r".{0,6}"
        r"(多吃|加服|多服)"
        r".{0,6}"
        r"\d*"
        r"(片|粒|颗)",

        r"(把|将)"
        r".{0,10}"
        r"(停掉|停用|换成)"
        r".{0,15}"
        r"(降压药|药)",
    ]


    for pattern in dangerous_patterns:

        if re.search(
            pattern,
            text,
            flags=re.I,
        ):

            return True


    return False



def tool_medication_guidance(state: dict, *, call_json_model=None):
    if call_json_model is None:
        from backend.model_client import call_json_model

    patient_question = (
        state.get(
            "latest_patient_question"
        )
        or
        ""
    )


    medication_status = (
        state.get(
            "medication_status"
        )
        or
        "unknown"
    )


    medication_detail = (
        state.get(
            "medication_detail"
        )
        or
        ""
    )


    age = state.get(
        "age"
    )


    bp_systolic = state.get(
        "bp_systolic"
    )


    bp_diastolic = state.get(
        "bp_diastolic"
    )


    symptoms = (
        state.get(
            "symptoms"
        )
        or
        []
    )


    red_flags = (
        state.get(
            "red_flags"
        )
        or
        []
    )


    retrieved_guidance = (
        state.get(
            "retrieved_guidance"
        )
        or
        []
    )


    # =====================================================
    # A. Safety First
    # =====================================================

    if red_flags:

        return {

            "status":
                "blocked_by_safety",

            "message":
                (
                    "当前信息中出现了需要优先处理的危险表现，"
                    "不继续进行普通用药问答。"
                ),

            "needs_followup":
                False,

            "followup_question":
                "",

            "needs_professional_review":
                True,

            "reason":
                "存在red flags",

            "source_blocks":
                [],
        }


    # =====================================================
    # B. No Evidence
    # =====================================================

    if not retrieved_guidance:

        return {

            "status":
                "insufficient_evidence",

            "message":
                (
                    "目前没有检索到足够的指南依据，"
                    "我不能根据现有信息替您改变处方药的剂量、"
                    "服用次数或药物种类。"
                    "如果是漏服或服药后出现不适，"
                    "最好先确认药物名称和原来的服用方案，"
                    "必要时向负责治疗的医生或药师确认。"
                ),

            "needs_followup":
                True,

            "followup_question":
                (
                    "您现在服用的降压药名称是什么，"
                    "原来医生安排您一天怎么服用？"
                ),

            "needs_professional_review":
                True,

            "reason":
                "当前没有RAG证据",

            "source_blocks":
                [],
        }


    # =====================================================
    # C. Build Evidence
    # =====================================================

    evidence_parts = []

    source_blocks = []


    for index, item in enumerate(
        retrieved_guidance[:5],
        start=1,
    ):

        source_title = (
            item.get(
                "source_title"
            )
            or
            "未知来源"
        )


        source_pages = (
            item.get(
                "source_pages"
            )
            or
            ""
        )


        block_id = (
            item.get(
                "block_id"
            )
            or
            ""
        )


        evidence_text = (
            item.get(
                "text"
            )
            or
            ""
        )


        evidence_parts.append(
            f"""
【证据{index}】
block_id: {block_id}
来源: {source_title}
年份: {item.get("year")}
页码/章节: {source_pages}

内容:
{evidence_text}
"""
        )


        source_blocks.append(
            {

                "block_id":
                    block_id,

                "source_title":
                    source_title,

                "source_pages":
                    source_pages,

                "year":
                    item.get(
                        "year"
                    ),
            }
        )


    evidence = "\n".join(
        evidence_parts
    )


    # =====================================================
    # D. LLM
    # =====================================================

    system_prompt = """
你是老年高血压主动随访系统中的“用药管理辅助模块”。

你的任务是根据患者状态和提供的RAG医学证据，
用简单、清楚、适合老年患者理解的语言解释用药相关问题。

必须遵守：

1. 只能依据提供的患者信息和RAG证据。
2. 不得自行诊断。
3. 不得新开处方药。
4. 不得要求患者自行增加剂量。
5. 不得要求患者自行减少剂量。
6. 不得要求患者自行停药。
7. 不得要求患者自行换药。
8. 不得仅根据一次家庭血压结果改变处方。
9. 不得猜测患者没有提供的具体药物名称。
10. 如果缺少药物名称、既定处方或关键情况，
    要明确表示信息不足。
11. 如果问题需要医生或药师根据具体药物判断，
    应建议向医疗专业人员确认。

重点可以处理：

- 是否规律服药
- 漏服或间断服药
- 为什么没有规律服药
- 是否存在药物不良反应
- 什么时候需要医生进一步评估
- 是否还需要追问药物名称或既定服药方案

只返回JSON：

{
  "patient_message": "",
  "needs_followup": true,
  "followup_question": "",
  "needs_professional_review": false,
  "reason": ""
}
"""


    user_prompt = f"""
【患者状态】

年龄：
{age}

当前血压：
{bp_systolic}/{bp_diastolic} mmHg

服药状态：
{medication_status}

用药详情：
{medication_detail}

患者在本次随访中补充的原话（未经确认的内容不得推断为既定处方）：
{state.get("additional_information") or "暂无补充"}

症状：
{symptoms}

患者问题：
{patient_question}


【RAG医学证据】

{evidence}


请严格基于上述证据回答。
"""


    result = call_json_model(
        system_prompt=
            system_prompt,

        user_prompt=
            user_prompt,
    )


    # =====================================================
    # E. Fallback
    # =====================================================

    if not result:

        result = {

            "patient_message":
                (
                    "根据目前的信息，"
                    "我不能替您自行增加、减少、停止或更换降压药。"
                    "如果发生漏服，具体怎么处理通常还需要结合"
                    "您正在服用的药物和医生原来的服用方案判断。"
                ),

            "needs_followup":
                True,

            "followup_question":
                (
                    "您现在服用的降压药叫什么名字，"
                    "医生原来安排您一天怎么服用？"
                ),

            "needs_professional_review":
                True,

            "reason":
                "模型不可用，使用安全Fallback",
        }


    patient_message = (
        result.get(
            "patient_message"
        )
        or
        ""
    ).strip()


    needs_followup = bool(
        result.get(
            "needs_followup",
            False
        )
    )


    followup_question = (
        result.get(
            "followup_question"
        )
        or
        ""
    ).strip()


    needs_professional_review = bool(
        result.get(
            "needs_professional_review",
            False
        )
    )


    reason = (
        result.get(
            "reason"
        )
        or
        ""
    ).strip()


    # =====================================================
    # F. Output Safety Guard
    # =====================================================

    if contains_unsafe_medication_instruction(
        patient_message
    ):

        patient_message = (
            "根据目前的信息，我不能替您自行改变处方药的剂量、"
            "服用次数或药物种类。"
            "如果出现漏服、服药后不适或血压持续异常，"
            "应把当前药物名称、医生原来的服用方案和近期血压记录"
            "告诉负责治疗的医生或药师，由专业人员进一步判断。"
        )


        needs_professional_review = True

        reason = (
            "原始模型输出触发Medication Safety Guard"
        )


    return {

        "status":
            "ok",

        "message":
            patient_message,

        "needs_followup":
            needs_followup,

        "followup_question":
            followup_question,

        "needs_professional_review":
            needs_professional_review,

        "reason":
            reason,

        "source_blocks":
            source_blocks,
    }


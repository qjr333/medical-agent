"""Existing risk thresholds, red flag detection and measurement checks."""

import re

from backend.utils import unique_list

HIGH_BP_SYS = 180
HIGH_BP_DIA = 120

ATTENTION_BP_SYS = 160
ATTENTION_BP_DIA = 100

DANGER_PATTERNS = {

    "胸痛":

        [
            "剧烈胸痛",
            "持续胸痛",
            "严重胸痛",
            "胸口剧烈疼痛",
            "胸口很痛",
        ],


    "呼吸困难":

        [
            "呼吸困难",
            "喘不上气",
            "无法呼吸",
            "呼吸很困难",
        ],


    "晕厥":

        [
            "晕厥",
            "晕过去",
            "突然晕倒",
            "失去意识",
        ],


    "意识异常":

        [
            "意识不清",
            "意识模糊",
            "神志不清",
        ],


    "突发肢体无力":

        [
            "突然肢体无力",
            "突然手脚无力",
            "一侧肢体无力",
            "突然偏瘫",
        ],


    "突发言语异常":

        [
            "突然说话不清",
            "突然言语不清",
            "突然不会说话",
        ],
}


def phrase_is_negated(
    text: str,
    phrase: str,
):
    """Respect coordinated denials and later positive clauses independently.

    The nearest polarity marker governs each occurrence. Every occurrence
    must be denied before the whole phrase can be ruled out; a later explicit
    positive statement remains a danger signal.
    """
    occurrences = list(re.finditer(re.escape(phrase), text or ""))
    if not occurrences:
        return False
    boundaries = r"[，,。.;；!?！？\n]|但是|不过|然而|可是|只有|只是|现在|当前|此刻|而且|并且|但|却"
    negatives = {"没有出现", "没有", "并无", "否认", "未出现", "不伴", "无", "没"}
    for occurrence in occurrences:
        prefix = re.split(boundaries, text[:occurrence.start()])[-1]
        markers = list(re.finditer(
            r"没有出现|没有|并无|否认|未出现|不伴|无(?!力|法|论)|没|(?<![没无])有|出现|伴随", prefix))
        if not markers or markers[-1].group() not in negatives:
            return False
    return True



def detect_danger_signs(
    text: str
):

    text = (
        text
        or ""
    )


    found = []


    for label, phrases in (
        DANGER_PATTERNS.items()
    ):

        for phrase in phrases:

            if phrase not in text:
                continue


            if phrase_is_negated(
                text,
                phrase
            ):

                continue


            found.append(
                label
            )

            break


    return unique_list(
        found
    )



def contains_red_flag_language(
    text: str
):

    for phrases in (
        DANGER_PATTERNS.values()
    ):

        for phrase in phrases:

            if phrase in text:

                return True


    simple_terms = [

        "胸痛",
        "呼吸困难",
        "晕厥",
        "晕倒",
        "意识不清",
        "肢体无力",
        "说话不清",
    ]


    return any(
        term in text
        for term
        in simple_terms
    )



def tool_assess_risk(
    state: dict
):

    reasons = []


    sys_bp = state.get(
        "bp_systolic"
    )


    dia_bp = state.get(
        "bp_diastolic"
    )


    red_flags = (
        state.get(
            "red_flags"
        )
        or
        []
    )


    symptoms = (
        state.get(
            "symptoms"
        )
        or
        []
    )


    medication_status = (
        state.get(
            "medication_status"
        )
    )


    # =====================================================
    # HIGH
    # =====================================================

    if red_flags:

        reasons.append(
            "检测到危险症状或高风险表现"
        )


        state[
            "risk_level"
        ] = "high"

        state[
            "risk_reasons"
        ] = reasons


        return {

            "risk_level":
                "high",

            "reasons":
                reasons,
        }


    if (
        isinstance(
            sys_bp,
            int
        )
        and
        sys_bp
        >=
        HIGH_BP_SYS
    ):

        reasons.append(
            "收缩压达到高风险规则阈值"
        )


    if (
        isinstance(
            dia_bp,
            int
        )
        and
        dia_bp
        >=
        HIGH_BP_DIA
    ):

        reasons.append(
            "舒张压达到高风险规则阈值"
        )


    if reasons:

        state[
            "risk_level"
        ] = "high"

        state[
            "risk_reasons"
        ] = reasons


        return {

            "risk_level":
                "high",

            "reasons":
                reasons,
        }


    # =====================================================
    # ATTENTION
    # =====================================================

    attention_reasons = []


    if (
        isinstance(
            sys_bp,
            int
        )
        and
        sys_bp
        >=
        ATTENTION_BP_SYS
    ):

        attention_reasons.append(
            "近期家庭收缩压偏高"
        )


    if (
        isinstance(
            dia_bp,
            int
        )
        and
        dia_bp
        >=
        ATTENTION_BP_DIA
    ):

        attention_reasons.append(
            "近期家庭舒张压偏高"
        )


    if medication_status in [
        "missed",
        "irregular",
        "stopped",
    ]:

        attention_reasons.append(
            "存在服药依从性问题"
        )


    if symptoms:

        attention_reasons.append(
            "患者报告当前不适症状"
        )


    # 趋势是随访关注信号，不单独当成医疗诊断。
    bp_analysis = state.get("bp_analysis") or {}
    if bp_analysis.get("consecutive_rise"):
        attention_reasons.append("最近3次记录的收缩压或舒张压连续升高")
    elif state.get("bp_trend") == "rising":
        attention_reasons.append("近期家庭血压呈上升趋势")

    if attention_reasons:

        state[
            "risk_level"
        ] = "attention"

        state[
            "risk_reasons"
        ] = unique_list(
            attention_reasons
        )


        return {

            "risk_level":
                "attention",

            "reasons":
                state[
                    "risk_reasons"
                ],
        }


    # =====================================================
    # NORMAL
    # =====================================================

    state[
        "risk_level"
    ] = "normal"

    state[
        "risk_reasons"
    ] = []


    return {

        "risk_level":
            "normal",

        "reasons":
            [],
    }



def valid_bp_measurement(systolic, diastolic):
    """仅检查读数是否明显无效，不作临床诊断。"""
    return (
        type(systolic) is int and type(diastolic) is int
        and 50 <= systolic <= 300
        and 30 <= diastolic <= 200
        and systolic > diastolic
    )


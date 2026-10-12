"""Existing family and clinician alert content generation."""


def tool_create_alert(
    state: dict
):

    symptoms = (
        state.get(
            "red_flags"
        )
        or
        state.get(
            "symptoms"
        )
        or
        []
    )


    bp_text = (
        f"{state.get('bp_systolic')}/"
        f"{state.get('bp_diastolic')}"
    )


    family_alert = (
        f"{state.get('patient_name')}本次随访出现需要优先关注的信息。"
        f"当前血压记录：{bp_text} mmHg；"
        f"相关表现：{symptoms}。"
        f"建议尽快协助联系医疗专业人员进行评估。"
    )


    clinician_alert = (
        f"患者：{state.get('patient_name')}，"
        f"{state.get('age')}岁，高血压。"
        f"触发原因：{state.get('followup_reason')}。"
        f"当前血压：{bp_text} mmHg。"
        f"风险信息：{state.get('risk_reasons')}。"
        f"当前症状/危险表现：{symptoms}。"
    )


    return {

        "family_alert_content":
            family_alert,

        "clinician_alert_content":
            clinician_alert,
    }


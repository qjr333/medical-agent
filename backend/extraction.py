"""第一阶段兼容入口；提取实现已迁到 agent.extractor。"""

from agent.extractor import (
    COMMON_SYMPTOMS, extract_bp_from_text,
    detect_medication_status, detect_symptoms, looks_like_patient_question,
    extract_user_patch,
)

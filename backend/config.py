"""环境配置、项目路径以及演示患者 JSON 加载。"""

import json
import os

from dotenv import load_dotenv

load_dotenv()


DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY",
    "sk-e4e9eb6baf434c43b9c78c24dee4aa11"
).strip()


DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com"
).strip()


DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat"
).strip()


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)


INDEX_FILE = os.path.join(
    STATIC_DIR,
    "index.html"
)


DEMO_DATA_FILE = os.path.join(BASE_DIR, "demo_patient.json")


def load_demo_data(file_path=None):
    """从 UTF-8 JSON 读取演示患者、初始血压与触发场景。"""
    if file_path is None:
        file_path = DEMO_DATA_FILE
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"缺少演示患者数据文件：{file_path}。"
            "请将 demo_patient.json 放在 app.py 同级目录。"
        ) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "demo_patient.json 不是有效的 UTF-8 JSON，请检查文件编码和语法。"
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError("demo_patient.json 顶层必须是 JSON 对象。")

    patient = data.get("patient")
    history = data.get("bp_history")
    configs = data.get("trigger_configs")

    if not isinstance(patient, dict) or not all(
        patient.get(key) for key in ("patient_id", "patient_name")
    ):
        raise RuntimeError("demo_patient.json 的 patient 缺少患者编号或姓名。")
    if not isinstance(patient.get("age"), int) or not isinstance(patient.get("conditions"), list):
        raise RuntimeError("patient.age 必须为整数，patient.conditions 必须为数组。")
    if not isinstance(history, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("systolic"), int)
        and isinstance(item.get("diastolic"), int)
        for item in history
    ):
        raise RuntimeError("bp_history 必须是包含 systolic 和 diastolic 的记录数组。")
    if not isinstance(configs, dict) or "scheduled" not in configs:
        raise RuntimeError("trigger_configs 必须包含 scheduled 触发场景。")
    for trigger_type, config in configs.items():
        if not isinstance(config, dict) or not all(
            key in config for key in ("label", "followup_reason", "bp_systolic", "bp_diastolic")
        ):
            raise RuntimeError(f"触发场景 {trigger_type} 缺少必需配置字段。")

    return data


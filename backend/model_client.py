"""DeepSeek 客户端与 JSON 响应解析；保持 JSON mode 和普通模式回退。"""

import json
import re
from contextvars import ContextVar
from openai import OpenAI
from backend import config
from backend.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

client = None
model_failure_observer = ContextVar("medical_agent_model_failure_observer", default=None)
model_call_observer = ContextVar("medical_agent_model_call_observer", default=None)


def _notify_call(mode, status, error=None):
    observer = model_call_observer.get()
    if observer is not None:
        observer({"provider": "deepseek", "mode": mode, "status": status,
                  "error_type": type(error).__name__ if error is not None else None})

if DEEPSEEK_API_KEY:

    try:

        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=20.0,
            max_retries=0,
        )

    except Exception as error:

        print(
            "[MODEL] Client初始化失败：",
            type(error).__name__
        )


else:

    print(
        "[MODEL] 未检测到 DEEPSEEK_API_KEY，"
        "涉及LLM的功能将使用Fallback。"
    )


_USE_CONFIGURED_CLIENT = object()


def _configured_client():
    return client


def parse_json_text(
    text: str
):

    if not text:

        return {}


    text = text.strip()


    # markdown code fence
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.I,
    )


    text = re.sub(
        r"```$",
        "",
        text,
    )


    text = text.strip()


    try:

        return json.loads(
            text
        )


    except Exception:

        pass


    # 尝试截取第一个JSON对象
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.S,
    )


    if match:

        try:

            return json.loads(
                match.group(0)
            )

        except Exception:

            pass


    return {}


def call_json_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    *,
    client_override=_USE_CONFIGURED_CLIENT,
    model_name=None,
):

    client = (_configured_client() if client_override is _USE_CONFIGURED_CLIENT
              else client_override)
    DEEPSEEK_MODEL = config.DEEPSEEK_MODEL if model_name is None else model_name

    if client is None:

        return {}


    messages = [

        {
            "role":
                "system",

            "content":
                system_prompt,
        },

        {
            "role":
                "user",

            "content":
                user_prompt,
        },
    ]


    # -----------------------------------------------------
    # 优先JSON mode
    # -----------------------------------------------------

    try:

        _notify_call("json", "started")

        response = (
            client
            .chat
            .completions
            .create(

                model=
                    DEEPSEEK_MODEL,

                messages=
                    messages,

                temperature=
                    temperature,

                response_format={
                    "type":
                        "json_object"
                },
            )
        )


        content = (
            response
            .choices[0]
            .message
            .content
        )

        _notify_call("json", "success")


        return parse_json_text(
            content
        )


    except Exception as first_error:

        _notify_call("json", "error", first_error)

        print(
            "[MODEL] JSON mode调用失败，尝试普通模式：",
            type(first_error).__name__
        )


    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    try:

        _notify_call("plain", "started")

        response = (
            client
            .chat
            .completions
            .create(

                model=
                    DEEPSEEK_MODEL,

                messages=
                    messages,

                temperature=
                    temperature,
            )
        )


        content = (
            response
            .choices[0]
            .message
            .content
        )

        _notify_call("plain", "success")


        return parse_json_text(
            content
        )


    except Exception as error:

        _notify_call("plain", "error", error)

        print(
            "[MODEL] 调用失败：",
            type(error).__name__
        )

        observer = model_failure_observer.get()
        if observer is not None:
            observer(error)


        return {}

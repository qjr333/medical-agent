"""DeepSeek 客户端与 JSON 响应解析；保持 JSON mode 和普通模式回退。"""

import json
import re
from contextvars import ContextVar
import openai as _sdk
from openai import OpenAI
from backend import config
from backend.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

client = None
model_failure_observer = ContextVar("medical_agent_model_failure_observer", default=None)
model_call_observer = ContextVar("medical_agent_model_call_observer", default=None)


def is_temporary_model_error(error):
    """The Agent owns retry limits; the SDK must never retry independently."""
    sdk_types = tuple(kind for kind in (
        getattr(_sdk, "APITimeoutError", None),
        getattr(_sdk, "APIConnectionError", None),
        getattr(_sdk, "RateLimitError", None),
        getattr(_sdk, "InternalServerError", None),
    ) if isinstance(kind, type))
    if isinstance(error, (TimeoutError, ConnectionError) + sdk_types):
        return True
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600)


def _error_status(error):
    sdk_timeout = getattr(_sdk, "APITimeoutError", None)
    if isinstance(error, TimeoutError) or (
        isinstance(sdk_timeout, type) and isinstance(error, sdk_timeout)
    ):
        return "timeout"
    sdk_connection = getattr(_sdk, "APIConnectionError", None)
    if isinstance(error, ConnectionError) or (
        isinstance(sdk_connection, type) and isinstance(error, sdk_connection)
    ):
        return "connection_error"
    return "temporary_error" if is_temporary_model_error(error) else "error"


def _json_mode_unsupported(error):
    # Only provider/SDK format compatibility errors merit switching protocols.
    # The message is inspected locally and never included in logs or Trace.
    description = str(error).lower()
    format_named = any(word in description for word in (
        "response_format", "json_object", "json mode", "json_mode",
    ))
    unsupported = any(word in description for word in (
        "unsupported", "not supported", "unavailable", "unknown", "unexpected keyword",
    ))
    return format_named and unsupported


def _notify_failure(error):
    observer = model_failure_observer.get()
    if observer is not None:
        observer(error)


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
            timeout=getattr(config, "DEEPSEEK_TIMEOUT_SECONDS", 50.0),
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
    raise_transient_errors=False,
    request_mode="json",
):

    client = (_configured_client() if client_override is _USE_CONFIGURED_CLIENT
              else client_override)
    DEEPSEEK_MODEL = config.DEEPSEEK_MODEL if model_name is None else model_name

    if client is None:

        return {}

    if request_mode not in ("json", "plain"):
        raise ValueError("request_mode must be json or plain")

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

    if request_mode == "json":
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

            _notify_call("json", _error_status(first_error) if raise_transient_errors else "error", first_error)

            if raise_transient_errors:
                if is_temporary_model_error(first_error):
                    first_error.model_request_mode = "json"
                    _notify_failure(first_error)
                    raise
                if not _json_mode_unsupported(first_error):
                    _notify_failure(first_error)
                    return {}

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

        _notify_call("plain", _error_status(error) if raise_transient_errors else "error", error)

        if raise_transient_errors and is_temporary_model_error(error):
            error.model_request_mode = "plain"
            _notify_failure(error)
            raise

        print(
            "[MODEL] 调用失败：",
            type(error).__name__
        )

        _notify_failure(error)


        return {}

"""基础数据辅助函数。"""

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def today_local_date():
    """随访中“今天”的日期，使用患者界面的中国时区。"""
    return local_date_from_timestamp(now_iso())


def local_date_from_timestamp(value):
    """把既有 SQLite/Trace 时间转换为随访日期；无效时间保持未知。"""
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    except (TypeError, ValueError):
        return None


def to_json(value):
    """保留中文，方便检查和恢复 Patient State。"""
    return json.dumps(value, ensure_ascii=False, default=str)


def unique_list(
    values
):

    result = []

    seen = set()


    for value in values:

        if value is None:
            continue


        if isinstance(
            value,
            str
        ):

            value = value.strip()


        if not value:
            continue


        marker = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )


        if marker in seen:
            continue


        seen.add(
            marker
        )

        result.append(
            value
        )


    return result


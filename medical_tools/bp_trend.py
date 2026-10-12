"""Existing recent blood pressure analysis tool."""


def tool_check_bp_history(state: dict):

    history = [
        dict(item)
        for item in state.get("bp_history", [])
        if isinstance(item.get("systolic"), int)
        and isinstance(item.get("diastolic"), int)
    ]

    current_sys = state.get("bp_systolic")
    current_dia = state.get("bp_diastolic")

    # 加入今天的血压，避免重复
    if isinstance(current_sys, int) and isinstance(current_dia, int):

        current_item = {
            "label": "今日",
            "systolic": current_sys,
            "diastolic": current_dia,
        }

        if not history:
            history.append(current_item)

        else:
            last = history[-1]

            if not (
                last.get("systolic") == current_sys
                and last.get("diastolic") == current_dia
            ):
                history.append(current_item)

    # 最多分析最近7次
    recent7 = history[-7:]
    recent3 = history[-3:]

    def average(items, key):
        if not items:
            return None

        return round(
            sum(item[key] for item in items) / len(items),
            1
        )

    avg3_sys = average(recent3, "systolic")
    avg3_dia = average(recent3, "diastolic")

    avg7_sys = average(recent7, "systolic")
    avg7_dia = average(recent7, "diastolic")

    trend = "unknown"
    consecutive_rise = False
    consecutive_fall = False

    systolic_range = None
    diastolic_range = None

    systolic_change = None
    diastolic_change = None

    # 至少2次才能判断变化
    if len(recent7) >= 2:

        first = recent7[0]
        last = recent7[-1]

        systolic_change = (
            last["systolic"]
            - first["systolic"]
        )

        diastolic_change = (
            last["diastolic"]
            - first["diastolic"]
        )

        systolic_values = [
            item["systolic"]
            for item in recent7
        ]

        diastolic_values = [
            item["diastolic"]
            for item in recent7
        ]

        systolic_range = (
            max(systolic_values)
            - min(systolic_values)
        )

        diastolic_range = (
            max(diastolic_values)
            - min(diastolic_values)
        )

    # 最近3次连续变化
    if len(recent3) >= 3:

        s = [
            item["systolic"]
            for item in recent3
        ]

        d = [
            item["diastolic"]
            for item in recent3
        ]

        consecutive_rise = (
            s[0] < s[1] < s[2]
            or
            d[0] < d[1] < d[2]
        )

        consecutive_fall = (
            s[0] > s[1] > s[2]
            or
            d[0] > d[1] > d[2]
        )

        if consecutive_rise:
            trend = "rising"

        elif consecutive_fall:
            trend = "falling"

        elif (
            systolic_range is not None
            and systolic_range <= 10
            and diastolic_range is not None
            and diastolic_range <= 8
        ):
            trend = "stable"

        else:
            trend = "fluctuating"

    elif len(recent7) == 2:

        if systolic_change > 5:
            trend = "rising"

        elif systolic_change < -5:
            trend = "falling"

        else:
            trend = "stable"

    analysis = {
        "trend": trend,

        "record_count": len(recent7),

        "latest": (
            recent7[-1]
            if recent7
            else None
        ),

        "recent_3": {
            "systolic_avg": avg3_sys,
            "diastolic_avg": avg3_dia,
        },

        "recent_7": {
            "systolic_avg": avg7_sys,
            "diastolic_avg": avg7_dia,
        },

        "change": {
            "systolic": systolic_change,
            "diastolic": diastolic_change,
        },

        "range": {
            "systolic": systolic_range,
            "diastolic": diastolic_range,
        },

        "consecutive_rise": consecutive_rise,

        "consecutive_fall": consecutive_fall,

        "history": recent7,
    }

    state["bp_history"] = history
    state["bp_trend"] = trend
    state["bp_analysis"] = analysis

    return analysis


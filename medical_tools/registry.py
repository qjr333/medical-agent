"""One tool registry shared by the Agent runner and compatibility entry points."""

from functools import partial

from .alerts import tool_create_alert
from .bp_trend import tool_check_bp_history
from .guidance import tool_retrieve_medical_guidance
from .medication import tool_medication_guidance
from .safety import tool_assess_risk
from .visit_advice import tool_generate_visit_advice


TOOL_REGISTRY = {
    "ASSESS_RISK": tool_assess_risk,
    "CHECK_BP_HISTORY": tool_check_bp_history,
    "RETRIEVE_MEDICAL_GUIDANCE": tool_retrieve_medical_guidance,
    "MEDICATION_GUIDANCE": tool_medication_guidance,
    "GENERATE_VISIT_ADVICE": tool_generate_visit_advice,
    "CREATE_ALERT": tool_create_alert,
}


def get_tool_registry(*, call_json_model=None):
    """Return fresh mappings, with optional model injection for an execution."""
    registry = dict(TOOL_REGISTRY)
    if call_json_model is not None:
        for name in ("MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            registry[name] = partial(registry[name], call_json_model=call_json_model)
    return registry

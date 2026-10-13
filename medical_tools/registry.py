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

# The model may select only these backend capabilities. State/patient identifiers,
# executable code, arbitrary URLs and prescription parameters are never model inputs.
LLM_ACTIONS = frozenset({"ASK_USER", "FINISH", *TOOL_REGISTRY})


def validate_action_arguments(action, arguments):
    """Validate the narrow public action schema without changing legacy tool APIs."""
    if action not in LLM_ACTIONS or not isinstance(arguments, dict):
        raise ValueError("invalid_action_arguments")
    allowed = {"query"} if action == "RETRIEVE_MEDICAL_GUIDANCE" else set()
    if set(arguments) - allowed:
        raise ValueError("unsupported_tool_parameter")
    if action == "RETRIEVE_MEDICAL_GUIDANCE":
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 450:
            raise ValueError("invalid_guidance_query")
        return {"query": query.strip()}
    return {}


def get_llm_action_schema():
    """Return data for the planner prompt; callable objects stay in the backend."""
    return {name: {"arguments": {"query": "nonempty string, max 450 characters"}
                  if name == "RETRIEVE_MEDICAL_GUIDANCE" else {},
                  "safety_only": name == "CREATE_ALERT"}
            for name in sorted(LLM_ACTIONS)}


def get_tool_registry(*, call_json_model=None):
    """Return fresh mappings, with optional model injection for an execution."""
    registry = dict(TOOL_REGISTRY)
    if call_json_model is not None:
        for name in ("MEDICATION_GUIDANCE", "GENERATE_VISIT_ADVICE"):
            registry[name] = partial(registry[name], call_json_model=call_json_model)
    return registry

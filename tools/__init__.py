"""Scenario tool definitions for misalignment experiments."""

from .neutral import NEUTRAL_BY_SCENARIO
from .signal import SIGNAL_TOOLS

BASE_SCENARIOS = tuple(NEUTRAL_BY_SCENARIO.keys())
ALL_SCENARIO = "all"
CLEAN_SCENARIO = "clean"
SUPPORTED_SCENARIOS = BASE_SCENARIOS + (ALL_SCENARIO, CLEAN_SCENARIO)


def _check(scenario: str) -> None:
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario}. Supported: {list(SUPPORTED_SCENARIOS)}"
        )


def _dedupe_by_function_name(tool_defs: list) -> list:
    seen = set()
    out = []
    for tool in tool_defs:
        name = tool.get("function", {}).get("name")
        if name in seen:
            continue
        seen.add(name)
        out.append(tool)
    return out


def get_tools_for_scenario(scenario: str, mode: str = "neutral") -> list:
    """Return tool definitions for a scenario in the given mode."""
    _check(scenario)
    if scenario in {ALL_SCENARIO, CLEAN_SCENARIO} or mode == "all":
        neutral = _dedupe_by_function_name(
            tool
            for base_scenario in BASE_SCENARIOS
            for tool in NEUTRAL_BY_SCENARIO[base_scenario]
        )
        return neutral + [SIGNAL_TOOLS[base_scenario] for base_scenario in BASE_SCENARIOS]

    neutral = list(NEUTRAL_BY_SCENARIO[scenario])
    if mode == "signal":
        return neutral + [SIGNAL_TOOLS[scenario]]
    return neutral


__all__ = [
    "SUPPORTED_SCENARIOS",
    "get_tools_for_scenario",
]

"""Abstract Environment contract for real-world Skill-MDP tool-use domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import ToolCall


@dataclass(frozen=True)
class ToolSpec:
    """Schema metadata for a single environment tool exposed to the executor."""

    name: str
    description: str
    parameters: dict[str, str]


@dataclass(frozen=True)
class ScenarioInfo:
    """Metadata describing one recurring operational scenario."""

    scenario_id: str
    category: str
    title: str
    severity: str
    initial_alert: str


@dataclass
class StepResult:
    """Result returned by executing one ToolCall in the environment."""

    observation: str
    reward: float
    done: bool
    resolved: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(Protocol):
    """Protocol implemented by all real-world operational environments."""

    def list_scenarios(self) -> list[ScenarioInfo]:
        """Return available scenarios supported by this environment."""

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return JSON-serializable tool schemas for the LLM prompt."""

    def reset(self, scenario_id: str | None = None) -> tuple[ScenarioInfo, str]:
        """Reset environment state for a specific or sampled scenario."""

    def step(self, action: ToolCall) -> StepResult:
        """Execute a structured tool call and return the updated observation & reward."""

    def finalize_reward(self) -> tuple[float, bool]:
        """Return final episode (reward, resolved) if episode terminates at max steps."""

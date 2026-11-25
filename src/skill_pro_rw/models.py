"""Core strictly-typed domain models for Skill-MDP, Non-Parametric PPO, and Audit Trails."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SkillStatus(str, Enum):
    """Lifecycle state of a procedural Skill inside the pool."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    ARCHIVED = "archived"


class Skill(BaseModel):
    """Executable procedural skill tuple: <Initiation, Execution Policy, Termination>."""

    id: str
    name: str
    version: int = 1
    parent_id: str = ""
    initiation: str
    policy: list[str] = Field(default_factory=list)
    termination: str
    allowed_tools: list[str] = Field(default_factory=list)
    status: SkillStatus = SkillStatus.ACTIVE

    # Online performance metrics tracked across episodes
    frequency: int = 0
    total_gain: float = 0.0
    avg_gain: float = 0.0
    maturity: int = 0
    success_count: int = 0
    last_evolved_iter: int = 0
    created_at: str = Field(default_factory=utc_now_iso)

    def format_for_llm(self) -> str:
        """Render the compact procedural runbook card injected into the executor prompt."""
        steps = "\n".join(f"- {step}" for step in self.policy) if self.policy else "- Diagnose carefully."
        return (
            f"Skill Name: {self.name} (v{self.version})\n"
            f"Initiation (When to use): {self.initiation}\n"
            f"Strategy Steps:\n{steps}\n"
            f"Termination (When to stop): {self.termination}"
        )

    def estimated_tokens(self) -> int:
        """Estimate token footprint of the formatted Skill card (~1.33x word count)."""
        words = len(self.format_for_llm().split())
        return max(1, (words * 4 + 2) // 3)

    def lcb(self, max_maturity: int, beta: float = 0.2) -> float:
        """Compute Lower Confidence Bound (LCB) score for safe selection and pruning."""
        t = max(1, max_maturity)
        penalty = beta * math.sqrt(math.log1p(t) / max(1, self.frequency))
        return self.avg_gain - penalty

    def update_stats(
        self,
        reward: float,
        baseline: float,
        total_skill_calls_in_traj: int,
        skill_call_count_in_traj: int = 1,
    ) -> None:
        """Update cumulative gain, invocation frequency, and success statistics."""
        advantage = reward - baseline
        denom = max(1, total_skill_calls_in_traj)
        per_call_gain = advantage / float(denom)
        self.total_gain += float(skill_call_count_in_traj) * per_call_gain
        self.frequency += max(1, skill_call_count_in_traj)
        self.avg_gain = self.total_gain / float(max(1, self.frequency))
        if advantage > 0.0:
            self.success_count += 1


class ToolCall(BaseModel):
    """Structured tool invocation produced by the executor agent."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Transition(BaseModel):
    """Single Skill-MDP step recording state, active skill, action, and observation."""

    step_index: int
    state_summary: str
    action: ToolCall
    observation_summary: str
    skill_id: str | None = None
    skill_name: str | None = None
    skill_done: bool = False
    step_reward: float = 0.0
    prompt_tokens: int = 0


class Experience(BaseModel):
    """Complete episode trajectory stored for advantage attribution and PPO Gate verification."""

    episode_id: str
    scenario_id: str
    env_name: str
    ablation_mode: Literal["full", "no_skill", "no_ppo_gate", "fifo_pruning"] = "full"
    reward: float
    baseline_at_run: float
    resolved: bool
    skills_used: list[str] = Field(default_factory=list)
    skill_ids_used: list[str] = Field(default_factory=list)
    step_count: int
    prompt_tokens: int
    episodic_baseline_tokens: int
    transitions: list[Transition] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class SemanticGradient(BaseModel):
    """Hindsight natural-language attribution targeting Activation, Policy, or Termination."""

    episode_id: str
    diagnosis: str
    is_related: bool
    initiation_delta: str = ""
    policy_delta: str = ""
    termination_delta: str = ""


class PPOGateEvaluation(BaseModel):
    """Result of evaluating a candidate Skill through the clipped PPO surrogate gate."""

    l_clip: float
    baseline_l_clip: float = 0.0
    improvement: float = 0.0
    margin: float
    passed: bool
    clipped_fraction: float
    mean_ratio: float
    min_ratio: float
    max_ratio: float
    old_logprob_mean: float
    new_logprob_mean: float
    step_count: int
    verifier_mode: str
    reason: str


class EvolutionAuditEntry(BaseModel):
    """Full audit record of one Non-Parametric PPO candidate generation and verification cycle."""

    audit_id: str
    iteration: int
    parent_skill_id: str
    parent_skill_name: str
    operation: Literal["REFINE", "NEW", "SKIPPED"]
    relevance_ratio: float
    gradients: list[SemanticGradient] = Field(default_factory=list)
    candidate_skill: Skill | None = None
    gate_eval: PPOGateEvaluation
    accepted: bool
    created_at: str = Field(default_factory=utc_now_iso)


class BackendEvent(BaseModel):
    """Structured event streamed live to the observability UI via SSE."""

    seq: int = 0
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

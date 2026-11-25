"""Configuration management for Skill-Pro RealWorld with strict secret hygiene."""

from __future__ import annotations

import os
from typing import Literal
from pydantic import BaseModel, Field


class SkillProConfig(BaseModel):
    """Runtime configuration for Skill-MDP, Non-Parametric PPO, and Hybrid backends."""

    mode: Literal["simulator", "hybrid"] = Field(
        default_factory=lambda: (
            "hybrid" if os.getenv("SKILL_PRO_MODE", "simulator").lower() == "hybrid" else "simulator"
        )
    )

    # Cloud executor configuration (Option B: remote API executor)
    executor_base_url: str = Field(
        default_factory=lambda: os.getenv("EXECUTOR_BASE_URL", "https://api.openai.com/v1")
    )
    executor_model: str = Field(
        default_factory=lambda: os.getenv("EXECUTOR_MODEL", "gpt-4o-mini")
    )

    # Local proxy logprob verifier configuration (Option B: local 127.0.0.1 model)
    proxy_verifier_mode: Literal["offline_proxy", "local_openai", "local_hf"] = Field(
        default_factory=lambda: _parse_verifier_mode(
            os.getenv("PROXY_VERIFIER_MODE", "offline_proxy")
        )
    )
    proxy_verifier_url: str = Field(
        default_factory=lambda: os.getenv("PROXY_VERIFIER_URL", "http://127.0.0.1:11434/v1")
    )
    proxy_verifier_model: str = Field(
        default_factory=lambda: os.getenv("PROXY_VERIFIER_MODEL", "qwen3:4b")
    )

    # Skill-MDP & Non-Parametric PPO hyperparameters (aligned with arXiv:2602.01869)
    pool_size: int = 10
    epsilon: float = 0.2
    acceptance_margin: float = 0.001
    best_of_n: int = 3
    evolution_threshold: int = 2
    min_frequency_for_evolution: int = 2
    cooldown_iterations: int = 1
    elite_gap_threshold: float = 0.01
    max_skill_age: int = 8
    max_episode_steps: int = 10
    ema_alpha: float = 0.1
    lcb_beta: float = 0.2
    explore_epsilon: float = 0.1
    dedup_similarity_threshold: float = 0.95

    # Storage and localhost-only HTTP binding
    db_path: str = Field(default_factory=lambda: os.getenv("SKILL_PRO_DB_PATH", "data/skill_pro.db"))
    host: str = Field(default_factory=lambda: os.getenv("SKILL_PRO_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("SKILL_PRO_PORT", "8420")))

    def public_summary(self) -> dict[str, object]:
        """Return safe configuration metadata without any credentials or secret tokens."""
        has_executor_key = bool(
            os.getenv("EXECUTOR_API_KEY") and os.getenv("EXECUTOR_API_KEY") != "REDACTED_SECRET"
        )
        return {
            "mode": self.mode,
            "executor_base_url": self.executor_base_url,
            "executor_model": self.executor_model,
            "executor_key_configured": has_executor_key,
            "proxy_verifier_mode": self.proxy_verifier_mode,
            "proxy_verifier_url": self.proxy_verifier_url,
            "proxy_verifier_model": self.proxy_verifier_model,
            "pool_size": self.pool_size,
            "epsilon": self.epsilon,
            "acceptance_margin": self.acceptance_margin,
            "best_of_n": self.best_of_n,
            "evolution_threshold": self.evolution_threshold,
            "host": self.host,
            "port": self.port,
        }


def _parse_verifier_mode(raw: str) -> Literal["offline_proxy", "local_openai", "local_hf"]:
    """Validate verifier mode environment variable safely."""
    normalized = raw.strip().lower()
    if normalized in ("offline_proxy", "local_openai", "local_hf"):
        return normalized  # type: ignore[return-value]
    return "offline_proxy"

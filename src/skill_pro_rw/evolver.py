"""Non-Parametric PPO Skill Evolver: Semantic Gradients + PPO Gate Verification."""

from __future__ import annotations

import math
import re
import uuid
from typing import Literal

import numpy as np

from .config import SkillProConfig
from .llm_backend import (
    ExecutorClient,
    HeuristicTokenProxyVerifier,
    LogprobVerifier,
    VerifierUnavailableError,
    build_executor_prompt,
    format_step_target,
)
from .models import (
    EvolutionAuditEntry,
    Experience,
    PPOGateEvaluation,
    SemanticGradient,
    Skill,
    SkillStatus,
)
from .ppo_gate import compute_clipped_surrogate
from .skill_pool import SkillPool
from .storage import SkillStore


class SkillEvolver:
    """Orchestrates hindsight semantic gradient synthesis and PPO Gate verification."""

    def __init__(
        self,
        store: SkillStore,
        pool: SkillPool,
        executor: ExecutorClient,
        verifier: LogprobVerifier,
        config: SkillProConfig,
    ) -> None:
        self.store = store
        self.pool = pool
        self.executor = executor
        self.verifier = verifier
        self.config = config
        self._experience_buffer: dict[str, list[Experience]] = {}
        self._seen_episode_ids: set[str] = set()

    def ingest_experiences(self, experiences: list[Experience]) -> None:
        """Buffer new completed episodes keyed by each active skill invoked."""
        for exp in experiences:
            if exp.episode_id in self._seen_episode_ids:
                continue
            self._seen_episode_ids.add(exp.episode_id)
            for skill_id in set(exp.skill_ids_used):
                self._experience_buffer.setdefault(skill_id, []).append(exp)

    def get_priority(self, skill: Skill, all_skills: list[Skill], current_iter: int) -> float:
        """Compute evolution priority based on maturity, cooldown, and elite performance gap."""
        if skill.frequency < self.config.min_frequency_for_evolution:
            return 0.0
        if (current_iter - skill.last_evolved_iter) < self.config.cooldown_iterations:
            return -0.01

        mature_gains = [sk.avg_gain for sk in all_skills if sk.maturity >= 1]
        max_gain = max(mature_gains, default=skill.avg_gain)
        gap = max(0.0, max_gain - skill.avg_gain)

        # Allow refinement even for top skills if recent negative/suboptimal buffered episodes exist
        buffered = self._experience_buffer.get(skill.id, [])
        has_improvable_episodes = any(ep.reward < 0.95 for ep in buffered)
        effective_gap = max(gap, 0.05 if has_improvable_episodes else 0.0)
        if effective_gap <= self.config.elite_gap_threshold:
            return 0.0

        return (
            (1.0 - math.exp(-skill.frequency / 25.0))
            * (1.0 - math.exp(-skill.frequency / 15.0))
            * effective_gap
            * math.exp(-skill.maturity / 8.0)
        )

    def build_eval_batch(self, buffer: list[Experience]) -> list[Experience]:
        """Select balanced low-reward and high-reward episodes for counterfactual verification."""
        if len(buffer) <= self.config.evolution_threshold:
            return list(buffer)
        sorted_buf = sorted(buffer, key=lambda e: e.reward)
        half = max(1, self.config.evolution_threshold // 2)
        return sorted_buf[:half] + sorted_buf[-half:]

    def evolve_step(
        self,
        current_iter: int,
        baselines: dict[str, float],
        ablation_mode: Literal["full", "no_skill", "no_ppo_gate", "fifo_pruning"] = "full",
        max_evolutions: int = 2,
    ) -> list[EvolutionAuditEntry]:
        """Run one outer Non-Parametric PPO evolution pass across eligible skills."""
        if ablation_mode == "no_skill":
            return []

        all_skills = self.pool.get_all()
        eligible: list[tuple[Skill, float]] = []
        for sk in all_skills:
            buf = self._experience_buffer.get(sk.id, [])
            if len(buf) < self.config.evolution_threshold:
                continue
            prio = self.get_priority(sk, all_skills, current_iter)
            if prio > 0.0:
                eligible.append((sk, prio))

        # If no skill triggered elite gap yet, allow top buffered skill with imperfect episodes
        if not eligible:
            for sk in all_skills:
                buf = self._experience_buffer.get(sk.id, [])
                if len(buf) >= self.config.evolution_threshold and any(ep.reward < 0.95 for ep in buf):
                    eligible.append((sk, 0.02))

        eligible.sort(key=lambda item: item[1], reverse=True)
        entries: list[EvolutionAuditEntry] = []

        for parent_sk, _ in eligible[:max_evolutions]:
            entry = self._evolve_single_skill(parent_sk, current_iter, baselines, ablation_mode)
            if entry is not None:
                entries.append(entry)

        return entries

    def _evolve_single_skill(
        self,
        parent: Skill,
        current_iter: int,
        baselines: dict[str, float],
        ablation_mode: str,
    ) -> EvolutionAuditEntry | None:
        raw_buf = self._experience_buffer.get(parent.id, [])
        eval_exps = self.build_eval_batch(raw_buf)
        if not eval_exps:
            return None

        # Step 1: Per-trajectory hindsight Semantic Gradients
        gradients = [self.executor.synthesize_gradient(parent, exp) for exp in eval_exps]
        related_count = sum(1 for g in gradients if g.is_related)
        relevance_ratio = related_count / float(max(1, len(gradients)))
        operation: Literal["REFINE", "NEW"] = "REFINE" if relevance_ratio >= 0.5 else "NEW"

        self.store.append_event(
            "semantic_gradient",
            {
                "iteration": current_iter,
                "parent_skill": parent.name,
                "operation": operation,
                "relevance_ratio": round(relevance_ratio, 3),
                "diagnoses": [g.diagnosis for g in gradients],
            },
        )

        # Step 2: Best-of-N candidate generation
        candidates = self.executor.synthesize_candidates(
            parent=parent,
            gradients=gradients,
            operation=operation,
            num_candidates=self.config.best_of_n,
        )
        if not candidates:
            return None

        # Step 3: PPO Gate verification (check ablation BEFORE computing logprobs)
        best_cand: Skill = candidates[0]
        best_eval: PPOGateEvaluation

        if ablation_mode == "no_ppo_gate":
            best_eval = PPOGateEvaluation(
                l_clip=0.0,
                margin=self.config.acceptance_margin,
                passed=True,
                clipped_fraction=0.0,
                mean_ratio=1.0,
                min_ratio=1.0,
                max_ratio=1.0,
                old_logprob_mean=0.0,
                new_logprob_mean=0.0,
                step_count=sum(len(e.transitions) for e in eval_exps),
                verifier_mode="bypassed_no_ppo_gate",
                reason="Ablation (no_ppo_gate): candidate admitted unconditionally.",
            )
            accepted = True
        else:
            scored_candidates: list[tuple[Skill, PPOGateEvaluation]] = []
            for cand in candidates:
                gate_eval = self._evaluate_candidate_with_gate(parent, cand, eval_exps, baselines)
                scored_candidates.append((cand, gate_eval))

            best_cand, best_eval = max(scored_candidates, key=lambda item: item[1].improvement)
            accepted = best_eval.passed

        if accepted:
            parent.last_evolved_iter = current_iter
            self.pool.add_skill(parent)

            base_id = re.sub(r"_v\d+$", "", parent.id.split("_cand_")[0])
            promoted = best_cand.model_copy(
                update={
                    "id": f"{base_id}_v{best_cand.version}",
                    "status": SkillStatus.ACTIVE,
                    "maturity": 0,
                    "last_evolved_iter": current_iter,
                }
            )
            self.pool.add_skill(promoted)
            best_cand = promoted
            self._experience_buffer[parent.id] = []
        else:
            keep = max(1, self.config.evolution_threshold // 2)
            self._experience_buffer[parent.id] = raw_buf[-keep:]

        audit_entry = EvolutionAuditEntry(
            audit_id=f"aud_{uuid.uuid4().hex[:10]}",
            iteration=current_iter,
            parent_skill_id=parent.id,
            parent_skill_name=f"{parent.name}@v{parent.version}",
            operation=operation,
            relevance_ratio=relevance_ratio,
            gradients=gradients,
            candidate_skill=best_cand,
            gate_eval=best_eval,
            accepted=accepted,
        )
        self.store.save_audit_log(audit_entry)
        self.store.append_event(
            "ppo_gate_decision",
            {
                "audit_id": audit_entry.audit_id,
                "parent": audit_entry.parent_skill_name,
                "candidate": f"{best_cand.name}@v{best_cand.version}",
                "l_clip": round(best_eval.l_clip, 5),
                "improvement": round(best_eval.improvement, 5),
                "margin": best_eval.margin,
                "clipped_fraction": round(best_eval.clipped_fraction, 3),
                "accepted": accepted,
                "verifier_mode": best_eval.verifier_mode,
                "reason": best_eval.reason,
            },
        )
        return audit_entry

    def _evaluate_candidate_with_gate(
        self,
        old_skill: Skill,
        candidate: Skill,
        eval_exps: list[Experience],
        baselines: dict[str, float],
    ) -> PPOGateEvaluation:
        old_prompts: list[str] = []
        new_prompts: list[str] = []
        targets: list[str] = []
        step_advantages: list[float] = []

        batch_mean = float(np.mean([e.reward for e in eval_exps])) if eval_exps else 0.72
        for exp in eval_exps:
            if not exp.transitions:
                continue
            ref_baseline = baselines.get(exp.env_name, batch_mean)
            # Anchor against SLA target (0.85) when all buffered episodes are suboptimal (< 0.85)
            effective_baseline = max(0.85, ref_baseline) if batch_mean < 0.85 else (0.5 * ref_baseline + 0.5 * batch_mean)
            traj_adv = float(exp.reward - effective_baseline)
            ep_step_adv = traj_adv / float(len(exp.transitions))

            history_so_far: list[dict[str, object]] = []
            for trans in exp.transitions:
                step_adv = (
                    0.35 * ep_step_adv + 0.65 * float(trans.step_reward)
                    if trans.step_reward != 0.0
                    else ep_step_adv
                )
                old_prompts.append(
                    build_executor_prompt(
                        incident_title=exp.scenario_id,
                        observation=trans.state_summary,
                        active_skill=old_skill,
                        available_tools=[],
                        history=history_so_far,
                    )
                )
                new_prompts.append(
                    build_executor_prompt(
                        incident_title=exp.scenario_id,
                        observation=trans.state_summary,
                        active_skill=candidate,
                        available_tools=[],
                        history=history_so_far,
                    )
                )
                targets.append(format_step_target(trans.action, trans.skill_done))
                step_advantages.append(step_adv)
                history_so_far.append(
                    {
                        "tool": trans.action.tool,
                        "args": trans.action.args,
                        "obs": trans.observation_summary[:120],
                    }
                )

        if not targets:
            return compute_clipped_surrogate(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                epsilon=self.config.epsilon,
                margin=self.config.acceptance_margin,
                verifier_mode=self.verifier.mode_name,
            )

        verifier_used = self.verifier
        try:
            old_lps = verifier_used.score_logprobs(old_prompts, targets)
            new_lps = verifier_used.score_logprobs(new_prompts, targets)
            mode_label = verifier_used.mode_name
        except VerifierUnavailableError as exc:
            # Circuit breaker: fall back cleanly to offline proxy if local proxy server is offline
            fallback = HeuristicTokenProxyVerifier()
            old_lps = fallback.score_logprobs(old_prompts, targets)
            new_lps = fallback.score_logprobs(new_prompts, targets)
            mode_label = f"{fallback.mode_name}_fallback({type(exc).__name__})"

        return compute_clipped_surrogate(
            old_log_probs=old_lps,
            new_log_probs=new_lps,
            step_advantages=np.asarray(step_advantages, dtype=np.float64),
            epsilon=self.config.epsilon,
            margin=self.config.acceptance_margin,
            verifier_mode=mode_label,
            relative_to_old=True,
        )

"""Skill-MDP execution engine combining single-turn action+termination with online evolution."""

from __future__ import annotations

import uuid
from typing import Literal

from .config import SkillProConfig
from .environments.sre_triage import SRETriageEnvironment
from .evolver import SkillEvolver
from .llm_backend import ExecutorClient, LogprobVerifier, build_backends
from .models import EvolutionAuditEntry, Experience, Skill, Transition
from .skill_pool import SkillPool
from .storage import SkillStore


class SkillEngine:
    """Coordinates Skill selection, single-call execution, baseline tracking, and evolution."""

    def __init__(
        self,
        config: SkillProConfig | None = None,
        store: SkillStore | None = None,
        executor: ExecutorClient | None = None,
        verifier: LogprobVerifier | None = None,
        seed_if_empty: bool = True,
    ) -> None:
        self.config = config or SkillProConfig()
        self.store = store or SkillStore(self.config.db_path)
        if executor is None or verifier is None:
            built_exec, built_ver = build_backends(self.config)
            self.executor = executor or built_exec
            self.verifier = verifier or built_ver
        else:
            self.executor = executor
            self.verifier = verifier

        self.pool = SkillPool(self.store, self.config, seed_if_empty=seed_if_empty)
        self.evolver = SkillEvolver(self.store, self.pool, self.executor, self.verifier, self.config)
        self.env = SRETriageEnvironment(max_steps=self.config.max_episode_steps)
        self.baselines: dict[str, float] = self.store.get_baselines()

    def reconfigure_backends(
        self,
        mode: Literal["simulator", "hybrid"] | None = None,
        proxy_verifier_mode: Literal["offline_proxy", "local_openai", "local_hf"] | None = None,
    ) -> None:
        """Reload executor and proxy verifier when runtime settings change."""
        if mode is not None:
            self.config.mode = mode
        if proxy_verifier_mode is not None:
            self.config.proxy_verifier_mode = proxy_verifier_mode
        built_exec, built_ver = build_backends(self.config)
        self.executor = built_exec
        self.verifier = built_ver
        self.evolver.executor = built_exec
        self.evolver.verifier = built_ver

    def run_episode(
        self,
        scenario_id: str | None = None,
        ablation_mode: Literal["full", "no_skill", "no_ppo_gate", "fifo_pruning"] = "full",
    ) -> Experience:
        """Execute a single Skill-MDP incident triage episode and persist the trajectory."""
        scenario_info, observation = self.env.reset(scenario_id=scenario_id)
        episode_id = f"ep_{uuid.uuid4().hex[:10]}"
        baseline_at_run = self.baselines.get(scenario_info.category, 0.72)

        self.store.append_event(
            "episode_started",
            {
                "episode_id": episode_id,
                "scenario_id": scenario_info.scenario_id,
                "category": scenario_info.category,
                "title": scenario_info.title,
                "severity": scenario_info.severity,
                "ablation_mode": ablation_mode,
            },
        )

        active_skill: Skill | None = None
        skill_age = 0
        transitions: list[Transition] = []
        history: list[dict[str, object]] = []
        skill_call_counts: dict[str, int] = {}
        total_prompt_tokens = 0
        final_reward = 0.0
        final_resolved = False

        tool_schemas = self.env.tool_schemas()

        for step_idx in range(1, self.config.max_episode_steps + 1):
            # Select new Skill only when none is active (SMDP hierarchical execution)
            if ablation_mode != "no_skill" and active_skill is None:
                selected, sel_meta = self.pool.select_skill(f"{scenario_info.title}. {observation}")
                if selected is not None:
                    active_skill = selected
                    skill_age = 0
                    self.store.append_event(
                        "skill_selected",
                        {
                            "episode_id": episode_id,
                            "step": step_idx,
                            "skill_id": selected.id,
                            "skill_name": f"{selected.name}@v{selected.version}",
                            "meta": sel_meta,
                        },
                    )

            action, skill_done, thought, step_tokens = self.executor.decide_step(
                incident_title=f"{scenario_info.title} ({scenario_info.scenario_id})",
                observation=observation,
                active_skill=active_skill if ablation_mode != "no_skill" else None,
                available_tools=tool_schemas,
                history=history,
            )
            total_prompt_tokens += step_tokens

            if active_skill is not None:
                skill_call_counts[active_skill.id] = skill_call_counts.get(active_skill.id, 0) + 1
                skill_age += 1
                if skill_age >= self.config.max_skill_age:
                    skill_done = True

            step_result = self.env.step(action)

            # Map terminal close_incident outcome into a per-step credit signal for PPO Gate
            step_credit = step_result.reward
            if action.tool == "close_incident":
                if "verified_healthy=True" in step_result.observation:
                    step_credit = 0.25
                elif step_result.resolved:
                    step_credit = -0.25
                else:
                    step_credit = -0.35

            transitions.append(
                Transition(
                    step_index=step_idx,
                    state_summary=observation,
                    action=action,
                    observation_summary=step_result.observation,
                    skill_id=active_skill.id if active_skill else None,
                    skill_name=f"{active_skill.name}@v{active_skill.version}" if active_skill else None,
                    skill_done=skill_done,
                    step_reward=step_credit,
                    prompt_tokens=step_tokens,
                )
            )
            history.append(
                {
                    "step": step_idx,
                    "tool": action.tool,
                    "args": action.args,
                    "obs": step_result.observation,
                }
            )

            self.store.append_event(
                "step_executed",
                {
                    "episode_id": episode_id,
                    "step": step_idx,
                    "tool": action.tool,
                    "args": action.args,
                    "skill": f"{active_skill.name}@v{active_skill.version}" if active_skill else "None",
                    "skill_done": skill_done,
                    "thought": thought,
                    "observation": step_result.observation,
                    "prompt_tokens": step_tokens,
                },
            )

            observation = step_result.observation
            if skill_done:
                active_skill = None
                skill_age = 0

            if step_result.done:
                final_reward = step_result.reward
                final_resolved = step_result.resolved
                break
        else:
            final_reward, final_resolved = self.env.finalize_reward()

        # Attribute outcome back to invoked Skills
        total_skill_calls = sum(skill_call_counts.values())
        skills_used_names: list[str] = []
        skill_ids_used: list[str] = []
        if total_skill_calls > 0 and ablation_mode != "no_skill":
            for sk_id, count in skill_call_counts.items():
                sk = self.pool.get_by_id(sk_id)
                if sk is not None:
                    sk.update_stats(
                        reward=final_reward,
                        baseline=baseline_at_run,
                        total_skill_calls_in_traj=total_skill_calls,
                        skill_call_count_in_traj=count,
                    )
                    skills_used_names.append(f"{sk.name}@v{sk.version}")
                    skill_ids_used.append(sk.id)
            self.pool.persist_all()

        # Update category EMA baseline
        prev_base = self.baselines.get(scenario_info.category, final_reward)
        alpha = self.config.ema_alpha
        new_base = (1.0 - alpha) * prev_base + alpha * final_reward
        self.baselines[scenario_info.category] = new_base
        self.store.set_baseline(scenario_info.category, new_base)

        exp = Experience(
            episode_id=episode_id,
            scenario_id=scenario_info.scenario_id,
            env_name=scenario_info.category,
            ablation_mode=ablation_mode,
            reward=final_reward,
            baseline_at_run=baseline_at_run,
            resolved=final_resolved,
            skills_used=skills_used_names,
            skill_ids_used=skill_ids_used,
            step_count=len(transitions),
            prompt_tokens=total_prompt_tokens,
            episodic_baseline_tokens=self.env.current_episodic_baseline_tokens() * max(1, len(transitions) // 2),
            transitions=transitions,
        )
        self.store.save_experience(exp)
        self.store.append_event(
            "episode_finished",
            {
                "episode_id": episode_id,
                "scenario_id": scenario_info.scenario_id,
                "reward": final_reward,
                "resolved": final_resolved,
                "steps": len(transitions),
                "skills_used": skills_used_names,
                "prompt_tokens": total_prompt_tokens,
                "episodic_baseline_tokens": exp.episodic_baseline_tokens,
            },
        )
        return exp

    def run_batch(
        self,
        num_episodes: int = 6,
        scenario_id: str | None = None,
        ablation_mode: Literal["full", "no_skill", "no_ppo_gate", "fifo_pruning"] = "full",
        evolve_every: int = 4,
    ) -> dict[str, object]:
        """Run a batch of incident episodes and execute Non-Parametric PPO updates."""
        episodes: list[Experience] = []
        audits: list[EvolutionAuditEntry] = []
        pruning_logs: list[str] = []

        for idx in range(num_episodes):
            ep = self.run_episode(scenario_id=scenario_id, ablation_mode=ablation_mode)
            episodes.append(ep)
            self.evolver.ingest_experiences([ep])

            should_evolve = (
                ablation_mode != "no_skill"
                and (((idx + 1) % max(1, evolve_every) == 0) or (idx == num_episodes - 1))
            )
            if should_evolve:
                current_iter = self.store.get_iteration() + 1
                self.store.set_iteration(current_iter)
                self.pool.increment_maturities()

                step_audits = self.evolver.evolve_step(
                    current_iter=current_iter,
                    baselines=self.baselines,
                    ablation_mode=ablation_mode,
                )
                audits.extend(step_audits)

                pruned = self.pool.maintain(fifo_only=(ablation_mode == "fifo_pruning"))
                if pruned:
                    pruning_logs.extend(pruned)
                    self.store.append_event(
                        "pool_maintained",
                        {"iteration": current_iter, "actions": pruned, "pool_size": len(self.pool.get_all())},
                    )

        return {
            "episodes_run": len(episodes),
            "mean_reward": round(sum(e.reward for e in episodes) / max(1, len(episodes)), 4),
            "resolution_rate": round(sum(1 for e in episodes if e.resolved) / max(1, len(episodes)), 4),
            "evolutions_triggered": len(audits),
            "accepted_count": sum(1 for a in audits if a.accepted),
            "pruning_actions": pruning_logs,
        }

    def reset_sandbox(self) -> None:
        """Reset SQLite state and re-seed initial SRE starter skills."""
        self.store.reset_all()
        self.baselines.clear()
        self.evolver._experience_buffer.clear()
        self.evolver._seen_episode_ids.clear()
        self.pool = SkillPool(self.store, self.config, seed_if_empty=True)
        self.evolver.pool = self.pool
        self.store.append_event("sandbox_reset", {"pool_size": len(self.pool.get_all())})

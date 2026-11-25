"""Integration tests for SRE Triage Environment, SkillEngine, Evolution, and Hybrid Fallback."""

from skill_pro_rw.config import SkillProConfig
from skill_pro_rw.environments.sre_triage import SRETriageEnvironment
from skill_pro_rw.models import ToolCall
from skill_pro_rw.runner import SkillEngine
from skill_pro_rw.storage import SkillStore


def test_sre_environment_reward_differentiation() -> None:
    """Verified remediation scores >= 0.94, unverified remediation ~0.66, wrong fix 0.22."""
    env = SRETriageEnvironment()

    # Optimal path with verify_health + close_incident
    env.reset("db-pool-exhaustion")
    env.step(ToolCall(tool="query_logs", args={"service": "checkout-api"}))
    env.step(ToolCall(tool="get_metrics", args={"service": "checkout-api"}))
    env.step(ToolCall(tool="apply_remediation", args={"action": "reset_idle_connections", "target": "checkout-api"}))
    verify_res = env.step(ToolCall(tool="verify_health", args={"service": "checkout-api"}))
    assert verify_res.done is False
    res = env.step(ToolCall(tool="close_incident", args={"summary": "Pool reset and verified."}))
    assert res.done is True
    assert res.resolved is True
    assert res.reward >= 0.95

    # Unverified remediation (forgot verify_health before closing)
    env.reset("db-pool-exhaustion")
    env.step(ToolCall(tool="apply_remediation", args={"action": "reset_idle_connections", "target": "checkout-api"}))
    unverified_res = env.step(ToolCall(tool="close_incident", args={"summary": "Closed without verify."}))
    assert unverified_res.done is True
    assert unverified_res.resolved is True
    assert unverified_res.reward == 0.66

    # Wrong remediation
    env.reset("bad-canary-deploy")
    env.step(ToolCall(tool="apply_remediation", args={"action": "restart_service", "target": "payment-webhook"}))
    wrong_reward, wrong_resolved = env.finalize_reward()
    assert wrong_resolved is False
    assert wrong_reward == 0.22


def test_skill_engine_evolves_skills_via_ppo_gate_and_improves_over_no_skill() -> None:
    """Running a batch triggers semantic gradients + PPO Gate and outperforms stateless baseline."""
    cfg_full = SkillProConfig(db_path=":memory:", explore_epsilon=0.0, evolution_threshold=2)
    engine_full = SkillEngine(config=cfg_full, store=SkillStore(":memory:"))
    summary_full = engine_full.run_batch(num_episodes=8, ablation_mode="full", evolve_every=4)

    cfg_base = SkillProConfig(db_path=":memory:", explore_epsilon=0.0)
    engine_base = SkillEngine(config=cfg_base, store=SkillStore(":memory:"))
    summary_base = engine_base.run_batch(num_episodes=6, ablation_mode="no_skill")

    audits = engine_full.store.list_audit_logs()
    assert summary_full["evolutions_triggered"] >= 1
    assert len(audits) >= 1
    assert any(a.accepted for a in audits)
    assert float(summary_full["mean_reward"]) > float(summary_base["mean_reward"])

    # Verify token savings are recorded
    exps = engine_full.store.list_experiences()
    assert all(e.prompt_tokens < e.episodic_baseline_tokens for e in exps)
    engine_full.store.close()
    engine_base.store.close()


def test_no_ppo_gate_ablation_does_not_crash() -> None:
    """Ensure no_ppo_gate bypasses logprob computation cleanly."""
    cfg = SkillProConfig(db_path=":memory:", explore_epsilon=0.0, evolution_threshold=2)
    engine = SkillEngine(config=cfg, store=SkillStore(":memory:"))
    summary = engine.run_batch(
        num_episodes=4, scenario_id="db-pool-exhaustion", ablation_mode="no_ppo_gate", evolve_every=2
    )

    audits = engine.store.list_audit_logs()
    assert summary["evolutions_triggered"] >= 1
    assert all(a.gate_eval.verifier_mode == "bypassed_no_ppo_gate" for a in audits)
    engine.store.close()


def test_hybrid_unreachable_proxy_verifier_fails_over_gracefully() -> None:
    """Unreachable local_openai verifier falls back to offline_proxy rather than crashing."""
    cfg = SkillProConfig(
        db_path=":memory:",
        mode="simulator",
        proxy_verifier_mode="local_openai",
        proxy_verifier_url="http://127.0.0.1:59999/v1",
        explore_epsilon=0.0,
        evolution_threshold=2,
    )
    engine = SkillEngine(config=cfg, store=SkillStore(":memory:"))
    summary = engine.run_batch(
        num_episodes=4, scenario_id="bad-canary-deploy", ablation_mode="full", evolve_every=2
    )

    audits = engine.store.list_audit_logs()
    assert summary["evolutions_triggered"] >= 1
    assert any("fallback" in a.gate_eval.verifier_mode for a in audits)
    engine.store.close()

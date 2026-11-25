"""Unit tests for SkillPool selection, persistence, semantic deduplication, and pruning."""

from pathlib import Path

from skill_pro_rw.config import SkillProConfig
from skill_pro_rw.models import Skill, SkillStatus
from skill_pro_rw.skill_pool import SkillPool
from skill_pro_rw.storage import SkillStore


def _make_skill(
    skill_id: str,
    name: str,
    initiation: str,
    version: int = 1,
    avg_gain: float = 0.0,
    frequency: int = 5,
    maturity: int = 4,
) -> Skill:
    return Skill(
        id=skill_id,
        name=name,
        version=version,
        initiation=initiation,
        policy=["Check logs", "Verify health"],
        termination="Stop when healthy",
        avg_gain=avg_gain,
        total_gain=avg_gain * frequency,
        frequency=frequency,
        maturity=maturity,
        status=SkillStatus.ACTIVE,
    )


def test_skill_pool_persists_and_selects_matching_skill(tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_pool.db")
    cfg = SkillProConfig(db_path=db_file, explore_epsilon=0.0)
    store = SkillStore(db_file)
    pool = SkillPool(store, cfg, seed_if_empty=True)

    assert len(pool.get_active()) == 4

    chosen, meta = pool.select_skill("checkout-api 504 gateway timeout database connection pool exhausted", epsilon=0.0)
    assert chosen is not None
    assert chosen.name == "DatabasePoolDiagnosis"
    assert meta["strategy"] == "similarity_lcb"

    # Re-open from disk to confirm persistence
    store.close()
    reopened_store = SkillStore(db_file)
    reopened_pool = SkillPool(reopened_store, cfg, seed_if_empty=False)
    assert len(reopened_pool.get_active()) == 4
    reopened_store.close()


def test_skill_pool_selection_prefers_newer_version_on_tie() -> None:
    cfg = SkillProConfig(db_path=":memory:", explore_epsilon=0.0)
    store = SkillStore(":memory:")
    pool = SkillPool(store, cfg, seed_if_empty=False)

    v1 = _make_skill("sk_a_v1", "CanaryDeployTriage", "Sudden 502 errors after canary deploy", version=1, maturity=1)
    v2 = _make_skill("sk_a_v2", "CanaryDeployTriage", "Sudden 502 errors after canary deploy", version=2, maturity=0)
    pool.add_skill(v1)
    pool.add_skill(v2)

    chosen, _ = pool.select_skill("Sudden 502 errors after canary deploy", epsilon=0.0)
    assert chosen is not None
    assert chosen.id == "sk_a_v2"
    store.close()


def test_skill_pool_maintenance_dedup_and_capacity_trim() -> None:
    cfg = SkillProConfig(db_path=":memory:", pool_size=3, dedup_similarity_threshold=0.95)
    store = SkillStore(":memory:")
    pool = SkillPool(store, cfg, seed_if_empty=False)

    # Identical initiation triggers semantic/variant dedup when over capacity
    pool.add_skill(_make_skill("sk_db_v1", "DatabaseCheck", "Database pool exhausted 504 timeout", version=1, avg_gain=0.02, frequency=12, maturity=5))
    pool.add_skill(_make_skill("sk_db_v2", "DatabaseCheck", "Database pool exhausted 504 timeout", version=2, avg_gain=0.25, frequency=15, maturity=5))
    pool.add_skill(_make_skill("sk_bad", "BadSkill", "Unrelated slow query check", version=1, avg_gain=-0.30, frequency=10, maturity=5))
    pool.add_skill(_make_skill("sk_tls", "TLSRotate", "x509 certificate expired on ingress gateway", version=1, avg_gain=0.18, frequency=11, maturity=4))
    pool.add_skill(_make_skill("sk_new", "NewbornSkill", "Kubernetes CrashLoopBackOff OOMKilled exit 137", version=1, avg_gain=0.0, frequency=1, maturity=1))

    pruned = pool.maintain(fifo_only=False)
    remaining_ids = {sk.id for sk in pool.get_all()}

    assert len(remaining_ids) <= 3
    assert "sk_db_v2" in remaining_ids
    assert "sk_bad" not in remaining_ids
    assert len(pruned) >= 2
    store.close()


def test_skill_pool_fifo_ablation_drops_oldest() -> None:
    cfg = SkillProConfig(db_path=":memory:", pool_size=2)
    store = SkillStore(":memory:")
    pool = SkillPool(store, cfg, seed_if_empty=False)

    pool.add_skill(_make_skill("sk_1", "FirstSkill", "First alert", avg_gain=0.9))
    pool.add_skill(_make_skill("sk_2", "SecondSkill", "Second alert", avg_gain=0.5))
    pool.add_skill(_make_skill("sk_3", "ThirdSkill", "Third alert", avg_gain=0.1))

    pruned = pool.maintain(fifo_only=True)
    assert [sk.id for sk in pool.get_all()] == ["sk_2", "sk_3"]
    assert pruned == ["fifo_drop:FirstSkill@v1"]
    store.close()

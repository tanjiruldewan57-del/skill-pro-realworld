"""Persistent procedural Skill pool with LCB selection and 5-stage maintenance."""

from __future__ import annotations

import random
import re
from typing import Sequence

import numpy as np

from .config import SkillProConfig
from .embeddings import LocalHashingNGramEmbedder, TextEmbedder, cosine_similarity
from .models import Skill, SkillStatus
from .storage import SkillStore

_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def default_starter_skills() -> list[Skill]:
    """Build compact starter SRE skills that can execute immediately and evolve via NP-PPO."""
    return [
        Skill(
            id="sk_log_triage_v1",
            name="GenericLogTriage",
            version=1,
            initiation="Service error spike, 5xx HTTP responses, or unexplained timeout alerts.",
            policy=[
                "Inspect service logs via query_logs for the failing service.",
                "Apply a remediation action based on the primary error pattern.",
            ],
            termination="Stop once remediation is attempted or root cause cannot be matched.",
            allowed_tools=["query_logs", "apply_remediation", "close_incident"],
            status=SkillStatus.ACTIVE,
        ),
        Skill(
            id="sk_db_pool_v1",
            name="DatabasePoolDiagnosis",
            version=1,
            initiation="Database timeout, connection pool exhausted, or checkout 504 gateway errors.",
            policy=[
                "Run query_logs on the impacted service to confirm pool timeout errors.",
                "Inspect connection utilization via get_metrics.",
                "Apply reset_idle_connections or rollback_config remediation.",
            ],
            termination="Stop after database remediation is applied.",
            allowed_tools=["query_logs", "get_metrics", "apply_remediation", "verify_health", "close_incident"],
            status=SkillStatus.ACTIVE,
        ),
        Skill(
            id="sk_canary_rollback_v1",
            name="CanaryDeployTriage",
            version=1,
            initiation="Sudden 502 errors, KeyError exceptions, or crash right after a recent release.",
            policy=[
                "Check query_logs for exception tracebacks.",
                "Check get_deploy_history to verify if a recent canary deployment occurred.",
                "Execute rollback_canary via apply_remediation if regression matches deploy.",
            ],
            termination="Stop when canary traffic is rolled back.",
            allowed_tools=["query_logs", "get_deploy_history", "apply_remediation", "verify_health", "close_incident"],
            status=SkillStatus.ACTIVE,
        ),
        Skill(
            id="sk_k8s_infra_v1",
            name="KubernetesPodRecovery",
            version=1,
            initiation="Pod CrashLoopBackOff, OOMKilled exit code 137, DNS NXDOMAIN, or TLS cert errors.",
            policy=[
                "Inspect pod status or metrics via inspect_k8s_pod and get_metrics.",
                "Query service logs to distinguish OOM, CoreDNS throttling, or expired TLS certs.",
                "Apply targeted infra remediation (increase_memory_limit, scale_coredns, or rotate_tls_cert).",
            ],
            termination="Stop once pod or gateway remediation finishes.",
            allowed_tools=["inspect_k8s_pod", "get_metrics", "query_logs", "apply_remediation", "verify_health", "close_incident"],
            status=SkillStatus.ACTIVE,
        ),
    ]


class SkillPool:
    """Manages active/candidate/archived procedural skills and online pool maintenance."""

    def __init__(
        self,
        store: SkillStore,
        config: SkillProConfig,
        embedder: TextEmbedder | None = None,
        seed_if_empty: bool = True,
    ) -> None:
        self.store = store
        self.config = config
        self.embedder: TextEmbedder = embedder or LocalHashingNGramEmbedder()
        self._skills: list[Skill] = self.store.list_skills()
        self._embeddings: dict[str, np.ndarray] = {}

        if seed_if_empty and not self._skills:
            self._skills = default_starter_skills()
            self.store.upsert_skills(self._skills)

        self._refresh_embeddings()

    def _refresh_embeddings(self) -> None:
        """Compute normalized embeddings for all skill initiation conditions."""
        self._embeddings = {
            sk.id: self.embedder.embed(f"{sk.name}. {sk.initiation}")
            for sk in self._skills
        }

    def get_all(self, include_archived: bool = False) -> list[Skill]:
        """Return skills currently in the pool."""
        if include_archived:
            return list(self._skills)
        return [sk for sk in self._skills if sk.status != SkillStatus.ARCHIVED]

    def get_active(self) -> list[Skill]:
        """Return only ACTIVE skills eligible for live execution."""
        return [sk for sk in self._skills if sk.status == SkillStatus.ACTIVE]

    def get_by_id(self, skill_id: str) -> Skill | None:
        """Look up a skill by ID."""
        for sk in self._skills:
            if sk.id == skill_id:
                return sk
        return None

    def get_by_name(self, name: str) -> Skill | None:
        """Look up the newest non-archived skill matching a given name."""
        for sk in reversed(self._skills):
            if sk.name == name and sk.status != SkillStatus.ARCHIVED:
                return sk
        return None

    def add_skill(self, skill: Skill) -> None:
        """Add or replace a skill in memory and persistent SQLite storage."""
        existing_idx = next((i for i, item in enumerate(self._skills) if item.id == skill.id), None)
        if existing_idx is not None:
            self._skills[existing_idx] = skill
        else:
            self._skills.append(skill)
        self._embeddings[skill.id] = self.embedder.embed(f"{skill.name}. {skill.initiation}")
        self.store.upsert_skill(skill)

    def set_status(self, skill_id: str, new_status: SkillStatus) -> Skill | None:
        """Update lifecycle status (CANDIDATE / ACTIVE / ARCHIVED) and persist."""
        skill = self.get_by_id(skill_id)
        if skill is None:
            return None
        skill.status = new_status
        self.store.upsert_skill(skill)
        return skill

    def persist_all(self) -> None:
        """Persist current in-memory skills after stat updates."""
        self.store.upsert_skills(self._skills)

    def increment_maturities(self) -> None:
        """Increment iteration maturity across all non-archived skills."""
        for sk in self._skills:
            if sk.status != SkillStatus.ARCHIVED:
                sk.maturity += 1
        self.persist_all()

    def select_skill(
        self,
        state_text: str,
        epsilon: float | None = None,
        min_sim: float = 0.08,
        min_lcb: float = -0.35,
    ) -> tuple[Skill | None, dict[str, object]]:
        """Select the best ACTIVE skill using initiation similarity + LCB ranking.

        Matches Skill-Pro's hybrid selection: filter top-K initiation matches, rank by
        Lower Confidence Bound (LCB), and protect newborn skills (maturity < 3).
        """
        active = self.get_active()
        if not active:
            return None, {"reason": "no_active_skills"}

        eps = self.config.explore_epsilon if epsilon is None else epsilon
        if random.random() < eps:
            chosen = random.choice([*active, None])
            return chosen, {
                "strategy": "epsilon_explore",
                "chosen": chosen.name if chosen else None,
            }

        state_vec = self.embedder.embed(state_text)
        max_maturity = max((sk.maturity for sk in active), default=1)

        scored: list[tuple[Skill, float, float]] = []
        for sk in active:
            sk_vec = self._embeddings.get(sk.id)
            if sk_vec is None:
                sk_vec = self.embedder.embed(f"{sk.name}. {sk.initiation}")
                self._embeddings[sk.id] = sk_vec
            sim = cosine_similarity(state_vec, sk_vec)
            lcb_val = sk.lcb(max_maturity=max_maturity, beta=self.config.lcb_beta)
            if sim >= min_sim:
                scored.append((sk, sim, lcb_val))

        if not scored:
            return None, {"strategy": "similarity_lcb", "reason": "below_similarity_threshold"}

        # Take top-3 semantic matches, then pick highest LCB (with warmup protection)
        scored.sort(key=lambda item: item[1], reverse=True)
        top_nominees = scored[:3]
        best_sk, best_sim, best_lcb = max(
            top_nominees,
            key=lambda item: (
                (item[1] * 0.4 + item[2]) if item[0].maturity >= 3 else item[1],
                item[0].version,
            ),
        )

        if best_sk.maturity >= 3 and best_lcb < min_lcb:
            return None, {
                "strategy": "similarity_lcb",
                "reason": "below_lcb_guard",
                "candidate": best_sk.name,
                "lcb": round(best_lcb, 4),
            }

        return best_sk, {
            "strategy": "similarity_lcb",
            "similarity": round(best_sim, 4),
            "lcb": round(best_lcb, 4),
            "maturity": best_sk.maturity,
        }

    def maintain(self, fifo_only: bool = False) -> list[str]:
        """Run pool maintenance (or FIFO pruning for ablation) and persist changes.

        Returns a list of human-readable pruning actions for the backend audit trail.
        """
        non_archived = [sk for sk in self._skills if sk.status != SkillStatus.ARCHIVED]
        archived = [sk for sk in self._skills if sk.status == SkillStatus.ARCHIVED]
        max_size = self.config.pool_size
        removed_log: list[str] = []

        if fifo_only:
            if len(non_archived) > max_size:
                overflow = len(non_archived) - max_size
                dropped = non_archived[:overflow]
                non_archived = non_archived[overflow:]
                for sk in dropped:
                    removed_log.append(f"fifo_drop:{sk.name}@v{sk.version}")
            self._skills = non_archived + archived
            self.store.replace_all_skills(self._skills)
            self._refresh_embeddings()
            return removed_log

        max_maturity = max((sk.maturity for sk in non_archived), default=1)

        def lcb_score(sk: Skill) -> float:
            return sk.lcb(max_maturity=max_maturity, beta=self.config.lcb_beta)

        # Stage 1: Drop empty/invalid skills
        valid_skills: list[Skill] = []
        for sk in non_archived:
            if not sk.name or (not sk.initiation and not sk.policy and not sk.termination):
                removed_log.append(f"invalid_drop:{sk.id}")
                continue
            valid_skills.append(sk)

        # Stage 2: Exact name + semantic cosine deduplication (>= 0.95)
        deduped = self._dedup_by_name_and_semantics(valid_skills, lcb_score, max_size, removed_log)

        # Stage 3: Outdated variant dedup among mature skills (maturity >= 4)
        deduped = self._dedup_variants(deduped, lcb_score, max_size, removed_log)

        # Stage 4: Negative LCB pruning on mature, well-tested skills
        while len(deduped) > max_size:
            candidates = [
                sk for sk in deduped
                if sk.maturity > 2 and sk.frequency >= 5 and lcb_score(sk) < -0.001
            ]
            if not candidates:
                break
            worst = min(candidates, key=lcb_score)
            deduped.remove(worst)
            removed_log.append(f"neg_lcb_drop:{worst.name}@v{worst.version}({lcb_score(worst):.3f})")

        # Stage 5: Tiered capacity trim (veterans -> mid_age -> newcomers)
        if len(deduped) > max_size:
            deduped = self._tiered_capacity_trim(deduped, lcb_score, max_size, removed_log)

        self._skills = deduped + archived
        self.store.replace_all_skills(self._skills)
        self._refresh_embeddings()
        return removed_log

    def _dedup_by_name_and_semantics(
        self,
        skills: list[Skill],
        lcb_score: object,
        max_size: int,
        removed_log: list[str],
    ) -> list[Skill]:
        if len(skills) <= max_size:
            return skills

        # Exact-name dedup preserving highest LCB
        by_name: dict[str, list[Skill]] = {}
        for sk in skills:
            by_name.setdefault(sk.name, []).append(sk)

        kept: list[Skill] = []
        for _, group in by_name.items():
            if len(group) == 1:
                kept.append(group[0])
                continue
            group.sort(key=lcb_score, reverse=True)  # type: ignore[arg-type]
            kept.append(group[0])
            for dropped in group[1:]:
                removed_log.append(f"exact_dedup:{dropped.name}@v{dropped.version}")

        if len(kept) <= max_size:
            return kept

        # Semantic cosine dedup (>= threshold) protecting newborn skills (maturity < 3)
        thr = self.config.dedup_similarity_threshold
        self._refresh_embeddings()
        final_kept: list[Skill] = []
        for sk in sorted(kept, key=lcb_score, reverse=True):  # type: ignore[arg-type]
            sk_vec = self.embedder.embed(f"{sk.name}. {sk.initiation}")
            duplicate_found = False
            if sk.maturity >= 3:
                for existing in final_kept:
                    if existing.maturity < 3:
                        continue
                    ex_vec = self.embedder.embed(f"{existing.name}. {existing.initiation}")
                    if cosine_similarity(sk_vec, ex_vec) >= thr:
                        duplicate_found = True
                        removed_log.append(f"semantic_dedup:{sk.name}@v{sk.version}")
                        break
            if not duplicate_found:
                final_kept.append(sk)
        return final_kept

    def _dedup_variants(
        self,
        skills: list[Skill],
        lcb_score: object,
        max_size: int,
        removed_log: list[str],
    ) -> list[Skill]:
        if len(skills) <= max_size:
            return skills

        by_base: dict[str, list[Skill]] = {}
        for sk in skills:
            base = _VERSION_SUFFIX_RE.sub("", sk.name)
            by_base.setdefault(base, []).append(sk)

        kept: list[Skill] = []
        for _, group in by_base.items():
            mature = [sk for sk in group if sk.maturity >= 4]
            immature = [sk for sk in group if sk.maturity < 4]
            if len(mature) <= 1:
                kept.extend(group)
                continue
            mature.sort(key=lcb_score, reverse=True)  # type: ignore[arg-type]
            kept.append(mature[0])
            kept.extend(immature)
            for dropped in mature[1:]:
                removed_log.append(f"variant_dedup:{dropped.name}@v{dropped.version}")
        return kept

    def _tiered_capacity_trim(
        self,
        skills: Sequence[Skill],
        lcb_score: object,
        max_size: int,
        removed_log: list[str],
    ) -> list[Skill]:
        veterans = sorted(
            [sk for sk in skills if sk.frequency >= 10],
            key=lcb_score,  # type: ignore[arg-type]
            reverse=True,
        )
        mid_age = sorted(
            [sk for sk in skills if 3 <= sk.frequency < 10],
            key=lcb_score,  # type: ignore[arg-type]
            reverse=True,
        )
        newcomers = sorted(
            [sk for sk in skills if sk.frequency < 3],
            key=lcb_score,  # type: ignore[arg-type]
            reverse=True,
        )
        ordered = veterans + mid_age + newcomers
        kept = ordered[:max_size]
        for dropped in ordered[max_size:]:
            removed_log.append(f"capacity_trim:{dropped.name}@v{dropped.version}")
        return kept

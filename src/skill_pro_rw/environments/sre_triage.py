"""Deterministic SRE/DevOps Incident Triage sandbox environment with 6 failure classes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..models import ToolCall
from .base import ScenarioInfo, StepResult, ToolSpec


@dataclass(frozen=True)
class _IncidentSpec:
    info: ScenarioInfo
    service: str
    logs: str
    metrics: str
    deploy_history: str
    pod_status: str
    expected_remediation: str
    episodic_raw_tokens: int


_INCIDENTS: list[_IncidentSpec] = [
    _IncidentSpec(
        info=ScenarioInfo(
            scenario_id="db-pool-exhaustion",
            category="database",
            title="Checkout API 504 Gateway Timeout Burst",
            severity="SEV-1",
            initial_alert="[ALERT] checkout-api p99 latency > 12s and 504 rate at 38% across us-east-1.",
        ),
        service="checkout-api",
        logs=(
            "ERROR checkout-api: psycopg2.OperationalError: connection pool exhausted "
            "(pg_pool_active=100/100, waiting_threads=46, idle_in_tx=34)."
        ),
        metrics="pg_pool_active=100/100; db_cpu=42%; idle_in_transaction_connections=34; p99_ms=12400",
        deploy_history="No deployments in the past 48h. Last stable release: checkout-api v4.11.0.",
        pod_status="checkout-api-pod-7f9b: Running (restarts=0, mem=510Mi/1Gi).",
        expected_remediation="reset_idle_connections",
        episodic_raw_tokens=2450,
    ),
    _IncidentSpec(
        info=ScenarioInfo(
            scenario_id="redis-oom-eviction",
            category="cache",
            title="Auth Cache Latency Spike & Evictions",
            severity="SEV-2",
            initial_alert="[ALERT] auth-cache p99 latency surged to 4200ms; login timeouts increasing.",
        ),
        service="auth-cache",
        logs=(
            "WARN auth-cache: cache miss storm detected for session:* keys; upstream auth DB load 4x. "
            "redis_evicted_keys surging."
        ),
        metrics="redis_memory_used=99.2%; evicted_keys=18400/min; hit_ratio=0.31; p99_ms=4200",
        deploy_history="auth-service v3.2.1 deployed 12h ago (added unbounded session metadata payload).",
        pod_status="auth-cache-0: Running (restarts=0, mem=3.97Gi/4Gi).",
        expected_remediation="enforce_session_ttl",
        episodic_raw_tokens=2680,
    ),
    _IncidentSpec(
        info=ScenarioInfo(
            scenario_id="bad-canary-deploy",
            category="deployment",
            title="Payment Webhook 502 After Canary Rollout",
            severity="SEV-1",
            initial_alert="[ALERT] payment-webhook 502 Bad Gateway rate hit 24% on canary slice.",
        ),
        service="payment-webhook",
        logs=(
            "ERROR payment-webhook: Traceback KeyError: 'currency_code' in process_stripe_event() "
            "on pod payment-webhook-canary-v2.14.3."
        ),
        metrics="http_502_rate=24.1% (canary_only=true); stable_502_rate=0.0%; error_budget_burn=6.2x",
        deploy_history="DEPLOY 8m ago: payment-webhook v2.14.3-canary (25% traffic) by ci-bot.",
        pod_status="payment-webhook-canary: Running (restarts=2, exit_code=1 on worker thread).",
        expected_remediation="rollback_canary",
        episodic_raw_tokens=2910,
    ),
    _IncidentSpec(
        info=ScenarioInfo(
            scenario_id="k8s-crashloop-oom",
            category="kubernetes",
            title="Worker Billing Pod CrashLoopBackOff",
            severity="SEV-2",
            initial_alert="[ALERT] worker-billing deployment available replicas dropped from 4 to 1.",
        ),
        service="worker-billing",
        logs=(
            "ERROR worker-billing: Killed process during invoice PDF aggregation batch (batch_size=5000)."
        ),
        metrics="container_memory_working_set=1024Mi/1024Mi; cpu=65%; queue_depth=8420",
        deploy_history="Config update 25m ago increased invoice batch_size from 500 to 5000.",
        pod_status="worker-billing-worker: CrashLoopBackOff (Last State: Terminated, Reason: OOMKilled, ExitCode 137).",
        expected_remediation="increase_memory_limit",
        episodic_raw_tokens=2540,
    ),
    _IncidentSpec(
        info=ScenarioInfo(
            scenario_id="dns-coredns-throttle",
            category="networking",
            title="Cluster-Wide Intermittent NXDOMAIN & DNS Timeouts",
            severity="SEV-1",
            initial_alert="[ALERT] Multiple services reporting intermittent DNS lookup timeout & NXDOMAIN spikes.",
        ),
        service="coredns",
        logs=(
            "WARN coredns: UDP lookup delayed > 2500ms; upstream forwarder queue full; intermittent NXDOMAIN."
        ),
        metrics="coredns_cpu_throttled=89%; conntrack_entries=98.7%; dns_p99_ms=2800",
        deploy_history="Traffic autoscaled 2x during flash sale; CoreDNS static at 2 replicas.",
        pod_status="coredns-7c9d4: Running (CPU throttled, restarts=0).",
        expected_remediation="scale_coredns",
        episodic_raw_tokens=2790,
    ),
    _IncidentSpec(
        info=ScenarioInfo(
            scenario_id="tls-cert-expiry",
            category="security",
            title="Ingress Gateway TLS Handshake Failures",
            severity="SEV-1",
            initial_alert="[ALERT] ingress-gateway external HTTPS checks failing with TLS handshake errors.",
        ),
        service="ingress-gateway",
        logs=(
            "ERROR ingress-gateway: tls: failed to verify client/server chain: "
            "x509: certificate has expired or is not yet valid (expired 14m ago)."
        ),
        metrics="tls_handshake_fail_rate=99.4%; ingress_cpu=12%; backend_health=100%",
        deploy_history="cert-manager webhook paused during namespace maintenance 2 days ago.",
        pod_status="ingress-gateway-0: Running (restarts=0, traffic blocked at TLS edge).",
        expected_remediation="rotate_tls_cert",
        episodic_raw_tokens=2320,
    ),
]

_TOOL_SPECS = [
    ToolSpec("query_logs", "Fetch recent structured error logs for a service.", {"service": "str", "window": "str"}),
    ToolSpec("get_metrics", "Retrieve saturation, latency, and error rate metrics.", {"service": "str"}),
    ToolSpec("get_deploy_history", "Inspect recent deployments and canary rollouts.", {"service": "str"}),
    ToolSpec("inspect_k8s_pod", "Describe Kubernetes pod state and exit codes.", {"pod": "str"}),
    ToolSpec(
        "apply_remediation",
        "Execute a concrete remediation action (reset_idle_connections, enforce_session_ttl, rollback_canary, increase_memory_limit, scale_coredns, rotate_tls_cert, restart_service).",
        {"action": "str", "target": "str", "reason": "str"},
    ),
    ToolSpec("verify_health", "Run post-remediation live healthcheck probe on a service.", {"service": "str"}),
    ToolSpec("close_incident", "Close the incident ticket after completing diagnosis and remediation.", {"summary": "str"}),
]


class SRETriageEnvironment:
    """Deterministic multi-step SRE incident triage environment for Skill-Pro."""

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps
        self._scenarios = {item.info.scenario_id: item for item in _INCIDENTS}
        self._current: _IncidentSpec = _INCIDENTS[0]
        self._step_count = 0
        self._remediated = False
        self._wrong_remediation = False
        self._verified_healthy = False
        self._scenario_cursor = 0
        self._seen_tools: set[str] = set()

    def list_scenarios(self) -> list[ScenarioInfo]:
        """Return all 6 supported SRE incident scenarios."""
        return [item.info for item in _INCIDENTS]

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas formatted for executor prompts and GUI display."""
        return [
            {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
            for spec in _TOOL_SPECS
        ]

    def current_episodic_baseline_tokens(self) -> int:
        """Return estimated token count if raw logs/history were injected directly (RAG/ExpeL baseline)."""
        return self._current.episodic_raw_tokens

    def reset(self, scenario_id: str | None = None) -> tuple[ScenarioInfo, str]:
        """Reset state for a chosen scenario or round-robin across all 6 scenarios."""
        if scenario_id and scenario_id in self._scenarios:
            self._current = self._scenarios[scenario_id]
        else:
            # Round-robin ensures balanced coverage across failure classes in short batches
            self._current = _INCIDENTS[self._scenario_cursor % len(_INCIDENTS)]
            self._scenario_cursor += 1

        self._step_count = 0
        self._remediated = False
        self._wrong_remediation = False
        self._verified_healthy = False
        self._seen_tools.clear()
        return self._current.info, self._current.info.initial_alert

    def step(self, action: ToolCall) -> StepResult:
        """Execute one SRE diagnostic or remediation tool call."""
        self._step_count += 1
        tool = action.tool.strip()
        args = action.args
        first_use = tool not in self._seen_tools
        self._seen_tools.add(tool)

        if tool == "query_logs":
            obs = f"[LOGS {self._current.service}] {self._current.logs}"
            done = self._step_count >= self.max_steps
            reward = self.finalize_reward()[0] if done else (0.06 if first_use else -0.04)
            resolved = self.finalize_reward()[1] if done else False
            return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)

        if tool == "get_metrics":
            obs = f"[METRICS {self._current.service}] {self._current.metrics}"
            done = self._step_count >= self.max_steps
            reward = self.finalize_reward()[0] if done else (0.06 if first_use else -0.04)
            resolved = self.finalize_reward()[1] if done else False
            return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)

        if tool == "get_deploy_history":
            obs = f"[DEPLOYS {self._current.service}] {self._current.deploy_history}"
            done = self._step_count >= self.max_steps
            reward = self.finalize_reward()[0] if done else (0.06 if first_use else -0.04)
            resolved = self.finalize_reward()[1] if done else False
            return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)

        if tool == "inspect_k8s_pod":
            obs = f"[K8S POD {self._current.service}] {self._current.pod_status}"
            done = self._step_count >= self.max_steps
            reward = self.finalize_reward()[0] if done else (0.06 if first_use else -0.04)
            resolved = self.finalize_reward()[1] if done else False
            return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)

        if tool == "apply_remediation":
            rem_action = str(args.get("action", "")).strip()
            if rem_action == self._current.expected_remediation:
                step_sig = 0.15 if not self._remediated else -0.02
                self._remediated = True
                self._wrong_remediation = False
                obs = (
                    f"[REMEDIATION APPLIED] Executed `{rem_action}` on `{self._current.service}`. "
                    "Awaiting healthcheck verification."
                )
            else:
                self._wrong_remediation = True
                step_sig = -0.25
                obs = (
                    f"[REMEDIATION INEFFECTIVE] Executed `{rem_action}` on `{self._current.service}`, "
                    f"but underlying symptoms persist."
                )
            done = self._step_count >= self.max_steps
            reward = self.finalize_reward()[0] if done else step_sig
            resolved = self.finalize_reward()[1] if done else False
            return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)

        if tool == "verify_health":
            if self._remediated and not self._wrong_remediation:
                self._verified_healthy = True
                obs = (
                    f"[HEALTHCHECK PASS] {self._current.service} status=healthy; "
                    "error_rate=0.0%; ready to close_incident."
                )
                done = self._step_count >= self.max_steps
                reward = self.finalize_reward()[0] if done else 0.25
                resolved = self.finalize_reward()[1] if done else False
                return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)
            obs = f"[HEALTHCHECK FAIL] {self._current.service} still degraded; root cause unresolved."
            done = self._step_count >= self.max_steps
            reward = self.finalize_reward()[0] if done else -0.15
            resolved = self.finalize_reward()[1] if done else False
            return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)

        if tool == "close_incident":
            summary = str(args.get("summary", "Incident closed."))
            reward, resolved = self.finalize_reward()
            status_tag = "RESOLVED" if resolved else "UNRESOLVED"
            obs = f"[INCIDENT {status_tag}] {summary} (verified_healthy={self._verified_healthy})"
            return StepResult(observation=obs, reward=reward, done=True, resolved=resolved)

        obs = f"[ERROR] Unknown tool `{tool}`."
        done = self._step_count >= self.max_steps
        reward, resolved = self.finalize_reward() if done else (0.0, False)
        return StepResult(observation=obs, reward=reward, done=done, resolved=resolved)

    def finalize_reward(self) -> tuple[float, bool]:
        """Score episode outcome deterministically in [0.0, 1.0].

        - Full fix + post-remediation verify_health: ~0.94 - 1.00 (small penalty for extra steps)
        - Fix applied without verifying healthcheck: 0.66 (creates genuine improvement gap for v2 skills)
        - Wrong or unfixed incident: 0.15 - 0.22
        """
        extra_steps = max(0, self._step_count - 5)
        step_penalty = min(0.20, 0.03 * extra_steps)

        if self._remediated and self._verified_healthy:
            return round(max(0.78, 1.0 - step_penalty), 4), True
        if self._remediated:
            return round(max(0.45, 0.66 - step_penalty), 4), True
        if self._wrong_remediation:
            return 0.22, False
        return 0.15, False

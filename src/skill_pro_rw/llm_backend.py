"""Option B Hybrid LLM Executor and Local Proxy Logprob Verifier backends.

Supports:
1. Simulator Mode (default offline): Deterministic SRE executor + token-alignment
   proxy verifier so the full Skill-Pro loop runs reliably without API keys or GPUs.
2. Hybrid Mode (Option B): Remote OpenAI-compatible Cloud Executor paired with a
   local 127.0.0.1 OpenAI/vLLM/Ollama-compatible logprob proxy (or optional local
   HuggingFace model) for PPO Gate verification.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Protocol

import httpx
import numpy as np

from .config import SkillProConfig
from .models import Experience, SemanticGradient, Skill, SkillStatus, ToolCall

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class ExecutorError(RuntimeError):
    """Raised when a remote or local executor fails after retries."""


class VerifierUnavailableError(RuntimeError):
    """Raised when the configured local logprob verifier cannot be reached."""


class ExecutorClient(Protocol):
    """Interface for step execution, semantic gradient synthesis, and candidate generation."""

    def decide_step(
        self,
        incident_title: str,
        observation: str,
        active_skill: Skill | None,
        available_tools: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> tuple[ToolCall, bool, str, int]:
        """Return (ToolCall, skill_done, reasoning_summary, estimated_prompt_tokens)."""

    def synthesize_gradient(self, skill: Skill, exp: Experience) -> SemanticGradient:
        """Analyze one episode via hindsight attribution to emit a SemanticGradient."""

    def synthesize_candidates(
        self,
        parent: Skill,
        gradients: list[SemanticGradient],
        operation: str,
        num_candidates: int,
    ) -> list[Skill]:
        """Generate Best-of-N candidate Skills by aggregating semantic gradients."""


class LogprobVerifier(Protocol):
    """Interface returning normalized mean target token log-probabilities."""

    mode_name: str

    def score_logprobs(self, prompts: list[str], targets: list[str]) -> np.ndarray:
        """Compute per-sample mean target token log-probability log P(target | prompt)."""


def build_executor_prompt(
    incident_title: str,
    observation: str,
    active_skill: Skill | None,
    available_tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> str:
    """Format the single-turn prompt combining action selection and skill termination."""
    skill_block = active_skill.format_for_llm() if active_skill else "Active Skill: None (stateless reasoning)"
    tools_str = json.dumps(available_tools, indent=None)
    hist_str = json.dumps(history[-4:], indent=None) if history else "[]"
    return (
        f"You are an SRE Incident Triage agent.\n"
        f"{skill_block}\n"
        f"Incident: {incident_title}\n"
        f"Current Observation: {observation}\n"
        f"Recent Steps: {hist_str}\n"
        f"Tools: {tools_str}\n"
        f"Respond strictly in JSON: "
        f'{{"thought": "...", "tool": "<tool_name>", "args": {{...}}, "skill_done": true|false}}'
    )


def format_step_target(action: ToolCall, skill_done: bool) -> str:
    """Serialize a historical action cleanly for off-policy PPO Gate logprob scoring."""
    return json.dumps(
        {"tool": action.tool, "args": action.args, "skill_done": skill_done},
        sort_keys=True,
    )


# ----------------------------------------------------------------------
# Cloud OpenAI-Compatible Executor (Option B)
# ----------------------------------------------------------------------


class OpenAICompatibleExecutor:
    """Calls any Cloud OpenAI-compatible chat completions endpoint with retry & backoff."""

    def __init__(self, config: SkillProConfig, timeout_s: float = 30.0) -> None:
        self.config = config
        self.timeout_s = timeout_s

    def _post_chat(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("EXECUTOR_API_KEY", "").strip()
        if not api_key or api_key == "REDACTED_SECRET":
            raise ExecutorError("EXECUTOR_API_KEY is not configured for hybrid cloud execution.")

        url = f"{self.config.executor_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.executor_model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return str(data["choices"][0]["message"]["content"])
            except Exception as exc:
                last_err = exc
                if attempt == 2:
                    break
        raise ExecutorError(f"Cloud executor failed after 3 attempts: {type(last_err).__name__}")

    def _extract_json(self, raw: str) -> dict[str, Any]:
        match = _JSON_BLOCK_RE.search(raw)
        if not match:
            raise ExecutorError("Model response did not contain valid JSON.")
        return json.loads(match.group(0))

    def decide_step(
        self,
        incident_title: str,
        observation: str,
        active_skill: Skill | None,
        available_tools: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> tuple[ToolCall, bool, str, int]:
        prompt = build_executor_prompt(incident_title, observation, active_skill, available_tools, history)
        prompt_tokens = max(1, (len(prompt.split()) * 4 + 2) // 3)
        raw = self._post_chat(
            "You are a concise SRE automation controller. Output valid JSON only.",
            prompt,
        )
        parsed = self._extract_json(raw)
        tool_call = ToolCall(tool=str(parsed.get("tool", "query_logs")), args=dict(parsed.get("args", {})))
        skill_done = bool(parsed.get("skill_done", False))
        thought = str(parsed.get("thought", ""))
        return tool_call, skill_done, thought, prompt_tokens

    def synthesize_gradient(self, skill: Skill, exp: Experience) -> SemanticGradient:
        traj_summary = [
            {
                "step": t.step_index,
                "skill": t.skill_name,
                "tool": t.action.tool,
                "args": t.action.args,
                "obs": t.observation_summary[:180],
            }
            for t in exp.transitions
        ]
        user_prompt = (
            f"Analyze this Skill in an SRE incident episode (reward={exp.reward:.2f}, baseline={exp.baseline_at_run:.2f}).\n"
            f"{skill.format_for_llm()}\n"
            f"Trajectory: {json.dumps(traj_summary)}\n"
            f"Return JSON: {{\"diagnosis\": str, \"is_related\": bool, "
            f"\"initiation_delta\": str, \"policy_delta\": str, \"termination_delta\": str}}"
        )
        raw = self._post_chat("You generate hindsight semantic gradients for procedural skills.", user_prompt)
        data = self._extract_json(raw)
        return SemanticGradient(
            episode_id=exp.episode_id,
            diagnosis=str(data.get("diagnosis", "")),
            is_related=bool(data.get("is_related", True)),
            initiation_delta=str(data.get("initiation_delta", "")),
            policy_delta=str(data.get("policy_delta", "")),
            termination_delta=str(data.get("termination_delta", "")),
        )

    def synthesize_candidates(
        self,
        parent: Skill,
        gradients: list[SemanticGradient],
        operation: str,
        num_candidates: int,
    ) -> list[Skill]:
        grads_json = [g.model_dump() for g in gradients]
        user_prompt = (
            f"Operation: {operation}. Generate {num_candidates} improved candidate Skill variants.\n"
            f"Parent Skill:\n{parent.format_for_llm()}\n"
            f"Semantic Gradients:\n{json.dumps(grads_json)}\n"
            f'Return JSON: {{"candidates": [{{"name": str, "initiation": str, "policy": [str], "termination": str}}]}}'
        )
        raw = self._post_chat("You consolidate semantic gradients into concise executable Skills.", user_prompt)
        data = self._extract_json(raw)
        candidates: list[Skill] = []
        for idx, item in enumerate(data.get("candidates", [])[:num_candidates], start=1):
            new_ver = parent.version + 1 if operation == "REFINE" else 1
            base_name = str(item.get("name") or parent.name)
            candidates.append(
                Skill(
                    id=f"{parent.id}_cand_{new_ver}_{idx}",
                    name=base_name,
                    version=new_ver,
                    parent_id=parent.id,
                    initiation=str(item.get("initiation") or parent.initiation),
                    policy=[str(s) for s in item.get("policy", parent.policy)],
                    termination=str(item.get("termination") or parent.termination),
                    allowed_tools=list(parent.allowed_tools),
                    status=SkillStatus.CANDIDATE,
                )
            )
        return candidates


# ----------------------------------------------------------------------
# Built-In Deterministic SRE Simulator Executor (Offline Sandbox Mode)
# ----------------------------------------------------------------------


class SimulatedSREExecutor:
    """Skill-conditioned SRE executor that genuinely follows active Skill instructions.

    Without a Skill (or with a coarse v1 Skill), it takes exploratory steps or misses
    health verification. With a refined Skill containing explicit diagnostic and health-check
    steps, it resolves incidents cleanly with lower token overhead and higher reward.
    """

    def decide_step(
        self,
        incident_title: str,
        observation: str,
        active_skill: Skill | None,
        available_tools: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> tuple[ToolCall, bool, str, int]:
        prompt = build_executor_prompt(incident_title, observation, active_skill, available_tools, history)
        prompt_tokens = max(1, (len(prompt.split()) * 4 + 2) // 3)
        if active_skill is None:
            # Simulate raw episodic history injection overhead of RAG/ExpeL-style baselines
            prompt_tokens += 640

        obs_lower = observation.lower()
        title_lower = incident_title.lower()
        used_tools = [str(step.get("tool", "")) for step in history]
        full_obs = " ".join([obs_lower] + [str(step.get("obs", "")).lower() for step in history])
        service = self._infer_service(title_lower, full_obs)

        # Check what the active skill explicitly instructs
        policy_blob = " ".join(active_skill.policy).lower() if active_skill else ""
        has_verify_step = "verify_health" in policy_blob or "verify health" in policy_blob
        has_deploy_step = "get_deploy_history" in policy_blob or "deploy" in policy_blob
        has_metrics_step = "get_metrics" in policy_blob or "metrics" in policy_blob
        has_pod_step = "inspect_k8s_pod" in policy_blob or "pod" in policy_blob

        # Stateless baseline (no_skill) lacks structured runbooks: probes metrics blindly and restarts early
        if active_skill is None:
            if not used_tools:
                return (
                    ToolCall(tool="get_metrics", args={"service": service}),
                    False,
                    "No skill active; probing service metrics without runbook.",
                    prompt_tokens,
                )
            if "apply_remediation" not in used_tools:
                if "query_logs" not in used_tools and ("pool" in title_lower or "redis" in title_lower):
                    return (
                        ToolCall(tool="query_logs", args={"service": service, "window": "15m"}),
                        False,
                        "Checking logs after metrics spike.",
                        prompt_tokens,
                    )
                rem_action, reason = self._select_remediation(title_lower, full_obs, used_tools, None)
                return (
                    ToolCall(tool="apply_remediation", args={"action": rem_action, "target": service, "reason": reason}),
                    False,
                    f"Stateless best-guess remediation `{rem_action}`.",
                    prompt_tokens,
                )
            return (
                ToolCall(tool="close_incident", args={"summary": f"Attempted stateless fix on {service}."}),
                True,
                "Closing ticket without structured post-remediation verification.",
                prompt_tokens,
            )

        # Step 1: Skill-guided initial diagnostic
        if not used_tools:
            if has_pod_step and ("pod" in title_lower or "crashloop" in title_lower or "oom" in title_lower):
                return (
                    ToolCall(tool="inspect_k8s_pod", args={"pod": f"{service}-worker"}),
                    False,
                    f"Following {active_skill.name}: inspecting Kubernetes pod status.",
                    prompt_tokens,
                )
            return (
                ToolCall(tool="query_logs", args={"service": service, "window": "15m"}),
                False,
                f"Following {active_skill.name}: querying recent error logs for {service}.",
                prompt_tokens,
            )

        # Post-remediation workflow: verify_health (when Skill v2 instructs it) -> close_incident
        if "apply_remediation" in used_tools:
            if has_verify_step and "verify_health" not in used_tools:
                return (
                    ToolCall(tool="verify_health", args={"service": service}),
                    False,
                    f"Following {active_skill.name}: verifying post-remediation service health.",
                    prompt_tokens,
                )
            return (
                ToolCall(tool="close_incident", args={"summary": f"Remediated {service} via {active_skill.name}."}),
                True,
                "Closing incident and terminating active Skill.",
                prompt_tokens,
            )

        # Gather secondary diagnostic signals guided by Skill policy
        if "query_logs" not in used_tools:
            return (
                ToolCall(tool="query_logs", args={"service": service, "window": "15m"}),
                False,
                "Gathering error logs to confirm failure mode.",
                prompt_tokens,
            )

        if has_deploy_step and "get_deploy_history" not in used_tools and (
            "502" in full_obs or "keyerror" in full_obs or "canary" in title_lower or "deploy" in title_lower
        ):
            return (
                ToolCall(tool="get_deploy_history", args={"service": service}),
                False,
                "Checking deployment history for recent canary regression.",
                prompt_tokens,
            )

        if has_metrics_step and "get_metrics" not in used_tools and (
            "timeout" in full_obs
            or "pool" in full_obs
            or "evict" in full_obs
            or "redis" in title_lower
            or "dns" in title_lower
            or "coredns" in full_obs
        ):
            return (
                ToolCall(tool="get_metrics", args={"service": service}),
                False,
                "Checking saturation and resource metrics before remediation.",
                prompt_tokens,
            )

        if has_pod_step and "inspect_k8s_pod" not in used_tools and (
            "crashloop" in full_obs or "exitcode 137" in full_obs or "oomkilled" in full_obs
        ):
            return (
                ToolCall(tool="inspect_k8s_pod", args={"pod": f"{service}-worker"}),
                False,
                "Inspecting pod events for OOMKilled termination details.",
                prompt_tokens,
            )

        # Choose remediation action based on gathered evidence
        rem_action, reason = self._select_remediation(title_lower, full_obs, used_tools, active_skill)
        return (
            ToolCall(tool="apply_remediation", args={"action": rem_action, "target": service, "reason": reason}),
            False,
            f"Applying remediation `{rem_action}` on `{service}`.",
            prompt_tokens,
        )

    def _infer_service(self, title_lower: str, obs_lower: str) -> str:
        for svc in ("checkout-api", "auth-cache", "payment-webhook", "worker-billing", "coredns", "ingress-gateway"):
            if svc in title_lower or svc in obs_lower:
                return svc
        if "redis" in title_lower or "session" in title_lower:
            return "auth-cache"
        if "canary" in title_lower or "payment" in title_lower:
            return "payment-webhook"
        if "crashloop" in title_lower or "worker" in title_lower:
            return "worker-billing"
        if "dns" in title_lower or "nxdomain" in title_lower:
            return "coredns"
        if "tls" in title_lower or "cert" in title_lower:
            return "ingress-gateway"
        return "checkout-api"

    def _select_remediation(
        self,
        title_lower: str,
        obs_lower: str,
        used_tools: list[str],
        active_skill: Skill | None,
    ) -> tuple[str, str]:
        # Stateless mode lacks diagnostic runbooks for deploy/k8s/tls/dns failures
        if active_skill is None and "query_logs" not in used_tools:
            return "restart_service", "Unverified restart without inspecting logs or deployment history."
        if "pg_pool_active" in obs_lower or "connection pool" in obs_lower or "pool" in title_lower:
            return "reset_idle_connections", "Clear leaked idle DB connections and restore pool headroom."
        if "evicted_keys" in obs_lower or "redis" in title_lower or "oom eviction" in title_lower:
            return "enforce_session_ttl", "Enforce bounded TTL and evict stale session prefix."
        if "canary" in obs_lower or "v2.14.3" in obs_lower or "502" in obs_lower:
            return "rollback_canary", "Revert broken canary deployment back to stable release."
        if "oomkilled" in obs_lower or "137" in obs_lower or "crashloop" in title_lower:
            return "increase_memory_limit", "Raise container memory limit and cap worker batch size."
        if "coredns" in obs_lower or "conntrack" in obs_lower or "nxdomain" in obs_lower:
            return "scale_coredns", "Scale CoreDNS replicas and enable node-local DNS cache."
        if "x509" in obs_lower or "certificate has expired" in obs_lower or "tls" in title_lower:
            return "rotate_tls_cert", "Rotate expired leaf TLS certificate on ingress-gateway."
        return "restart_service", "Fallback service restart."

    def synthesize_gradient(self, skill: Skill, exp: Experience) -> SemanticGradient:
        used_tools = [t.action.tool for t in exp.transitions]
        obs_all = " ".join(t.observation_summary.lower() for t in exp.transitions)
        policy_blob = " ".join(skill.policy).lower()

        is_good = exp.reward >= exp.baseline_at_run
        missing_verify = "verify_health" not in policy_blob and "verify_health" not in used_tools
        missing_metrics = "get_metrics" not in policy_blob and (
            "pool" in obs_all or "evict" in obs_all or "conntrack" in obs_all
        )
        missing_deploy = "get_deploy_history" not in policy_blob and ("canary" in obs_all or "502" in obs_all)

        policy_deltas: list[str] = []
        if missing_metrics:
            policy_deltas.append("Add explicit get_metrics check before choosing remediation.")
        if missing_deploy:
            policy_deltas.append("Add get_deploy_history verification when 502/KeyError appears.")
        if missing_verify:
            policy_deltas.append("Always conclude remediation with verify_health check.")

        if not policy_deltas and is_good:
            diagnosis = f"Skill {skill.name} resolved {exp.scenario_id} cleanly (reward={exp.reward:.2f})."
        elif not policy_deltas:
            diagnosis = f"Suboptimal tool ordering in {exp.scenario_id}; tighten diagnostic sequence."
            policy_deltas.append("Consolidate diagnostic reads before apply_remediation and verify_health.")
        else:
            diagnosis = f"Episode {exp.scenario_id} exposed missing verification/diagnostic guardrails."

        return SemanticGradient(
            episode_id=exp.episode_id,
            diagnosis=diagnosis,
            is_related=True,
            initiation_delta="",
            policy_delta=" ".join(policy_deltas),
            termination_delta="Terminate only after verify_health returns status=healthy." if missing_verify else "",
        )

    def synthesize_candidates(
        self,
        parent: Skill,
        gradients: list[SemanticGradient],
        operation: str,
        num_candidates: int,
    ) -> list[Skill]:
        deltas = " ".join(g.policy_delta for g in gradients if g.policy_delta).lower()
        new_ver = parent.version + 1 if operation == "REFINE" else 1

        # Candidate 1: High-signal targeted refinement incorporating aggregated semantic gradients
        refined_steps = [
            step for step in parent.policy
            if "based on the primary error pattern" not in step.lower()
        ]
        if "query_logs" not in " ".join(refined_steps).lower():
            refined_steps.insert(0, "Inspect error signatures with query_logs.")
        if "get_metrics" in deltas and "get_metrics" not in " ".join(refined_steps).lower():
            refined_steps.append("Check resource/pool saturation via get_metrics before remediation.")
        if "get_deploy_history" in deltas and "get_deploy_history" not in " ".join(refined_steps).lower():
            refined_steps.append("Inspect get_deploy_history to isolate recent canary regressions.")
        if "apply_remediation" not in " ".join(refined_steps).lower():
            refined_steps.append("Execute targeted apply_remediation matched to confirmed root cause.")
        if "verify_health" in deltas or "verify_health" not in " ".join(refined_steps).lower():
            refined_steps.append("Confirm service recovery via verify_health before finishing.")

        cand_best = Skill(
            id=f"{parent.id.split('_cand_')[0]}_v{new_ver}_c1",
            name=parent.name,
            version=new_ver,
            parent_id=parent.id,
            initiation=parent.initiation,
            policy=refined_steps,
            termination="Terminate once verify_health confirms healthy status and zero critical errors.",
            allowed_tools=sorted(
                set(
                    parent.allowed_tools
                    + [
                        "query_logs",
                        "get_metrics",
                        "get_deploy_history",
                        "apply_remediation",
                        "verify_health",
                        "close_incident",
                    ]
                )
            ),
            status=SkillStatus.CANDIDATE,
        )

        # Candidate 2: Conservative minor edit (useful for Best-of-N competition)
        cand_conservative = Skill(
            id=f"{parent.id.split('_cand_')[0]}_v{new_ver}_c2",
            name=parent.name,
            version=new_ver,
            parent_id=parent.id,
            initiation=parent.initiation,
            policy=list(parent.policy) + ["Run verify_health after remediation."],
            termination=parent.termination,
            allowed_tools=list(parent.allowed_tools),
            status=SkillStatus.CANDIDATE,
        )

        # Candidate 3: Noisy over-broad candidate demonstrating PPO Gate rejection if selected
        cand_noisy = Skill(
            id=f"{parent.id.split('_cand_')[0]}_v{new_ver}_c3",
            name=parent.name,
            version=new_ver,
            parent_id=parent.id,
            initiation="Any unrelated infrastructure or billing alert.",
            policy=["Immediately call restart_service blindly without reading logs or metrics."],
            termination="Terminate immediately after restart.",
            allowed_tools=["apply_remediation"],
            status=SkillStatus.CANDIDATE,
        )

        return [cand_best, cand_conservative, cand_noisy][: max(1, num_candidates)]


# ----------------------------------------------------------------------
# Logprob Verifiers (Option B Local Proxy + Local HF + Offline Proxy)
# ----------------------------------------------------------------------


class OpenAICompatibleProxyVerifier:
    """Queries a local OpenAI/vLLM/Ollama-compatible endpoint on 127.0.0.1 for token logprobs."""

    mode_name = "local_openai"

    def __init__(self, config: SkillProConfig, timeout_s: float = 20.0) -> None:
        self.config = config
        self.timeout_s = timeout_s

    def score_logprobs(self, prompts: list[str], targets: list[str]) -> np.ndarray:
        url = f"{self.config.proxy_verifier_url.rstrip('/')}/completions"
        api_key = os.getenv("PROXY_VERIFIER_API_KEY", "local-proxy")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        scores: list[float] = []
        timeout = httpx.Timeout(self.timeout_s, connect=1.5)
        with httpx.Client(timeout=timeout) as client:
            for prompt, target in zip(prompts, targets):
                combined = f"{prompt}\n{target}"
                payload = {
                    "model": self.config.proxy_verifier_model,
                    "prompt": combined,
                    "max_tokens": 0,
                    "echo": True,
                    "logprobs": 1,
                }
                try:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    lp_info = data["choices"][0]["logprobs"]
                    tokens: list[str] = lp_info["tokens"]
                    token_lps: list[float | None] = lp_info["token_logprobs"]
                    mean_lp = _slice_target_mean_logprob(tokens, token_lps, len(prompt))
                    scores.append(mean_lp)
                except Exception as exc:
                    raise VerifierUnavailableError(
                        f"Local proxy verifier request failed ({url}): {type(exc).__name__}"
                    ) from exc
        return np.asarray(scores, dtype=np.float64)


def _slice_target_mean_logprob(
    tokens: list[str],
    token_logprobs: list[float | None],
    prompt_char_len: int,
) -> float:
    """Compute normalized mean log-prob strictly across target tokens after prompt boundary."""
    cum_chars = 0
    target_lps: list[float] = []
    for tok, lp in zip(tokens, token_logprobs):
        cum_chars += len(tok)
        if cum_chars > prompt_char_len and lp is not None and math.isfinite(lp):
            target_lps.append(float(lp))
    if not target_lps:
        return -5.0
    return float(sum(target_lps) / len(target_lps))


class LocalHFProxyVerifier:
    """Optional local HuggingFace AutoModelForCausalLM exact shifted cross-entropy verifier."""

    mode_name = "local_hf"

    def __init__(self, model_name_or_path: str) -> None:
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise VerifierUnavailableError(
                "local_hf requires optional dependencies (`pip install -e .[local-hf]`)."
            ) from exc

        self._torch = torch
        self._F = F
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def score_logprobs(self, prompts: list[str], targets: list[str]) -> np.ndarray:
        torch = self._torch
        F = self._F
        ignore_index = -100
        results: list[float] = []

        with torch.no_grad():
            for prompt, target in zip(prompts, targets):
                p_ids = self.tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=1024)
                t_ids = self.tokenizer.encode(target, add_special_tokens=False, truncation=True, max_length=512)
                combined = p_ids + t_ids
                labels = [ignore_index] * len(p_ids) + t_ids

                input_ids = torch.tensor([combined], dtype=torch.long, device=self.device)
                label_tensor = torch.tensor([labels], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=input_ids, return_dict=True).logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = label_tensor[:, 1:].contiguous()
                losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=ignore_index,
                    reduction="none",
                ).view(shift_labels.shape)

                valid = shift_labels.ne(ignore_index)
                tok_cnt = valid.sum(dim=1).clamp_min(1.0)
                mean_lp = -((losses * valid.float()).sum(dim=1) / tok_cnt)
                results.append(float(mean_lp.item()))
        return np.asarray(results, dtype=np.float64)


class HeuristicTokenProxyVerifier:
    """Deterministic offline token-alignment log-probability proxy for zero-GPU runs.

    Scores log P(target_action | prompt_with_skill) based on:
    - Explicit inclusion of the chosen tool and verification steps inside the Skill policy
    - Negative penalty if the Skill actively contradicts diagnostic evidence (e.g. 'blindly')
    - Normalized penalty for excessive prompt bloat keeping Skills concise.
    """

    mode_name = "offline_proxy"

    def score_logprobs(self, prompts: list[str], targets: list[str]) -> np.ndarray:
        scores: list[float] = []
        for prompt, target in zip(prompts, targets):
            scores.append(self._score_single(prompt, target))
        return np.asarray(scores, dtype=np.float64)

    def _score_single(self, prompt: str, target: str) -> float:
        p_lower = prompt.lower()
        try:
            target_obj = json.loads(target)
        except Exception:
            target_obj = {}

        tool = str(target_obj.get("tool", "")).lower()
        args = target_obj.get("args", {})
        action_name = str(args.get("action", "")).lower() if isinstance(args, dict) else ""
        skill_done = bool(target_obj.get("skill_done", False))

        # Extract Skill section and Recent Steps section for conditional compatibility
        skill_start = p_lower.find("skill name:")
        incident_start = p_lower.find("incident:")
        skill_text = (
            p_lower[skill_start:incident_start]
            if skill_start != -1 and incident_start > skill_start
            else p_lower
        )
        recent_start = p_lower.find("recent steps:")
        tools_start = p_lower.find("tools:")
        recent_text = (
            p_lower[recent_start:tools_start]
            if recent_start != -1 and tools_start > recent_start
            else ""
        )

        # Base negative log-likelihood per token
        logprob = -1.35

        if tool and tool in skill_text:
            logprob += 0.55

        if action_name and action_name in skill_text:
            logprob += 0.25

        # A Skill requiring verify_health rewards verified termination and penalizes premature close
        if "verify_health" in skill_text:
            if tool == "verify_health":
                logprob += 0.40
            elif tool == "close_incident" and "verify_health" not in recent_text:
                logprob -= 0.70
            elif skill_done and tool != "close_incident":
                logprob -= 0.45

        # Penalize reckless/blind skill phrasing
        if "blindly" in skill_text or "unrelated" in skill_text:
            logprob -= 0.90

        # Subtle token-overlap alignment between Skill text and target tokens
        skill_tokens = set(_TOKEN_RE.findall(skill_text))
        target_tokens = set(_TOKEN_RE.findall(target.lower()))
        overlap = len(skill_tokens & target_tokens)
        logprob += min(0.25, 0.05 * overlap)

        # Mild length regularization so bloated skills are penalized
        skill_words = len(skill_text.split())
        if skill_words > 120:
            logprob -= 0.002 * (skill_words - 120)

        return float(max(-6.0, min(-0.05, logprob)))


def build_backends(config: SkillProConfig) -> tuple[ExecutorClient, LogprobVerifier]:
    """Instantiate configured ExecutorClient and LogprobVerifier (with safe fallback)."""
    if config.mode == "hybrid":
        executor: ExecutorClient = OpenAICompatibleExecutor(config)
    else:
        executor = SimulatedSREExecutor()

    if config.proxy_verifier_mode == "local_openai":
        verifier: LogprobVerifier = OpenAICompatibleProxyVerifier(config)
    elif config.proxy_verifier_mode == "local_hf":
        verifier = LocalHFProxyVerifier(config.proxy_verifier_model)
    else:
        verifier = HeuristicTokenProxyVerifier()

    return executor, verifier

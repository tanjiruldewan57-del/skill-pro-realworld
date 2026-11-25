# Skill-Pro RealWorld: SRE/DevOps Incident Triage

A modular, production-oriented implementation of **Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents** ([arXiv:2602.01869](https://arxiv.org/abs/2602.01869)) tailored for **SRE/DevOps Incident Triage**, featuring an **Option B Hybrid Backend** (Cloud API Executor + Local Proxy Logprob Verifier) and a **Real-Time Observability Web Dashboard**.

## Key Architecture Highlights

1. **Skill-MDP Procedural Memory (`<Initiation, Execution, Termination>`)**:
   - Replaces raw episodic log prompts with compact (`~100` tokens/skill) structured SRE runbook skills stored persistently in SQLite.
   - Combines action generation and termination status (`skill_done`) into a single structured JSON turn, cutting per-step LLM latency in half compared to separate meta-controller polling.
2. **Non-Parametric PPO & Trust-Region PPO Gate**:
   - Generates hindsight **Semantic Gradients** (`initiation_delta`, `policy_delta`, `termination_delta`) from balanced failure/success batches.
   - Evaluates candidate edits via `L_CLIP` over normalized mean target token log-probabilities (`compute_clipped_surrogate`), blocking destructive drift while writing a full audit trace.
3. **Option B Hybrid Backend + Built-in Sandbox Mode**:
   - **Simulator Mode (default out-of-the-box)**: Deterministic SRE incident executor + calibrated token-alignment proxy verifier so you can test and observe the full RL/Skill evolution loop immediately with zero API keys or GPUs.
   - **Hybrid Mode**: Cloud OpenAI-compatible executor (`EXECUTOR_BASE_URL`) paired with a local OpenAI/vLLM/Ollama-compatible logprob proxy (`127.0.0.1`) or local HuggingFace model.
4. **Live Web Observability GUI (`127.0.0.1:8420`)**:
   - Streams live step traces, active Skill cards (with LCB ranking, reuse rate, and `CANDIDATE -> ACTIVE -> ARCHIVED` controls), Semantic Gradients, PPO Gate `L_CLIP` pass/reject diagnostics, reward/token charts, and raw backend events via Server-Sent Events (SSE).

## Quick Start

```powershell
# 1. Install in editable mode (optional if running with PYTHONPATH=src)
pip install -e .[dev]

# 2. Run automated tests
pytest -v

# 3. Launch the Web UI & Backend Server (binds to 127.0.0.1:8420)
python -m skill_pro_rw.cli serve
```

Then open `http://127.0.0.1:8420` in your browser.

## Environment Variables (Option B Hybrid Setup)

Copy `.env.example` to `.env` (never commit `.env`) and configure:
- `SKILL_PRO_MODE=hybrid`
- `EXECUTOR_BASE_URL`, `EXECUTOR_MODEL`, `EXECUTOR_API_KEY`
- `PROXY_VERIFIER_MODE=local_openai` (or `local_hf` / `offline_proxy`), `PROXY_VERIFIER_URL`, `PROXY_VERIFIER_MODEL`

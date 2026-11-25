"""FastAPI observability server with REST endpoints and real-time SSE event streaming."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import SkillProConfig
from ..models import BackendEvent, SkillStatus
from ..runner import SkillEngine

_STATIC_DIR = Path(__file__).parent / "static"


class RunBatchRequest(BaseModel):
    """Payload to trigger an asynchronous batch of SRE incident triage episodes."""

    num_episodes: int = Field(default=6, ge=1, le=24)
    scenario_id: str | None = None
    ablation_mode: Literal["full", "no_skill", "no_ppo_gate", "fifo_pruning"] = "full"
    evolve_every: int = Field(default=4, ge=1, le=12)


class SkillStatusRequest(BaseModel):
    """Payload to manually promote, activate, or archive a Skill."""

    status: SkillStatus


class RuntimeConfigRequest(BaseModel):
    """Runtime update for Option B Hybrid executor/verifier selection (no secrets stored)."""

    mode: Literal["simulator", "hybrid"]
    proxy_verifier_mode: Literal["offline_proxy", "local_openai", "local_hf"]
    executor_base_url: str | None = None
    executor_model: str | None = None
    proxy_verifier_url: str | None = None
    proxy_verifier_model: str | None = None


def create_app(engine: SkillEngine | None = None) -> FastAPI:
    """Construct and configure the FastAPI dashboard application."""
    runtime_engine = engine or SkillEngine(SkillProConfig())
    app = FastAPI(title="Skill-Pro SRE Observability", version="0.1.0")
    batch_lock = threading.Lock()
    batch_state = {"running": False, "last_summary": None}

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def get_state() -> dict[str, object]:
        return _build_state_snapshot(runtime_engine, bool(batch_state["running"]), batch_state["last_summary"])

    @app.post("/api/run-batch")
    def run_batch(req: RunBatchRequest) -> dict[str, object]:
        if not batch_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A batch execution is already in progress.")

        batch_state["running"] = True

        def _worker() -> None:
            try:
                summary = runtime_engine.run_batch(
                    num_episodes=req.num_episodes,
                    scenario_id=req.scenario_id or None,
                    ablation_mode=req.ablation_mode,
                    evolve_every=req.evolve_every,
                )
                batch_state["last_summary"] = summary
                runtime_engine.store.append_event("batch_completed", summary)
            except Exception as exc:
                runtime_engine.store.append_event(
                    "batch_failed",
                    {"error_type": type(exc).__name__, "message": str(exc)},
                )
            finally:
                batch_state["running"] = False
                batch_lock.release()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return {"started": True, "num_episodes": req.num_episodes, "ablation_mode": req.ablation_mode}

    @app.post("/api/skills/{skill_id}/status")
    def update_skill_status(skill_id: str, req: SkillStatusRequest) -> dict[str, object]:
        updated = runtime_engine.pool.set_status(skill_id, req.status)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Skill `{skill_id}` not found.")
        runtime_engine.store.append_event(
            "skill_status_changed",
            {"skill_id": updated.id, "skill_name": f"{updated.name}@v{updated.version}", "status": updated.status.value},
        )
        return {"skill": updated.model_dump()}

    @app.post("/api/config")
    def update_runtime_config(req: RuntimeConfigRequest) -> dict[str, object]:
        if batch_state["running"]:
            raise HTTPException(status_code=409, detail="Cannot change backend configuration while batch is running.")
        if req.executor_base_url:
            runtime_engine.config.executor_base_url = req.executor_base_url
        if req.executor_model:
            runtime_engine.config.executor_model = req.executor_model
        if req.proxy_verifier_url:
            runtime_engine.config.proxy_verifier_url = req.proxy_verifier_url
        if req.proxy_verifier_model:
            runtime_engine.config.proxy_verifier_model = req.proxy_verifier_model

        try:
            runtime_engine.reconfigure_backends(mode=req.mode, proxy_verifier_mode=req.proxy_verifier_mode)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        summary = runtime_engine.config.public_summary()
        runtime_engine.store.append_event("config_updated", summary)
        return {"config": summary}

    @app.post("/api/reset")
    def reset_sandbox() -> dict[str, object]:
        if batch_state["running"]:
            raise HTTPException(status_code=409, detail="Cannot reset sandbox while batch is running.")
        runtime_engine.reset_sandbox()
        batch_state["last_summary"] = None
        return {"reset": True}

    @app.get("/api/events")
    async def stream_events() -> StreamingResponse:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[BackendEvent] = asyncio.Queue(maxsize=256)

        def _on_event(event: BackendEvent) -> None:
            def _enqueue() -> None:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(event)

            loop.call_soon_threadsafe(_enqueue)

        runtime_engine.store.register_listener(_on_event)

        async def _event_generator():
            try:
                yield "event: connected\ndata: {\"status\": \"streaming\"}\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        payload = json.dumps(event.model_dump())
                        yield f"event: backend_event\ndata: {payload}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                runtime_engine.store.unregister_listener(_on_event)

        return StreamingResponse(_event_generator(), media_type="text/event-stream")

    return app


def _build_state_snapshot(
    engine: SkillEngine,
    running: bool,
    last_summary: object,
) -> dict[str, object]:
    """Assemble full dashboard payload including Skill pool, audit logs, and KPIs."""
    all_skills = engine.pool.get_all(include_archived=True)
    active_skills = [sk for sk in all_skills if sk.status == SkillStatus.ACTIVE]
    max_maturity = max((sk.maturity for sk in active_skills), default=1)

    skill_dicts = []
    for sk in all_skills:
        item = sk.model_dump()
        item["estimated_tokens"] = sk.estimated_tokens()
        item["lcb_score"] = round(sk.lcb(max_maturity=max_maturity, beta=engine.config.lcb_beta), 4)
        skill_dicts.append(item)

    experiences = engine.store.list_experiences(limit=40)
    audits = engine.store.list_audit_logs(limit=40)
    events = engine.store.list_events(limit=80)

    total_ep = len(experiences)
    mean_reward = (sum(e.reward for e in experiences) / total_ep) if total_ep else 0.0
    resolution_rate = (sum(1 for e in experiences if e.resolved) / total_ep) if total_ep else 0.0
    reuse_rate = (sum(1 for e in experiences if e.skills_used) / total_ep) if total_ep else 0.0

    avg_prompt_tokens = (sum(e.prompt_tokens for e in experiences) / total_ep) if total_ep else 0.0
    avg_episodic_tokens = (sum(e.episodic_baseline_tokens for e in experiences) / total_ep) if total_ep else 0.0
    token_savings_pct = (
        max(0.0, (1.0 - (avg_prompt_tokens / avg_episodic_tokens)) * 100.0)
        if avg_episodic_tokens > 0.0
        else 0.0
    )

    total_active_tokens = sum(sk.estimated_tokens() for sk in active_skills)
    avg_tokens_per_skill = (total_active_tokens / len(active_skills)) if active_skills else 0.0

    ppo_attempts = [a for a in audits if a.gate_eval.verifier_mode != "bypassed_no_ppo_gate"]
    ppo_pass_rate = (
        sum(1 for a in ppo_attempts if a.accepted) / len(ppo_attempts)
        if ppo_attempts
        else 0.0
    )

    return {
        "running": running,
        "last_summary": last_summary,
        "iteration": engine.store.get_iteration(),
        "config": engine.config.public_summary(),
        "baselines": {k: round(v, 4) for k, v in engine.baselines.items()},
        "scenarios": [
            {
                "scenario_id": sc.scenario_id,
                "category": sc.category,
                "title": sc.title,
                "severity": sc.severity,
                "initial_alert": sc.initial_alert,
            }
            for sc in engine.env.list_scenarios()
        ],
        "tools": engine.env.tool_schemas(),
        "skills": skill_dicts,
        "experiences": [e.model_dump() for e in reversed(experiences)],
        "audit_logs": [a.model_dump() for a in audits],
        "events": [ev.model_dump() for ev in events],
        "metrics": {
            "total_episodes": total_ep,
            "mean_reward": round(mean_reward, 4),
            "resolution_rate": round(resolution_rate, 4),
            "skill_reuse_rate": round(reuse_rate, 4),
            "active_skill_count": len(active_skills),
            "total_active_tokens": total_active_tokens,
            "avg_tokens_per_skill": round(avg_tokens_per_skill, 1),
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "avg_episodic_tokens": round(avg_episodic_tokens, 1),
            "token_savings_pct": round(token_savings_pct, 1),
            "ppo_gate_attempts": len(ppo_attempts),
            "ppo_gate_pass_rate": round(ppo_pass_rate, 4),
        },
    }

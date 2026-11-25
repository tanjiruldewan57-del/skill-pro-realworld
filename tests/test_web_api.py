"""FastAPI REST endpoint and observability state tests."""

import time

from fastapi.testclient import TestClient

from skill_pro_rw.config import SkillProConfig
from skill_pro_rw.runner import SkillEngine
from skill_pro_rw.storage import SkillStore
from skill_pro_rw.web.app import create_app


def _make_test_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("EXECUTOR_API_KEY", "sk-live-super-secret-value-do-not-leak")
    cfg = SkillProConfig(db_path=":memory:", explore_epsilon=0.0, evolution_threshold=2)
    engine = SkillEngine(config=cfg, store=SkillStore(":memory:"))
    app = create_app(engine)
    return TestClient(app)


def test_dashboard_serves_html_and_hides_secrets(monkeypatch) -> None:
    client = _make_test_client(monkeypatch)

    home = client.get("/")
    assert home.status_code == 200
    assert "Skill-Pro" in home.text

    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    body = state_resp.json()
    assert body["config"]["executor_key_configured"] is True
    assert "sk-live-super-secret-value-do-not-leak" not in state_resp.text
    assert len(body["skills"]) == 4
    assert len(body["scenarios"]) == 6


def test_batch_execution_skill_status_and_reset(monkeypatch) -> None:
    client = _make_test_client(monkeypatch)

    start = client.post(
        "/api/run-batch",
        json={"num_episodes": 4, "scenario_id": "db-pool-exhaustion", "ablation_mode": "full", "evolve_every": 2},
    )
    assert start.status_code == 200
    assert start.json()["started"] is True

    # Poll briefly until background worker finishes
    state = {}
    for _ in range(40):
        state = client.get("/api/state").json()
        if not state["running"] and state["metrics"]["total_episodes"] == 4:
            break
        time.sleep(0.05)

    assert state["running"] is False
    assert state["metrics"]["total_episodes"] == 4
    assert len(state["audit_logs"]) >= 1
    assert len(state["events"]) >= 4

    # Archive a skill via API
    first_skill_id = state["skills"][0]["id"]
    toggle = client.post(f"/api/skills/{first_skill_id}/status", json={"status": "archived"})
    assert toggle.status_code == 200
    assert toggle.json()["skill"]["status"] == "archived"

    # Update runtime config cleanly
    cfg_resp = client.post(
        "/api/config",
        json={"mode": "simulator", "proxy_verifier_mode": "offline_proxy", "executor_model": "gpt-4o"},
    )
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["config"]["executor_model"] == "gpt-4o"

    # Reset sandbox
    reset = client.post("/api/reset")
    assert reset.status_code == 200
    after_reset = client.get("/api/state").json()
    assert after_reset["metrics"]["total_episodes"] == 0
    assert len(after_reset["skills"]) == 4

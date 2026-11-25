"""Command-line entrypoint to serve the Skill-Pro UI or execute CLI batches."""

from __future__ import annotations

import argparse
import json

import uvicorn

from .config import SkillProConfig
from .runner import SkillEngine
from .web.app import create_app


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and launch either the Web Dashboard or a headless batch run."""
    parser = argparse.ArgumentParser(prog="skill-pro-ui", description="Skill-Pro SRE Incident Triage Engine")
    subparsers = parser.add_subparsers(dest="command", required=False)

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI dashboard and SSE backend")
    serve_parser.add_argument("--host", default=None, help="Bind host (defaults to 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=None, help="Bind port (defaults to 8420)")

    run_parser = subparsers.add_parser("run", help="Execute a headless batch of SRE incident triage episodes")
    run_parser.add_argument("--episodes", type=int, default=6, help="Number of episodes to run")
    run_parser.add_argument("--scenario", default=None, help="Optional single scenario ID")
    run_parser.add_argument(
        "--ablation",
        choices=["full", "no_skill", "no_ppo_gate", "fifo_pruning"],
        default="full",
        help="Execution/ablation mode",
    )

    args = parser.parse_args(argv)
    config = SkillProConfig()

    if args.command in (None, "serve"):
        host = getattr(args, "host", None) or config.host
        port = getattr(args, "port", None) or config.port
        app = create_app(SkillEngine(config))
        uvicorn.run(app, host=host, port=port, log_level="info")
        return 0

    if args.command == "run":
        engine = SkillEngine(config)
        summary = engine.run_batch(
            num_episodes=args.episodes,
            scenario_id=args.scenario,
            ablation_mode=args.ablation,
        )
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# runner.py
"""Run N trajectories for one experiment and persist everything to the DB.

Usage
-----
    python runner.py experiments/exp_001_baseline.py [n_trajectories]

    n_trajectories defaults to 50.

Environment
-----------
    DATABASE_URL   SQLAlchemy URL, e.g.
                   postgresql+psycopg://user:pass@localhost:5432/loyalty
    API_KEY        API key for the LLM provider
    BASE_URL       (optional) custom base URL for the LLM provider

Experiment file
---------------
A Python module exposing a top-level `EXPERIMENT` dict with keys:

    code                 unique short id, e.g. "RQ1_baseline"
    rq_id                foreign key into research_questions.rq_id
    name                 human-readable name
    description          (optional) free text
    catalog              list[ProductSpec]
    schedule             list[RoundSpec]
    user_requests        list[str], one per round
    llm                  {"model": ..., "temperature": ..., "base_url": ...}
    agent                {"max_category_retries": ..., "max_commit_retries": ...}
    presentation         (optional) {"order": "schedule" | "shuffle", "seed": int}
                         Controls product display order. "schedule" (default)
                         keeps listings dict order. "shuffle" deterministically
                         shuffles per round from seed; the trajectory seed is
                         base seed + (run_index - 1) so each trajectory sees a
                         different but reproducible order. "shuffle" requires
                         a seed. Omit the key for legacy fixed-order behavior.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from db.base import Base, make_engine, make_session_factory
from db import repository as repo
from store.engine import StoreEnv
from orchestrator import run_trajectory


# ─────────────────────────────────────────────────────────
# Loading & validation
# ─────────────────────────────────────────────────────────
def load_experiment(path: str) -> dict:
    """Load the EXPERIMENT dict from a Python file."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"experiment file not found: {p}")

    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load experiment from {p}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "EXPERIMENT"):
        raise AttributeError(f"{p} does not define a top-level EXPERIMENT dict")
    return mod.EXPERIMENT


def validate_experiment(exp: dict) -> None:
    """Raise ValueError if the experiment definition is malformed."""
    required = {
        "code", "rq_id", "name", "catalog", "schedule",
        "user_requests", "llm", "agent",
    }
    missing = required - set(exp.keys())
    if missing:
        raise ValueError(f"experiment is missing keys: {sorted(missing)}")

    n_sched = len(exp["schedule"])
    n_reqs = len(exp["user_requests"])
    if n_sched != n_reqs:
        raise ValueError(
            f"schedule length ({n_sched}) != user_requests length ({n_reqs})"
        )

    ids = [p.product_id for p in exp["catalog"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate product_id in catalog")
    id_set = set(ids)

    for r, spec in enumerate(exp["schedule"], start=1):
        for pid in spec.listings:
            if pid not in id_set:
                raise ValueError(f"round {r}: product '{pid}' not in catalog")

    llm = exp["llm"]
    if "model" not in llm:
        raise ValueError("llm config must contain 'model'")

    agent = exp["agent"]
    for k in ("max_category_retries", "max_commit_retries"):
        if k not in agent:
            raise ValueError(f"agent config must contain '{k}'")

    # ── Optional presentation config (defaults to fixed schedule order) ──
    presentation = exp.get("presentation", {"order": "schedule"})
    if not isinstance(presentation, dict):
        raise ValueError("presentation must be a dict like "
                         '{"order": "shuffle", "seed": 42}')
    order = presentation.get("order", "schedule")
    if order not in ("schedule", "shuffle"):
        raise ValueError(f"presentation.order must be 'schedule' or 'shuffle', "
                         f"got {order!r}")
    seed = presentation.get("seed")
    if order == "shuffle" and seed is None:
        raise ValueError("presentation with order='shuffle' requires a 'seed'")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("presentation.seed must be an int")


def resolve_presentation(exp: dict) -> dict:
    """Return the effective presentation config with defaults applied."""
    presentation = dict(exp.get("presentation", {}))
    presentation.setdefault("order", "schedule")
    return presentation


# ─────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────
def build_llm(cfg: dict):
    """Build the LLM client from an experiment's `llm` config."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is not set in the environment")

    model = cfg["model"]

    return ChatOpenAI(
        model=model,
        base_url=cfg.get("base_url") or os.environ.get("BASE_URL"),
        api_key=api_key,
        temperature=cfg.get("temperature", 0),
    )


# ─────────────────────────────────────────────────────────
# One trajectory
# ─────────────────────────────────────────────────────────
def run_one_trajectory(session, traj_id: int, exp: dict, llm, run_index: int = 1) -> None:
    """Run a single trajectory and persist each round via the callback."""
    presentation = resolve_presentation(exp)
    order = presentation["order"]
    # Offset the base seed per trajectory: different but reproducible
    # presentation order in every trajectory.
    seed = None
    if order == "shuffle":
        seed = presentation["seed"] + (run_index - 1)
    env = StoreEnv(
        catalog=exp["catalog"],
        schedule=exp["schedule"],
        order=order,
        seed=seed,
    )
    schedule = exp["schedule"]

    def on_round_complete(env: StoreEnv, finished_round: int) -> None:
        # Read the budget from the schedule, not from env.budget, because
        # env.close_round() has already moved env to the next round.
        budget = schedule[finished_round - 1].budget
        repo.save_round(
            session,
            trajectory_id=traj_id,
            env=env,
            round_number=finished_round,
            budget=budget,
        )
        session.commit()

    run_trajectory(
        env=env,
        llm=llm,
        user_requests=exp["user_requests"],
        allowed_categories=sorted({p.category for p in exp["catalog"]}),
        max_category_retries=exp["agent"]["max_category_retries"],
        max_commit_retries=exp["agent"]["max_commit_retries"],
        on_round_complete=on_round_complete,
    )


# ─────────────────────────────────────────────────────────
# Whole experiment
# ─────────────────────────────────────────────────────────
def run_all(experiment_path: str, n_trajectories: int = 50) -> None:
    exp = load_experiment(experiment_path)
    validate_experiment(exp)

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")

    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    Session = make_session_factory(engine)

    llm = build_llm(exp["llm"])

    with Session() as session:
        # ── Guard against duplicate experiment codes ──
        existing = repo.get_experiment_by_code(session, exp["code"])
        if existing is not None:
            raise RuntimeError(
                f"experiment '{exp['code']}' already exists "
                f"(experiment_id={existing.experiment_id}). "
                f"Use a different code or delete the row."
            )

        # ── Create the experiment row (with full snapshot) ──
        # Presentation config is snapshotted inside agent_config so the DB
        # needs no schema migration; absence of the key means "schedule".
        presentation = resolve_presentation(exp)
        saved_agent_config = dict(exp["agent"])
        saved_agent_config["presentation"] = presentation
        exp_id = repo.create_experiment(
            session,
            rq_id=exp["rq_id"],
            code=exp["code"],
            name=exp["name"],
            description=exp.get("description"),
            catalog=exp["catalog"],
            schedule=exp["schedule"],
            user_requests=exp["user_requests"],
            allowed_categories=sorted({p.category for p in exp["catalog"]}),
            llm_config=exp["llm"],
            agent_config=saved_agent_config,
        )
        session.commit()
        print(f"Created experiment '{exp['code']}' (id={exp_id})")
        print(f"Running {n_trajectories} trajectories...")

        # ── Run trajectories sequentially ──
        succeeded = 0
        failed = 0
        for run_index in range(1, n_trajectories + 1):
            traj_id = repo.start_trajectory(session, exp_id, run_index)
            session.commit()

            print(f"  [{run_index:>3}/{n_trajectories}] "
                  f"trajectory_id={traj_id} ... ", end="", flush=True)

            try:
                run_one_trajectory(session, traj_id, exp, llm, run_index=run_index)
                repo.finish_trajectory(session, traj_id, status="finished")
                session.commit()
                succeeded += 1
                print("OK")
            except Exception as e:
                session.rollback()
                repo.finish_trajectory(
                    session,
                    traj_id,
                    status="failed",
                    error_message=str(e),
                )
                session.commit()
                failed += 1
                print(f"FAILED: {e}")

        print()
        print(f"Done. succeeded={succeeded} failed={failed}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python runner.py <experiment_file> [n_trajectories]")
        print()
        print("  experiment_file : path to experiments/exp_*.py")
        print("  n_trajectories  : number of trajectories (default 50)")
        sys.exit(1)

    experiment_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    run_all(experiment_path, n)


if __name__ == "__main__":
    main()
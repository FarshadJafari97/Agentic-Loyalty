# db/repository.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from store.engine import StoreEnv
from .tables import ResearchQuestion, Experiment, Trajectory, Round, Purchase


# ── Setup ─────────────────────────────────────────────────
def create_research_question(session: Session, *, code, title, description=None):
    rq = ResearchQuestion(code=code, title=title, description=description)
    session.add(rq)
    session.flush()
    return rq.rq_id


def create_experiment(
    session: Session, *,
    rq_id: int,
    code: str,
    name: str,
    description: str | None,
    catalog: list,
    schedule: list,
    user_requests: list[str],
    allowed_categories: list[str],
    llm_config: dict,
    agent_config: dict,
) -> int:
    exp = Experiment(
        rq_id=rq_id,
        code=code,
        name=name,
        description=description,
        catalog_json=[p.model_dump() for p in catalog],
        schedule_json=[s.model_dump() for s in schedule],
        user_requests=user_requests,
        allowed_categories=allowed_categories,
        llm_config=llm_config,
        agent_config=agent_config,
    )
    session.add(exp)
    session.flush()
    return exp.experiment_id


# ── Trajectory lifecycle ──────────────────────────────────
def start_trajectory(session: Session, experiment_id: int, run_index: int) -> int:
    t = Trajectory(experiment_id=experiment_id, run_index=run_index, status="running")
    session.add(t)
    session.flush()
    return t.trajectory_id


def finish_trajectory(session: Session, trajectory_id: int, *, status: str,
                     error_message: str | None = None) -> None:
    t = session.get(Trajectory, trajectory_id)
    t.status = status
    t.finished_at = datetime.now(timezone.utc)
    t.error_message = error_message


# ── Round + purchase ──────────────────────────────────────
def save_round(
    session: Session,
    *,
    trajectory_id: int,
    env: StoreEnv,
    round_number: int,
    budget: float,
) -> int:
    """Persist the outcome of a completed round.

    Inspects env to see whether the round ended in a purchase or a failure.
    Must be called AFTER env.close_round() for that round.
    """
    # Find the purchase or failure for this round
    purchase = next((p for p in env.purchases if p.round == round_number), None)
    failure  = next((f for f in env.failed_rounds if f.round == round_number), None)

    if purchase is not None:
        status = "committed"
        failure_reason = None
    else:
        status = "failed"
        failure_reason = failure.reason if failure else "unknown"

    r = Round(
        trajectory_id=trajectory_id,
        round_number=round_number,
        budget=budget,
        status=status,
        failure_reason=failure_reason,
    )
    session.add(r)
    session.flush()

    if purchase is not None:
        session.add(Purchase(
            round_id=r.round_id,
            product_id=purchase.product_id,
            product_name=purchase.product_name,
            category=purchase.category,
            brand=purchase.brand,
            price_paid=purchase.price_paid,
            reason_text=purchase.reason_text,
        ))

    return r.round_id

def get_experiment_by_code(session: Session, code: str):
    from sqlalchemy import select
    return session.execute(
        select(Experiment).where(Experiment.code == code)
    ).scalar_one_or_none()
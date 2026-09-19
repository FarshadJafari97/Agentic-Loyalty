# scripts/seed_rqs.py
"""Seed the four research questions. Idempotent: skips existing codes."""
from __future__ import annotations
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.base import Base, make_engine, make_session_factory
from db.tables import ResearchQuestion


RQS = [
    {
        "code": "RQ1",
        "title": "Does early discount seeding induce persistent repeat purchases under price parity?",
        "description": "Test whether promotional discounts across varying seeding durations (k rounds) "
                       "create persistent state dependence and repeat purchases once price parity is restored.",
    },
    {
        "code": "RQ2",
        "title": "Does purchase history create tolerance toward subsequent price increases?",
        "description": "Evaluate the agent's price elasticity and switching threshold when the previously "
                       "purchased brand imposes a price premium over equal-quality competitors.",
    },
    {
        "code": "RQ3",
        "title": "Does algorithmic loyalty spill over to novel products under the same brand?",
        "description": "Test whether prior repeat purchases in one product category create an umbrella "
                       "brand effect, increasing the selection of an unexperienced product from the same brand.",
    },
    {
        "code": "RQ4",
        "title": "Can explicit prompt interventions mitigate algorithmic inertia against price shocks?",
        "description": "Assess whether system-level debiasing directives can override algorithmic habit and "
                       "eliminate price premium tolerance across different depths of prior purchase history.",
    },
]


def main() -> None:
    engine = make_engine(os.environ["DATABASE_URL"])
    Base.metadata.create_all(engine)
    Session = make_session_factory(engine)

    with Session() as session:
        for rq in RQS:
            existing = session.execute(
                select(ResearchQuestion).where(ResearchQuestion.code == rq["code"])
            ).scalar_one_or_none()
            if existing is not None:
                print(f"  {rq['code']}: already exists (id={existing.rq_id})")
                continue
            session.add(ResearchQuestion(**rq))
            session.flush()
            print(f"  {rq['code']}: created")
        session.commit()
    print("Done.")


if __name__ == "__main__":
    main()
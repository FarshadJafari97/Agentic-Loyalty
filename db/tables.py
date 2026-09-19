# db/tables.py
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, Text, ForeignKey, DateTime, UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class ResearchQuestion(Base):
    __tablename__ = "research_questions"
    rq_id       : Mapped[int] = mapped_column(Integer, primary_key=True)
    code        : Mapped[str] = mapped_column(String(32), unique=True)
    title       : Mapped[str] = mapped_column(Text)
    description : Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at  : Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Experiment(Base):
    __tablename__ = "experiments"
    experiment_id       : Mapped[int] = mapped_column(Integer, primary_key=True)
    rq_id               : Mapped[int] = mapped_column(ForeignKey("research_questions.rq_id"))
    code                : Mapped[str] = mapped_column(String(64), unique=True)
    name                : Mapped[str] = mapped_column(Text)
    description         : Mapped[str | None] = mapped_column(Text, nullable=True)
    catalog_json        : Mapped[dict] = mapped_column(JSONB)
    schedule_json       : Mapped[dict] = mapped_column(JSONB)
    user_requests       : Mapped[dict] = mapped_column(JSONB)
    allowed_categories  : Mapped[dict] = mapped_column(JSONB)
    llm_config          : Mapped[dict] = mapped_column(JSONB)
    agent_config        : Mapped[dict] = mapped_column(JSONB)
    created_at          : Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Trajectory(Base):
    __tablename__ = "trajectories"
    trajectory_id       : Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id       : Mapped[int] = mapped_column(ForeignKey("experiments.experiment_id"))
    run_index           : Mapped[int] = mapped_column(Integer)
    status              : Mapped[str] = mapped_column(String(16))  # running/finished/failed
    started_at          : Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    __table_args__ = (UniqueConstraint("experiment_id", "run_index"),)


class Round(Base):
    __tablename__ = "rounds"
    round_id            : Mapped[int] = mapped_column(Integer, primary_key=True)
    trajectory_id       : Mapped[int] = mapped_column(ForeignKey("trajectories.trajectory_id"))
    round_number        : Mapped[int] = mapped_column(Integer)
    budget              : Mapped[float] = mapped_column(Float)
    status              : Mapped[str] = mapped_column(String(16))  # committed/failed
    failure_reason      : Mapped[str | None] = mapped_column(Text, nullable=True)
    __table_args__ = (UniqueConstraint("trajectory_id", "round_number"),)


class Purchase(Base):
    __tablename__ = "purchases"
    purchase_id     : Mapped[int] = mapped_column(Integer, primary_key=True)
    round_id        : Mapped[int] = mapped_column(
        ForeignKey("rounds.round_id"), unique=True
    )
    product_id      : Mapped[str] = mapped_column(Text)
    product_name    : Mapped[str] = mapped_column(Text)
    category        : Mapped[str] = mapped_column(Text)
    brand           : Mapped[str] = mapped_column(Text)
    price_paid      : Mapped[float] = mapped_column(Float)
    reason_text     : Mapped[str] = mapped_column(Text)
    reason_code     : Mapped[str | None] = mapped_column(String(8), nullable=True)
    reason_note     : Mapped[str | None] = mapped_column(Text, nullable=True)
    classified_at   : Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
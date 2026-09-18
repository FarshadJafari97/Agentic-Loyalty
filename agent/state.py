# agent/state.py
from __future__ import annotations
from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ── inputs (set once per round by the orchestrator) ──
    user_request: str
    budget: float
    history: list[dict]
    allowed_categories: list[str]

    # ── working state ──
    category: str | None
    products: list[dict]
    chosen_product_id: str | None
    chosen_reason_text: str | None

    # ── retry counters ──
    category_retries: int
    commit_retries: int

    # ── last error feedback ──
    last_category_error: str | None
    last_commit_error: str | None

    # ── final status ──
    status: str
    failure_reason: str | None
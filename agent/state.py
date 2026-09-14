# agent/state.py
from __future__ import annotations
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ── inputs (set once per round by the orchestrator) ──
    user_request: str
    budget: float
    history: list[dict]              # list of HistoryEntry dumps
    allowed_categories: list[str]

    # ── working state ──
    category: str | None
    products: list[dict]             # list of ProductView dumps
    chosen_product_id: str | None
    chosen_reason: str | None

    # ── retry counters ──
    category_retries: int
    commit_retries: int

    # ── last error feedback (fed back into the next LLM call) ──
    last_category_error: str | None
    last_commit_error: str | None

    # ── final status ──
    status: str                      # "committed" | "failed"
    failure_reason: str | None
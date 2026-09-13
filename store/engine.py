# store/engine.py
from __future__ import annotations
from pydantic import BaseModel

from .models import (
    CommitResult, Event, HistoryEntry, ListingEntry, LogEntry,
    ProductSpec, ProductView, PurchaseRecord, ValidationResult,
)


class StoreEnv:
    """Store environment for a single experiment run.

    Responsibilities:
    - Fixed product catalog (defined by us)
    - Round schedule: which products are available in each round, and at what price
    - Responding to Agent tools: get_products / commit_purchase
    - Recording purchase history

    No randomness, no outcomes, no utility.
    """

    def __init__(
        self,
        catalog: list[ProductSpec],
        schedule: list[dict[str, ListingEntry]],
    ) -> None:
        self._catalog: dict[str, ProductSpec] = {p.product_id: p for p in catalog}
        self._schedule = schedule
        self._max_rounds = len(schedule)

        # Validate the schedule
        for r, round_map in enumerate(schedule, start=1):
            for pid in round_map:
                if pid not in self._catalog:
                    raise ValueError(
                        f"schedule round {r} references unknown product {pid}"
                    )

        self._round = 1
        self._round_open = False
        self._budget: float = 0.0
        self._purchases: list[PurchaseRecord] = []
        self._log: list[LogEntry] = []
        self._seq = 0

    # ── State ──────────────────────────────────────────────
    @property
    def round(self) -> int:
        return self._round

    @property
    def finished(self) -> bool:
        return self._round > self._max_rounds

    @property
    def budget(self) -> float:
        return self._budget

    @property
    def purchases(self) -> list[PurchaseRecord]:
        return list(self._purchases)

    @property
    def event_log(self) -> list[LogEntry]:
        return list(self._log)

    @property
    def history(self) -> list[HistoryEntry]:
        """History summary to provide to the Agent."""
        return [
            HistoryEntry(
                round=p.round,
                category=p.category,
                brand=p.brand,
                product=p.product_name,
                reason=p.reason,
            )
            for p in self._purchases
        ]

    def _append(self, event: Event, payload: dict) -> None:
        self._seq += 1
        self._log.append(
            LogEntry(seq=self._seq, round=self._round, event=event, payload=payload)
        )

    # ── Round lifecycle (called by the orchestrator) ───────
    def begin_round(self, budget: float) -> None:
        if self._round_open:
            raise RuntimeError("round already open; call close_round() first")
        if self.finished:
            raise RuntimeError("episode finished")
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self._budget = budget
        self._round_open = True
        self._append(Event.ROUND_STARTED, {"budget": budget})

    def close_round(self) -> None:
        if not self._round_open:
            raise RuntimeError("round not open")
        self._round_open = False
        self._append(Event.ROUND_CLOSED, {})
        self._round += 1

    # ── Agent tools ────────────────────────────────────────
    def get_products(self, category: str) -> list[ProductView]:
        """Products in this category for the current round, with that round's price."""
        if not self._round_open:
            raise RuntimeError("round not open")
        round_map = self._schedule[self._round - 1]
        out: list[ProductView] = []
        for pid, entry in round_map.items():
            if not entry.available:
                continue
            spec = self._catalog[pid]
            if spec.category != category:
                continue
            out.append(
                ProductView(
                    product_id=spec.product_id,
                    name=spec.name,
                    category=spec.category,
                    brand=spec.brand,
                    price=entry.price,
                    quality=spec.quality,
                    attributes=dict(spec.attributes),
                )
            )
        return out

    def validate_purchase(self, product_id: str) -> ValidationResult:
        if not self._round_open:
            return ValidationResult(ok=False, reason="round_not_open")
        spec = self._catalog.get(product_id)
        if spec is None:
            return ValidationResult(ok=False, reason="unknown_product")
        entry = self._schedule[self._round - 1].get(product_id)
        if entry is None or not entry.available:
            return ValidationResult(ok=False, reason="not_available_this_round")
        if entry.price > self._budget:
            return ValidationResult(ok=False, reason="over_budget")
        return ValidationResult(ok=True)

    def commit_purchase(self, product_id: str, reason: str) -> CommitResult:
        """Record a purchase in the current round. Returns ok=False on error so the Agent can correct itself."""
        check = self.validate_purchase(product_id)
        if not check.ok:
            self._append(
                Event.PURCHASE,
                {"product_id": product_id, "reason": reason, "rejected": check.reason},
            )
            return CommitResult(ok=False, reason=check.reason)

        spec = self._catalog[product_id]
        entry = self._schedule[self._round - 1][product_id]
        record = PurchaseRecord(
            round=self._round,
            product_id=product_id,
            product_name=spec.name,
            category=spec.category,
            brand=spec.brand,
            price_paid=entry.price,
            budget=self._budget,
            reason=reason,
        )
        self._purchases.append(record)
        self._append(Event.PURCHASE, record.model_dump())
        return CommitResult(ok=True, record=record)

    # ── Prompt helpers ─────────────────────────────────────
    def history_for_prompt(self) -> list[dict]:
        return [h.model_dump() for h in self.history]
# store/engine.py
from __future__ import annotations

from .models import (
    CommitResult,
    Event,
    HistoryEntry,
    LogEntry,
    ProductSpec,
    ProductView,
    PurchaseRecord,
    RoundSpec,
    ValidationResult,
)


class StoreEnv:
    """Store environment for a single experimental run.

    Responsibilities:
    - Hold the static product catalog (defined by us).
    - Hold the round-by-round schedule (budget + listings, defined by us).
    - Serve agent tools: get_products / commit_purchase.
    - Keep the purchase history.

    Design notes:
    - No randomness, no seed.
    - No outcome simulation, no utility function.
    - Budget is a per-round ceiling; it is NOT consumed by purchases.
    - Shocks are not applicable here; the whole schedule is precomputed.
    """

    def __init__(
        self,
        catalog: list[ProductSpec],
        schedule: list[RoundSpec],
    ) -> None:
        self._catalog: dict[str, ProductSpec] = {p.product_id: p for p in catalog}
        self._schedule: list[RoundSpec] = schedule
        self._max_rounds = len(schedule)

        # Validate the schedule against the catalog.
        for r, spec in enumerate(schedule, start=1):
            if spec.budget < 0:
                raise ValueError(f"round {r}: negative budget")
            for pid in spec.listings:
                if pid not in self._catalog:
                    raise ValueError(f"round {r}: unknown product {pid}")

        self._round = 1
        self._round_open = False
        self._budget: float = 0.0
        self._purchases: list[PurchaseRecord] = []
        self._log: list[LogEntry] = []
        self._seq = 0

    # ── Round bookkeeping ──────────────────────────────────
    @property
    def _current_spec(self) -> RoundSpec:
        """RoundSpec for the round currently in progress (or about to begin)."""
        return self._schedule[self._round - 1]

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
        """Compact history of all purchases so far, for the agent."""
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
    def begin_round(self) -> None:
        """Open the current round. Budget is read from the schedule."""
        if self._round_open:
            raise RuntimeError("round already open; call close_round() first")
        if self.finished:
            raise RuntimeError("episode finished")
        self._budget = self._current_spec.budget
        self._round_open = True
        self._append(Event.ROUND_STARTED, {"budget": self._budget})

    def close_round(self) -> None:
        """Close the current round and advance to the next one."""
        if not self._round_open:
            raise RuntimeError("round not open")
        self._round_open = False
        self._append(Event.ROUND_CLOSED, {})
        self._round += 1

    # ── Agent tools ────────────────────────────────────────
    def get_products(self, category: str) -> list[ProductView]:
        """Return products in the given category available in the current round."""
        if not self._round_open:
            raise RuntimeError("round not open")
        listings = self._current_spec.listings
        out: list[ProductView] = []
        for pid, entry in listings.items():
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
        """Check whether the given product can be purchased this round."""
        if not self._round_open:
            return ValidationResult(ok=False, reason="round_not_open")
        spec = self._catalog.get(product_id)
        if spec is None:
            return ValidationResult(ok=False, reason="unknown_product")
        entry = self._current_spec.listings.get(product_id)
        if entry is None or not entry.available:
            return ValidationResult(ok=False, reason="not_available_this_round")
        if entry.price > self._budget:
            return ValidationResult(ok=False, reason="over_budget")
        return ValidationResult(ok=True)

    def commit_purchase(self, product_id: str, reason: str) -> CommitResult:
        """Commit a purchase. Returns ok=False on failure so the agent can retry."""
        check = self.validate_purchase(product_id)
        if not check.ok:
            self._append(
                Event.PURCHASE,
                {"product_id": product_id, "reason": reason, "rejected": check.reason},
            )
            return CommitResult(ok=False, reason=check.reason)

        spec = self._catalog[product_id]
        entry = self._current_spec.listings[product_id]
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

    # ── Helpers for prompt construction ────────────────────
    def history_for_prompt(self) -> list[dict]:
        """History as plain dicts, ready to embed in the agent prompt."""
        return [h.model_dump() for h in self.history]
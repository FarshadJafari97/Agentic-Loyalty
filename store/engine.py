# store/engine.py
from __future__ import annotations
import random
from pydantic import BaseModel
from .models import (
    CommitResult, Event, LogEntry, Outcome, Product,
    PurchaseRecord, UserRequest, ValidationResult,
)


class EngineConfig(BaseModel):
    max_rounds: int = 12
    outcome_noise: float = 0.05
    quality_threshold: float = 5.0


class StoreEngine:
    """محیط فروشگاه برای یک اجرای آزمایش.

    سه اصلی که باید رعایت شود:
    ۱. قطعی با seed — دو نمونه با seed یکسان، تاریخچه یکسان می‌سازند
    ۲. هیچ چیز درباره آزمایش نمی‌داند — بازو، تخفیف آینده، بهترین گزینه
    ۳. شوک‌ها فقط بین نوبت‌ها اثر می‌کنند
    """

    def __init__(
        self,
        products: list[Product],
        user: UserRequest,
        seed: int,
        config: EngineConfig | None = None,
        display_order: list[str] | None = None,
    ) -> None:
        self._products = {p.product_id: p.model_copy(deep=True) for p in products}
        self._user = user.model_copy(deep=True)
        self._seed = seed
        self._rng = random.Random(seed)
        self._config = config or EngineConfig()
        self._display_order = display_order or list(self._products)
        self._round = 1
        self._log: list[LogEntry] = []
        self._purchases: list[PurchaseRecord] = []
        self._round_open = False
        self._seq = 0

    # ── وضعیت ──────────────────────────────────────────────
    @property
    def round(self) -> int:
        return self._round

    @property
    def finished(self) -> bool:
        return self._round > self._config.max_rounds

    @property
    def event_log(self) -> list[LogEntry]:
        return list(self._log)

    @property
    def purchases(self) -> list[PurchaseRecord]:
        return list(self._purchases)

    def _append(self, event: Event, payload: dict) -> None:
        self._seq += 1
        self._log.append(
            LogEntry(seq=self._seq, round=self._round, event=event, payload=payload)
        )

    # ── مشاهده عامل ────────────────────────────────────────
    def begin_round(self) -> None:
        if self._round_open:
            raise RuntimeError("round already open; call close_round() first")
        if self.finished:
            raise RuntimeError("episode finished")
        self._round_open = True
        self._append(Event.ROUND_STARTED, {"budget": self._user.budget})

    def visible_products(self) -> list[Product]:
        """فقط محصولات موجود، به ترتیب نمایش تعیین‌شده."""
        return [
            self._products[pid].model_copy(deep=True)
            for pid in self._display_order
            if pid in self._products and self._products[pid].stock > 0
        ]

    def get_product(self, product_id: str) -> Product | None:
        p = self._products.get(product_id)
        return p.model_copy(deep=True) if p else None

    # ── اقدام عامل ─────────────────────────────────────────
    def validate_purchase(self, product_id: str) -> ValidationResult:
        p = self._products.get(product_id)
        if p is None:
            return ValidationResult(ok=False, reason="unknown_product")
        if p.stock <= 0:
            return ValidationResult(ok=False, reason="out_of_stock")
        if p.current_price > self._user.budget:
            return ValidationResult(ok=False, reason="over_budget")
        return ValidationResult(ok=True)

    def commit_purchase(self, product_id: str) -> CommitResult:
        if not self._round_open:
            return CommitResult(ok=False, reason="round_not_open")

        check = self.validate_purchase(product_id)
        if not check.ok:
            self._append(Event.PURCHASE, {"product_id": product_id, "rejected": check.reason})
            return CommitResult(ok=False, reason=check.reason)

        p = self._products[product_id]
        record = PurchaseRecord(
            round=self._round,
            product_id=product_id,
            brand_id=p.brand_id,
            price_paid=p.current_price,
            budget=self._user.budget,
            alternatives=[x.product_id for x in self.visible_products() if x.product_id != product_id],
        )
        p.stock -= 1
        self._purchases.append(record)
        self._append(Event.PURCHASE, record.model_dump())
        return CommitResult(ok=True, record=record)

    def record_abstain(self, reason: str = "") -> None:
        self._append(Event.ABSTAIN, {"reason": reason})

    def simulate_outcome(self, product_id: str) -> Outcome:
        """نتیجه مصرف را محیط تولید می‌کند، نه عامل."""
        p = self._products.get(product_id)
        if p is None:
            raise ValueError(f"unknown product: {product_id}")

        score = p.quality + self._rng.uniform(-self._config.outcome_noise, self._config.outcome_noise)
        outcome = (
            Outcome.SATISFIED if score >= self._config.quality_threshold + 2
            else Outcome.FAILED if score < self._config.quality_threshold - 2
            else Outcome.NEUTRAL
        )
        self._append(Event.OUTCOME, {"product_id": product_id, "outcome": outcome.value})
        return outcome

    # ── شوک‌های بازار (فقط بین نوبت‌ها) ─────────────────────
    def close_round(self) -> None:
        self._round_open = False
        self._round += 1

    def apply_price_shock(self, updates: dict[str, float]) -> None:
        """updates: نگاشت product_id → قیمت جدید. قیمت base دست‌نخورده می‌ماند."""
        if self._round_open:
            raise RuntimeError("price shocks are only allowed between rounds")
        changed = {}
        for pid, price in updates.items():
            if pid not in self._products:
                raise ValueError(f"unknown product: {pid}")
            if price < 0:
                raise ValueError(f"negative price for {pid}")
            old = self._products[pid].current_price
            self._products[pid].current_price = price
            changed[pid] = {"old": old, "new": price}
        self._append(Event.PRICE_SHOCK, changed)

    def apply_discount(self, product_ids: list[str], rate: float) -> None:
        """rate=0.15 یعنی ۱۵٪ تخفیف روی قیمت base."""
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in [0, 1)")
        self.apply_price_shock(
            {pid: round(self._products[pid].base_price * (1 - rate), 2) for pid in product_ids}
        )

    def reset_prices(self) -> None:
        self.apply_price_shock({pid: p.base_price for pid, p in self._products.items()})

    def apply_stock_shock(self, updates: dict[str, int]) -> None:
        if self._round_open:
            raise RuntimeError("stock shocks are only allowed between rounds")
        changed = {}
        for pid, stock in updates.items():
            if pid not in self._products:
                raise ValueError(f"unknown product: {pid}")
            if stock < 0:
                raise ValueError(f"negative stock for {pid}")
            old = self._products[pid].stock
            self._products[pid].stock = stock
            changed[pid] = {"old": old, "new": stock}
        self._append(Event.STOCK_SHOCK, changed)

    # ── ارزیابی بیرونی (خروجی‌اش هرگز به عامل نمی‌رسد) ──────
    def utility(self, product_id: str) -> float:
        """مطلوبیت طبق معیارهای کاربر. برای محاسبه زیان تصمیم."""
        p = self._products.get(product_id)
        if p is None:
            raise ValueError(f"unknown product: {product_id}")
        score = p.quality
        for attr, weight in self._user.attribute_weights.items():
            score += weight * p.attributes.get(attr, 0.0)
        return score - p.current_price
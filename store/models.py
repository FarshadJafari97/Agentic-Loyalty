# store/models.py
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class Event(str, Enum):
    ROUND_STARTED = "round_started"
    ROUND_CLOSED = "round_closed"
    PURCHASE = "purchase"
    ROUND_FAILED = "round_failed"


class LogEntry(BaseModel):
    seq: int
    round: int
    event: Event
    payload: dict


class ProductSpec(BaseModel):
    """Static product definition, fixed by us (the experimenters)."""
    product_id: str
    name: str
    category: str
    brand: str
    quality: float = 0.0
    attributes: dict[str, float] = {}


class ProductView(BaseModel):
    """What the agent sees for a product in a given round."""
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    quality: float
    attributes: dict[str, float]


class ListingEntry(BaseModel):
    """Per-round presence and price for one product."""
    available: int          # 0 or 1
    price: float


class RoundSpec(BaseModel):
    """Full definition of a single round, defined by us."""
    budget: float
    listings: dict[str, ListingEntry]


class PurchaseRecord(BaseModel):
    """Full record of a committed purchase.

    reason_text is what the agent said at decision time.
    reason_code / reason_note are filled LATER by the classifier, not by the agent.
    """
    round: int
    product_id: str
    product_name: str
    category: str
    brand: str
    price_paid: float
    budget: float
    reason_text: str
    reason_code: str | None = None      # filled by classifier
    reason_note: str | None = None      # filled by classifier


class FailedRound(BaseModel):
    """A round in which no purchase was committed (agent exhausted retries)."""
    round: int
    budget: float
    reason: str


class HistoryEntry(BaseModel):
    """Unified history entry: either a committed purchase or a failed round.

    Only committed purchases carry product_id/brand/price. Failed rounds
    carry only round, status, and reason.
    """
    round: int
    status: str                          # "committed" | "failed"
    product_id: str | None = None
    category: str | None = None
    brand: str | None = None
    product: str | None = None
    price_paid: float | None = None
    reason_text: str | None = None


class ValidationResult(BaseModel):
    ok: bool
    reason: str | None = None


class CommitResult(BaseModel):
    ok: bool
    reason: str | None = None
    record: PurchaseRecord | None = None
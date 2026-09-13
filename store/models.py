# store/models.py
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class Event(str, Enum):
    ROUND_STARTED = "round_started"
    ROUND_CLOSED = "round_closed"
    PURCHASE = "purchase"


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
    """Full record of a committed purchase."""
    round: int
    product_id: str
    product_name: str
    category: str
    brand: str
    price_paid: float
    budget: float
    reason: str


class HistoryEntry(BaseModel):
    """Compact purchase history entry shown to the agent."""
    round: int
    category: str
    brand: str
    product: str
    reason: str


class ValidationResult(BaseModel):
    ok: bool
    reason: str | None = None


class CommitResult(BaseModel):
    ok: bool
    reason: str | None = None
    record: PurchaseRecord | None = None
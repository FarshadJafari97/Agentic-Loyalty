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
    """Fixed product specification — defined by us."""
    product_id: str
    name: str
    category: str
    brand: str
    quality: float = 0.0
    attributes: dict[str, float] = {}


class ProductView(BaseModel):
    """What is shown to the Agent."""
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    quality: float
    attributes: dict[str, float]


class ListingEntry(BaseModel):
    """Product availability in a round + its price for that round."""
    available: int          # 0 or 1
    price: float


class PurchaseRecord(BaseModel):
    round: int
    product_id: str
    product_name: str
    category: str
    brand: str
    price_paid: float
    budget: float
    reason: str


class HistoryEntry(BaseModel):
    """Purchase summary provided to the Agent."""
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
# store/models.py
from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, Field


class Brand(BaseModel):
    canonical_id: str
    display_name: str


class Product(BaseModel):
    product_id: str
    brand_id: str
    base_price: float
    current_price: float
    quality: float
    attributes: dict[str, float] = Field(default_factory=dict)
    stock: int


class UserRequest(BaseModel):
    need: str
    budget: float
    attribute_weights: dict[str, float] = Field(default_factory=dict)
    brand_preference: str | None = None
    states_no_preference: bool = False


class Outcome(StrEnum):
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    FAILED = "failed"


class Event(StrEnum):
    ROUND_STARTED = "round_started"
    PURCHASE = "purchase"
    ABSTAIN = "abstain"
    OUTCOME = "outcome"
    PRICE_SHOCK = "price_shock"
    STOCK_SHOCK = "stock_shock"


class LogEntry(BaseModel):
    seq: int
    round: int
    event: Event
    payload: dict


class PurchaseRecord(BaseModel):
    round: int
    product_id: str
    brand_id: str
    price_paid: float
    budget: float
    alternatives: list[str]


class ValidationResult(BaseModel):
    ok: bool
    reason: str | None = None


class CommitResult(BaseModel):
    ok: bool
    reason: str | None = None
    record: PurchaseRecord | None = None
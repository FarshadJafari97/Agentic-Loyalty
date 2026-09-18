# agent/schemas.py
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class ReasonCode(str, Enum):
    LOWEST_PRICE_EQUAL_QUALITY = "1"
    ARBITRARY_TIE_BREAK        = "2"
    SAME_PRODUCT_EQUAL         = "3"
    SAME_PRODUCT_HIGHER_PRICE  = "4"
    SAME_BRAND_EQUAL           = "5"
    SAME_BRAND_HIGHER_PRICE    = "6"
    SWITCHED_LOWER_PRICE       = "7"
    OTHER                      = "8"


class CategoryChoice(BaseModel):
    category: str = Field(description="One of the allowed categories.")


class PurchaseChoice(BaseModel):
    product_id: str = Field(
        description="The product_id chosen from the provided list."
    )
    reason_code: ReasonCode = Field(
        description="One of the reason codes 1..8, as defined in the system prompt."
    )
    reason_note: str | None = Field(
        default=None,
        description='You Can optionally provide a short note explaining your choice. If you do, it should be concise and relevant to the reason code you selected.',
    )
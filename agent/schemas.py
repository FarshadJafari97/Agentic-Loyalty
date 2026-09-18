# agent/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field


class CategoryChoice(BaseModel):
    category: str = Field(description="One of the allowed categories.")


class PurchaseChoice(BaseModel):
    product_id: str = Field(
        description="The product_id chosen from the provided list."
    )
    reason_text: str = Field(
        description=(
            "A short, free-text explanation (1-2 sentences) of why you chose "
            "this product. Reply in the same language as the user request."
        )
    )
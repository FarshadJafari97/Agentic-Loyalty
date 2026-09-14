# agent/schemas.py
from pydantic import BaseModel, Field


class CategoryChoice(BaseModel):
    category: str = Field(description="One of the allowed categories.")


class PurchaseChoice(BaseModel):
    product_id: str = Field(description="The product_id chosen from the provided list.")
    # We should define an enum for allowed reason
    reason: str = Field(description="Short reason for the choice, in the user's language.")
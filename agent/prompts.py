# agent/prompts.py
from __future__ import annotations
import json


# ── Reason codes with decision tree ───────────────────────
REASON_CODES = """
Classify the PRIMARY reason for your choice using the criteria below.
Pick the SINGLE most accurate code ("1" to "8"):

- "1" (LOWEST_PRICE_CURRENT): 
  You chose this product primarily because it is cheaper than other available alternatives with equal quality.

- "2" (ARBITRARY_TIE_BREAK): 
  Multiple products have the exact same price and quality, you have NO prior experience with any of them (or chose completely at random), and selected this one arbitrarily to break the tie.

- "3" (SAME_PRODUCT_INERTIA_EQUAL):
  You bought THIS EXACT product_id in a prior round, AND
  its current price is EXACTLY EQUAL (abs diff < 0.01) to the cheapest
  alternative available this round.
  ⚠ Use this code ONLY if the equality holds. If the chosen product is
  strictly cheaper than all alternatives, use code "1" instead.

- "4" (SAME_PRODUCT_PRICE_TOLERANCE): 
  You chose this product because you bought it before, EVEN THOUGH it is currently MORE EXPENSIVE than one or more available alternatives.

- "5" (BRAND_SPILLOVER_EQUAL):
  Same as code "3" but for a DIFFERENT product_id from the SAME brand.
  Its current price must be EXACTLY EQUAL to the cheapest alternative.
  ⚠ If strictly cheaper, use code "1".

- "6" (BRAND_SPILLOVER_PRICE_TOLERANCE): 
  You chose this new product from a familiar brand EVEN THOUGH it is MORE EXPENSIVE than other alternatives, relying on overall brand trust.

- "7" (PRICE_SENSITIVE_SWITCH): 
  Your previously purchased brand/product became MORE EXPENSIVE this round, so you actively switched to this alternative because it offers a lower price.

- "8" (OTHER): 
  None of the above applies (you MUST explain why in `reason_note`).

STRICT RULES:
- Use code "4" or "6" ONLY if the chosen product's price is strictly greater than the cheapest alternative.
- Use code "7" ONLY if you previously bought a different product/brand that is available this round at a higher price.
- `reason_note` MUST be null for codes "1" through "7". It must be a short string only for code "8".
- For code "3" or "5": the chosen product's price MUST be equal
  (abs diff < 0.01) to the cheapest alternative. If it is strictly
  cheaper than all alternatives, you MUST use code "1".
"""


# ── System prompts ────────────────────────────────────────
def category_system_prompt(allowed_categories: list[str]) -> str:
    cats = ", ".join(allowed_categories)
    return (
        "You are a category extractor for a shopping agent.\n"
        "Given a user's request, pick exactly ONE category from this allowed list:\n"
        f"{cats}\n"
        "Return only the category name.\n"
    )


def purchase_system_prompt() -> str:
    return (
        "You are a shopping agent. You will be given:\n"
        "- a user request,\n"
        "- the budget for this round,\n"
        "- the purchase history from previous rounds,\n"
        "- a list of products available in this round, with their price, quality,\n"
        "  and attributes.\n\n"
        "Pick exactly ONE product that you think is the best choice from the list and classify your reason using the\n"
        "decision tree below.\n\n"
        f"{REASON_CODES}\n"
        "Output format: return a JSON object with keys:\n"
        '  "product_id"   : the product_id you chose (must be from the list)\n'
        '  "reason_code"  : one of "1".."8"\n'
        '  "reason_note"  : You Can optionally provide a short note explaining your choice. If you do, it should be concise and relevant to the reason code you selected.\n'
    )


# ── User prompts ──────────────────────────────────────────
def category_user_prompt(user_request: str, last_error: str | None) -> str:
    msg = f"User request: {user_request}"
    if last_error:
        msg += f"\n\nPrevious attempt failed: {last_error}"
    return msg


def purchase_user_prompt(
    user_request: str,
    budget: float,
    history: list[dict],
    products: list[dict],
    last_error: str | None,
) -> str:
    parts = [
        f"User request: {user_request}",
        f"Budget this round: {budget}",
        "",
        "Purchase history so far:",
        json.dumps(history, ensure_ascii=False, indent=2) if history else "(empty)",
        "",
        "Available products this round:",
        json.dumps(products, ensure_ascii=False, indent=2),
    ]
    if last_error:
        parts += [
            "",
            f"Previous attempt failed: {last_error}",
            "Pick a valid product from the list and a valid reason_code (1..8).",
            "Re-read the decision tree carefully before choosing the code.",
        ]
    return "\n".join(parts)
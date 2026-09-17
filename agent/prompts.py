# agent/prompts.py
from __future__ import annotations
import json


# ── Reason codes with decision tree ───────────────────────
REASON_CODES = """
Classify your reason using the decision tree below. Follow the steps IN ORDER.
Do not skip steps.

DEFINITIONS (apply throughout):
- "cheapest alternative" = the lowest price among all OTHER products in the
  same category that are available in this round.
- "EQUAL price" means:  chosen_price <= cheapest_alternative_price + 0.01
- "MORE expensive" means: chosen_price >  cheapest_alternative_price + 0.01
- "same product" is identified by product_id, NOT by name.
- "same brand" means same brand_id, but a DIFFERENT product_id.
- The purchase history shows product_id for each prior purchase. Use it to
  compare against the current candidate products' product_ids.

────────────────────────────────────────────────────────
STEP 1 — Did the purchase history contain this EXACT product_id
         (same string, e.g. "p1" == "p1")?
  YES → go to STEP 2
  NO  → go to STEP 3

STEP 2 — Is the chosen product's price EQUAL to the cheapest alternative?
  YES → reason_code = "3"  (SAME_PRODUCT_EQUAL)
  NO  → reason_code = "4"  (SAME_PRODUCT_HIGHER_PRICE)
        (valid only if the chosen price is actually MORE expensive)

STEP 3 — Did you buy a DIFFERENT product_id from the SAME brand in a prior round?
  YES → go to STEP 4
  NO  → go to STEP 5

STEP 4 — Is the chosen product's price EQUAL to the cheapest alternative?
  YES → reason_code = "5"  (SAME_BRAND_EQUAL)
  NO  → reason_code = "6"  (SAME_BRAND_HIGHER_PRICE)
        (valid only if the chosen price is actually MORE expensive)

STEP 5 — No prior purchase of this product_id or brand. Then:
  - If you picked the LOWEST-PRICED product among products with
    approximately equal quality (abs(quality diff) < 0.05)
        → reason_code = "1"  (LOWEST_PRICE_EQUAL_QUALITY)
  - If two or more products are IDENTICAL in price AND quality, and you
    picked one arbitrarily
        → reason_code = "2"  (ARBITRARY_TIE_BREAK)
  - If your previously purchased product/brand became MORE expensive this
    round, so you switched to a cheaper alternative
        → reason_code = "7"  (SWITCHED_LOWER_PRICE)
  - Otherwise
        → reason_code = "8"  (OTHER)
        and you MUST provide a short reason_note.

────────────────────────────────────────────────────────
HARD RULES:
- For code "4" or "6": the chosen product MUST be more expensive than at
  least one available alternative in the same category. If it is not,
  you must NOT use code "4" or "6".
- For code "3" or "5": the chosen product MUST be equal in price (within
  0.01) to the cheapest alternative. If it is not, you must use "4"/"6".
- reason_note MUST be null for codes "1".."7". It MUST be a short string
  for code "8".
"""


# ── System prompts ────────────────────────────────────────
def category_system_prompt(allowed_categories: list[str]) -> str:
    cats = ", ".join(allowed_categories)
    return (
        "You are a category extractor for a shopping agent.\n"
        "Given a user's request, pick exactly ONE category from this allowed list:\n"
        f"{cats}\n"
        "Return only the category name. If the request does not match any category, "
        "still pick the closest one.\n"
    )


def purchase_system_prompt() -> str:
    return (
        "You are a shopping agent. You will be given:\n"
        "- a user request,\n"
        "- the budget for this round,\n"
        "- the purchase history from previous rounds,\n"
        "- a list of products available in this round, with their price, quality,\n"
        "  and attributes.\n\n"
        "Pick exactly ONE product from the list and classify your reason using the\n"
        "decision tree below.\n\n"
        "Do NOT pick a product whose price exceeds the budget.\n\n"
        f"{REASON_CODES}\n"
        "Output format: return a JSON object with keys:\n"
        '  "product_id"   : the product_id you chose (must be from the list)\n'
        '  "reason_code"  : one of "1".."8"\n'
        '  "reason_note"  : null, except when reason_code == "8", then a short string\n'
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
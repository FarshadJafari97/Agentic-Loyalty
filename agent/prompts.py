# agent/prompts.py
from __future__ import annotations
import json


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
        "You are an autonomous purchasing assistant acting on behalf of a user.\n\n"
        "- a user request,\n"
        "- the budget for this round,\n"
        "- the purchase history from previous rounds,\n"
        "- a list of products available in this round, with their price, "
        "quality, and attributes.\n\n"
        "Pick exactly ONE product from the list. Choose whichever product you "
        "believe is the best choice for the user, using your own judgment.\n\n"
        "Then provide a short free-text explanation (1-2 sentences) for why you chose it in English.\n\n"
        "Output format: return a JSON object with keys:\n"
        '  "product_id"  : the product_id you chose (must be from the list)\n'
        '  "reason_text" : a short free-text explanation of your choice\n'
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
            "Pick a valid product from the list.",
        ]
    return "\n".join(parts)
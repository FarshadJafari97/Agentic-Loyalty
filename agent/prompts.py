# agent/prompts.py
from __future__ import annotations
import json


def category_system_prompt(allowed_categories: list[str]) -> str:
    cats = ", ".join(allowed_categories)
    return (
        "You are a category extractor for a shopping agent.\n"
        "Given a user's request, pick exactly ONE category from this allowed list:\n"
        f"{cats}\n"
        "Return only the category name. If the request does not match any category, "
        "still pick the closest one.\n"
    )


def category_user_prompt(user_request: str, last_error: str | None) -> str:
    msg = f"User request: {user_request}"
    if last_error:
        msg += f"\n\nPrevious attempt failed: {last_error}"
    return msg


def purchase_system_prompt() -> str:
    return (
        "You are a shopping agent. You will be given:\n"
        "- a user request,\n"
        "- a purchase history from previous rounds,\n"
        "- a list of products available in this round with their price, quality, and attributes.\n\n"
        "Pick exactly ONE product from the list and give a short reason for your choice.\n"
        "Respect the budget: do not pick a product whose price exceeds the budget.\n"
        "Reply in the same language as the user's request.\n"
    )


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
        parts += ["", f"Previous attempt failed: {last_error}", "Pick a valid product from the list."]
    return "\n".join(parts)
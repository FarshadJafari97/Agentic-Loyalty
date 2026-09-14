# agent/nodes.py
from __future__ import annotations
from .state import AgentState
from .schemas import CategoryChoice, PurchaseChoice
from .prompts import (
    category_system_prompt, category_user_prompt,
    purchase_system_prompt, purchase_user_prompt,
)
from store.engine import StoreEnv


# ── Node: extract_category ────────────────────────────────
def make_extract_category(llm, allowed_categories: list[str]):
    structured = llm.with_structured_output(CategoryChoice)

    def extract_category(state: AgentState) -> dict:
        result = structured.invoke([
            {"role": "system", "content": category_system_prompt(allowed_categories)},
            {"role": "user",   "content": category_user_prompt(
                state["user_request"], state.get("last_category_error"),
            )},
        ])
        return {
            "category": result.category,
            "category_retries": state.get("category_retries", 0),
            "last_category_error": None,
        }
    return extract_category


# ── Node: fetch_products (no LLM) ─────────────────────────
def make_fetch_products(env: StoreEnv):
    def fetch_products(state: AgentState) -> dict:
        products = env.get_products(state["category"])
        if not products:
            return {
                "products": [],
                "category_retries": state.get("category_retries", 0) + 1,
                "last_category_error": (
                    f"category '{state['category']}' returned no products this round. "
                    f"Pick a different category."
                ),
            }
        return {
            "products": [p.model_dump() for p in products],
            "last_category_error": None,
        }
    return fetch_products


# ── Node: decide (LLM picks product + reason) ─────────────
def make_decide(llm):
    structured = llm.with_structured_output(PurchaseChoice)

    def decide(state: AgentState) -> dict:
        result = structured.invoke([
            {"role": "system", "content": purchase_system_prompt()},
            {"role": "user",   "content": purchase_user_prompt(
                user_request=state["user_request"],
                budget=state["budget"],
                history=state["history"],
                products=state["products"],
                last_error=state.get("last_commit_error"),
            )},
        ])
        return {
            "chosen_product_id": result.product_id,
            "chosen_reason": result.reason,
            "last_commit_error": None,
        }
    return decide


# ── Node: commit (no LLM) ─────────────────────────────────
def make_commit(env: StoreEnv):
    def commit(state: AgentState) -> dict:
        result = env.commit_purchase(
            product_id=state["chosen_product_id"],
            reason=state["chosen_reason"],
        )
        if result.ok:
            return {"status": "committed", "failure_reason": None}
        return {
            "commit_retries": state.get("commit_retries", 0) + 1,
            "last_commit_error": (
                f"commit failed: {result.reason}. "
                f"Pick a product that is available this round and within budget."
            ),
        }
    return commit


# ── Node: finalize ────────────────────────────────────────
def make_finalize(env: StoreEnv):
    def finalize(state: AgentState) -> dict:
        if state.get("status") == "committed":
            return {}
        # not committed → record failed round
        reason = (
            state.get("last_commit_error")
            or state.get("last_category_error")
            or "agent_failed"
        )
        env.record_failed_round(reason)
        return {"status": "failed", "failure_reason": reason}
    return finalize
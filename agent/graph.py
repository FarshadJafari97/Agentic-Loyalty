# agent/graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    make_extract_category, make_fetch_products,
    make_decide, make_commit, make_finalize,
)
from store.engine import StoreEnv


def build_graph(
    env: StoreEnv,
    llm,
    *,
    allowed_categories: list[str],
    max_category_retries: int = 2,
    max_commit_retries: int = 3,
):
    g = StateGraph(AgentState)

    g.add_node("extract_category", make_extract_category(llm, allowed_categories))
    g.add_node("fetch_products",   make_fetch_products(env))
    g.add_node("decide",           make_decide(llm))
    g.add_node("commit",           make_commit(env))
    g.add_node("finalize",         make_finalize(env))

    g.set_entry_point("extract_category")

    g.add_edge("extract_category", "fetch_products")

    # fetch_products → decide if products exist
    #                → extract_category if retries left
    #                → finalize if exhausted
    def after_fetch(state: AgentState) -> str:
        if state["products"]:
            return "decide"
        if state.get("category_retries", 0) < max_category_retries:
            return "extract_category"
        return "finalize"

    g.add_conditional_edges("fetch_products", after_fetch, {
        "decide": "decide",
        "extract_category": "extract_category",
        "finalize": "finalize",
    })

    g.add_edge("decide", "commit")

    # commit → finalize if committed
    #        → decide   if retries left
    #        → finalize if exhausted
    def after_commit(state: AgentState) -> str:
        if state.get("status") == "committed":
            return "finalize"
        if state.get("commit_retries", 0) < max_commit_retries:
            return "decide"
        return "finalize"

    g.add_conditional_edges("commit", after_commit, {
        "decide": "decide",
        "finalize": "finalize",
    })

    g.add_edge("finalize", END)

    return g.compile()
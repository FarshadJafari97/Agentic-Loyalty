# orchestrator.py
from __future__ import annotations
from store.engine import StoreEnv
from store.models import ProductSpec, RoundSpec
from agent.graph import build_graph


def run_episode(
    env: StoreEnv,
    llm,
    *,
    user_requests: list[str],          # one per round, length == len(schedule)
    allowed_categories: list[str],
    max_category_retries: int = 2,
    max_commit_retries: int = 3,
):
    """Run one full episode: env loops through all rounds, calling the agent each round."""
    graph = build_graph(
        env, llm,
        allowed_categories=allowed_categories,
        max_category_retries=max_category_retries,
        max_commit_retries=max_commit_retries,
    )

    round_idx = 0
    while not env.finished:
        env.begin_round()

        initial_state = {
            "user_request": user_requests[round_idx],
            "budget": env.budget,
            "history": [h.model_dump() for h in env.history],
            "allowed_categories": allowed_categories,
            "products": [],
            "category_retries": 0,
            "commit_retries": 0,
            "status": "in_progress",
        }

        final_state = graph.invoke(initial_state)

        if final_state.get("status") != "committed":
            # finalize already called env.record_failed_round
            pass

        env.close_round()
        round_idx += 1

    return env
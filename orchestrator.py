# orchestrator.py
"""Orchestrator for a single trajectory.

Runs the agent graph round by round. The environment and the agent know
nothing about the database; persistence is delegated to an optional
callback (`on_round_complete`) provided by the caller.
"""
from __future__ import annotations
from typing import Callable

from store.engine import StoreEnv
from agent.graph import build_graph


def run_single_round(
    env: StoreEnv,
    graph,
    *,
    user_request: str,
    allowed_categories: list[str],
):
    """Execute one round of the agent graph against the env.

    Caller MUST have called env.begin_round() before this.
    Caller MUST call env.close_round() after this.
    """
    initial_state = {
        "user_request": user_request,
        "budget": env.budget,
        "history": [h.model_dump() for h in env.history],
        "allowed_categories": allowed_categories,
        "products": [],
        "category_retries": 0,
        "commit_retries": 0,
        "status": "in_progress",
    }
    return graph.invoke(initial_state)


def run_trajectory(
    env: StoreEnv,
    llm,
    *,
    user_requests: list[str],
    allowed_categories: list[str],
    max_category_retries: int = 2,
    max_commit_retries: int = 3,
    on_round_complete: Callable[[StoreEnv, int], None] | None = None,
) -> StoreEnv:
    """Run all rounds of one trajectory.

    Parameters
    ----------
    env : StoreEnv
        The environment. It is reset by the caller (a fresh instance per
        trajectory) and is the only source of truth during the run.
    llm
        The language model bound to the graph.
    user_requests : list[str]
        One request per round. Its length must equal env.max_rounds.
    allowed_categories : list[str]
        Categories the agent is allowed to request.
    max_category_retries, max_commit_retries : int
        Retry budgets passed to the graph.
    on_round_complete : callable, optional
        Called once per round, AFTER env.close_round() has advanced the
        round counter. Signature: (env, finished_round_number) -> None.
        Use it for persistence. Errors raised here propagate up and
        abort the trajectory.

    Returns
    -------
    StoreEnv
        The same env instance, now finished.
    """
    if len(user_requests) != env.max_rounds:
        raise ValueError(
            f"user_requests length ({len(user_requests)}) does not match "
            f"env.max_rounds ({env.max_rounds})"
        )

    graph = build_graph(
        env,
        llm,
        allowed_categories=allowed_categories,
        max_category_retries=max_category_retries,
        max_commit_retries=max_commit_retries,
    )

    round_idx = 0
    while not env.finished:
        env.begin_round()

        run_single_round(
            env,
            graph,
            user_request=user_requests[round_idx],
            allowed_categories=allowed_categories,
        )

        env.close_round()

        if on_round_complete is not None:
            # env._round has already advanced; pass the number of the round
            # that just finished explicitly.
            on_round_complete(env, round_idx + 1)

        round_idx += 1

    return env
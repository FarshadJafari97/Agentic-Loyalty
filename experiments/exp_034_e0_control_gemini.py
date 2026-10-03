# experiments/exp_034_e0_control_gemini.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ1_E0_control_gemini",
    "rq_id": 1,
    "name": "E0 Control (no history, gemini-3.5-flash-lite) — Baseline brand preference at parity",
    "description": (
        "Same design as RQ1_E0_control but run with gemini-3.5-flash-lite. "
        "Control for RQ1. Single-round trajectories with empty history and "
        "strict price parity (15.0 vs 15.0) on Laundry Detergent. Measures the "
        "intrinsic pick probability of each of the three fictional brands "
        "(Nordvik / Zephyr / Auralis) without any seeding. Run 90 independent "
        "trajectories (90 purchases) to estimate the no-history baseline."
    ),

    # NOTE: runner.py uses the CLI count (default 50). Run with:
    #   python runner.py experiments/exp_034_e0_control_gemini.py 120
    "runs": 120,

    "llm": {
        "model": "gemini-3.5-flash-lite",
        "temperature": 0.7,
    },

    "agent": {
        "max_category_retries": 2,
        "max_commit_retries": 3,
    },

    "presentation": {
        "order": "shuffle",
        "seed": 232,
    },

    "catalog": [
        ProductSpec(
            product_id="p_nordvik",
            name="Laundry Detergent",
            category="cleaning",
            brand="Nordvik",
            quality=0.8,
        ),
        ProductSpec(
            product_id="p_zephyr",
            name="Laundry Detergent",
            category="cleaning",
            brand="Zephyr",
            quality=0.8,
        ),
        ProductSpec(
            product_id="p_auralis",
            name="Laundry Detergent",
            category="cleaning",
            brand="Auralis",
            quality=0.8,
        ),
    ],

    # ── Schedule: single round at parity, no history ──
    "schedule": [
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=15.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
    ],

    "user_requests": ["I want laundry detergent"],
}

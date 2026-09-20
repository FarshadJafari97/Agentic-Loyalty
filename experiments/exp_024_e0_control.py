# experiments/exp_024_e0_control.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ1_E0_control",
    "rq_id": 1,
    "name": "E0 Control (no history) — Baseline brand preference at parity",
    "description": (
        "Control for RQ1. Single-round trajectories with empty history and "
        "strict price parity (15.0 vs 15.0) on Laundry Detergent. Measures the "
        "intrinsic pick probability of each of the three fictional brands "
        "(Nordvik / Zephyr / Auralis) without any seeding. Run 90 independent "
        "trajectories (90 purchases) to estimate the no-history baseline."
    ),

    # NOTE: runner.py uses the CLI count (default 50). Run with:
    #   python runner.py experiments/exp_024_e0_control.py 90
    "runs": 90,

    "llm": {
        "model": "gpt-5.6-luna",
        "temperature": 0.7,
    },

    "agent": {
        "max_category_retries": 2,
        "max_commit_retries": 3,
    },

    "presentation": {
        "order": "shuffle",
        "seed": 100,
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

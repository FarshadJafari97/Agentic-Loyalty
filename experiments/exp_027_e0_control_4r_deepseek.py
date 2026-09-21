# experiments/exp_027_e0_control_4r_deepseek.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ1_E0_control_4r_deepseek",
    "rq_id": 1,
    "name": "E0 Control 4-round (deepseek-4.1-flash) — Baseline at sustained parity",
    "description": (
        "Same design as RQ1_E0_control_4r but run with deepseek-4.1-flash. "
        "Four rounds, all at strict price parity (15.0 vs 15.0) on "
        "Laundry Detergent with no discount seeding. Run 120 independent "
        "trajectories."
    ),

    # NOTE: runner.py uses the CLI count (default 50). Run with:
    #   python runner.py experiments/exp_027_e0_control_4r_deepseek.py 120
    "runs": 120,

    "llm": {
        "model": "deepseek-v4.1-flash",
        "temperature": 0.7,
    },

    "agent": {
        "max_category_retries": 2,
        "max_commit_retries": 3,
    },

    "presentation": {
        "order": "shuffle",
        "seed": 225,
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

    # ── Schedule: 4 rounds, all at parity ──
    "schedule": [
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=15.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=15.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=15.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=15.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
    ],

    "user_requests": ["I want laundry detergent"] * 4,
}

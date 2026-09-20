# experiments/exp_012_e2_k3_p5.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ2_E2_k3_p5",
    "rq_id": 2,
    "name": "E2 (k=3, +5% Price Premium) — Low Price Shock",
    "description": (
        "Evaluates price resilience under a 5% premium after 3 discount rounds. "
        "Rounds 1-3 seed Nordvik with a 20% discount (12.0 vs 15.0). "
        "Round 4 tests choice when Nordvik is 5% more expensive (15.75 vs 15.0)."
    ),

    "runs": 50,

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
        "seed": 212,
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

    # ── Schedule: 3 Seeding Rounds + 1 Evaluation Round (+5% Shock) ──
    "schedule": [
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=12.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=12.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=12.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
        # Round 4: +5% price premium shock (15.75 vs 15.0)
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=15.75),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
    ],

    "user_requests": ["I want laundry detergent"] * 4,
}

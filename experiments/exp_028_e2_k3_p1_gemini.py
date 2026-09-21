# experiments/exp_028_e2_k3_p1_gemini.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ2_E2_k3_p1_gemini",
    "rq_id": 2,
    "name": "E2 (k=3, +1%, gemini-3.5-flash-lite) — Low Price Shock",
    "description": (
        "Same design as RQ2_E2_k3_p1 but run with gemini-3.5-flash-lite. "
        "Rounds 1-3 seed Nordvik with a 20% discount (12.0 vs 15.0). "
        "Round 4 tests choice when Nordvik is 1% more expensive (15.15 vs 15.0)."
    ),

    "runs": 50,

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
        "seed": 226,
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

    # ── Schedule: 3 Seeding Rounds + 1 Evaluation Round (+1% Shock) ──
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
        # Round 4: +1% price premium shock (15.15 vs 15.0)
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=15.15),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
    ],

    "user_requests": ["I want laundry detergent"] * 4,
}

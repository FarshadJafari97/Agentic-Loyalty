# experiments/exp_001_e1_k1.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ1_E1_k1_seeding",
    "rq_id": 1,
    "name": "E1 Treatment (k=1) — 1 Round Discount followed by 6 Parity Rounds",
    "description": (
        "Investigates algorithmic brand loyalty and state dependence decay. "
        "Nordvik receives an initial 20% promotional discount in round 1 (12.0 vs 15.0), "
        "followed by 6 post-discount evaluation rounds under strict price parity (15.0) "
        "and equal quality."
    ),

    # Number of independent trajectory repetitions
    "runs": 50,

    "llm": {
        "model": "gpt-5.6-luna",
        # enough logical diversity while maintaining reliable JSON output
        "temperature": 0.7,
    },

    "agent": {
        "max_category_retries": 2,
        "max_commit_retries": 3,
    },

    # ── Position Bias Control ─────────────────
    # The catalog order is randomly shuffled in every round and trajectory
    # to prevent Nordvik's choice from being driven by its position in the list.
    "presentation": {
        "order": "shuffle",
        "seed": 101,  # For each trajectory: seed + (run_index - 1)
    },

    # Use laundry detergent instead of milk to align with the brand
    # contagion experiment (E3)
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

    # ── Price Schedule: 1 Discount Round + 6 Post-Discount Evaluation Rounds ──
    "schedule": [
        # Round 1 (initial Nordvik discount):
        RoundSpec(
            budget=100.0,
            listings={
                "p_nordvik": ListingEntry(available=1, price=12.0),
                "p_zephyr": ListingEntry(available=1, price=15.0),
                "p_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
        # Rounds 2–7 (full price parity — post-discount):
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

    "user_requests": ["I want laundry detergent"] * 7,
}
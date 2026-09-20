# experiments/exp_022_e3_k3_parity.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ3_E3_k3_parity",
    "rq_id": 3,
    "name": "E3 (k=3, parity) — Brand Spillover at Equal Price",
    "description": (
        "Tests umbrella branding / brand spillover. "
        "Rounds 1-3 seed Nordvik with a 20% discount on Laundry Detergent "
        "(12.0 vs 15.0). Round 4 switches to Dish Soap at full price parity "
        "(15.0 vs 15.0) to see if the agent still picks Nordvik."
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
        "seed": 222,
    },

    "catalog": [
        ProductSpec(
            product_id="ld_nordvik",
            name="Laundry Detergent",
            category="laundry",
            brand="Nordvik",
            quality=0.8,
        ),
        ProductSpec(
            product_id="ld_zephyr",
            name="Laundry Detergent",
            category="laundry",
            brand="Zephyr",
            quality=0.8,
        ),
        ProductSpec(
            product_id="ld_auralis",
            name="Laundry Detergent",
            category="laundry",
            brand="Auralis",
            quality=0.8,
        ),
        ProductSpec(
            product_id="ds_nordvik",
            name="Dish Soap",
            category="dish",
            brand="Nordvik",
            quality=0.8,
        ),
        ProductSpec(
            product_id="ds_zephyr",
            name="Dish Soap",
            category="dish",
            brand="Zephyr",
            quality=0.8,
        ),
        ProductSpec(
            product_id="ds_auralis",
            name="Dish Soap",
            category="dish",
            brand="Auralis",
            quality=0.8,
        ),
    ],

    # ── Schedule: 3 Seeding Rounds (laundry) + 1 Spillover Test (dish, parity) ──
    "schedule": [
        # Rounds 1-3: seed Nordvik on laundry (20% discount)
        RoundSpec(
            budget=100.0,
            listings={
                "ld_nordvik": ListingEntry(available=1, price=12.0),
                "ld_zephyr": ListingEntry(available=1, price=15.0),
                "ld_auralis": ListingEntry(available=1, price=15.0),
                "ds_nordvik": ListingEntry(available=0, price=15.0),
                "ds_zephyr": ListingEntry(available=0, price=15.0),
                "ds_auralis": ListingEntry(available=0, price=15.0),
            },
        ),
        RoundSpec(
            budget=100.0,
            listings={
                "ld_nordvik": ListingEntry(available=1, price=12.0),
                "ld_zephyr": ListingEntry(available=1, price=15.0),
                "ld_auralis": ListingEntry(available=1, price=15.0),
                "ds_nordvik": ListingEntry(available=0, price=15.0),
                "ds_zephyr": ListingEntry(available=0, price=15.0),
                "ds_auralis": ListingEntry(available=0, price=15.0),
            },
        ),
        RoundSpec(
            budget=100.0,
            listings={
                "ld_nordvik": ListingEntry(available=1, price=12.0),
                "ld_zephyr": ListingEntry(available=1, price=15.0),
                "ld_auralis": ListingEntry(available=1, price=15.0),
                "ds_nordvik": ListingEntry(available=0, price=15.0),
                "ds_zephyr": ListingEntry(available=0, price=15.0),
                "ds_auralis": ListingEntry(available=0, price=15.0),
            },
        ),
        # Round 4: spillover test on dish soap at parity (15.0 vs 15.0)
        RoundSpec(
            budget=100.0,
            listings={
                "ld_nordvik": ListingEntry(available=0, price=12.0),
                "ld_zephyr": ListingEntry(available=0, price=15.0),
                "ld_auralis": ListingEntry(available=0, price=15.0),
                "ds_nordvik": ListingEntry(available=1, price=15.0),
                "ds_zephyr": ListingEntry(available=1, price=15.0),
                "ds_auralis": ListingEntry(available=1, price=15.0),
            },
        ),
    ],

    "user_requests": ["I want laundry detergent"] * 3 + ["I want dish soap"],
}

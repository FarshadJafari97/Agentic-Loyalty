# experiments/exp_000_test.py
from store.models import ProductSpec, RoundSpec, ListingEntry

EXPERIMENT = {
    "code": "RQ1_Test_10rounds",
    "rq_id": 1,                              
    "name": "Baseline — 10 rounds of milk with price changes",
    "description": (
        "No discount condition. Three milk products with equal quality and "
        "fictional brands. Prices change across 10 rounds to observe the "
        "agent's loyalty vs price-sensitivity."
    ),

    "llm": {
        "model": "gpt-5.6-luna",
        "temperature": 1.0,
    },

    "agent": {
        "max_category_retries": 2,
        "max_commit_retries": 3,
    },

    "catalog": [
        ProductSpec(product_id="m_nordvik", name="Milk", category="dairy",
                    brand="Nordvik", quality=0.8),
        ProductSpec(product_id="m_zephyr",  name="Milk", category="dairy",
                    brand="Zephyr",  quality=0.8),
        ProductSpec(product_id="m_auralis", name="Milk", category="dairy",
                    brand="Auralis", quality=0.8),
    ],

    "schedule": [
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=14.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=14.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=14.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=15.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=16.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=17.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=18.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=19.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=20.0),
            "m_zephyr":  ListingEntry(available=1, price=15.0),
            "m_auralis": ListingEntry(available=1, price=18.0),
        }),
        RoundSpec(budget=100.0, listings={
            "m_nordvik": ListingEntry(available=1, price=21.0),
            "m_zephyr":  ListingEntry(available=1, price=16.0),
            "m_auralis": ListingEntry(available=1, price=14.0),
        }),
    ],

    "user_requests": ["I want milk"] * 10,
}
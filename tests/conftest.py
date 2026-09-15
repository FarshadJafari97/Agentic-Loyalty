# tests/conftest.py
import pytest
from store.engine import StoreEnv
from store.models import ProductSpec, ListingEntry, RoundSpec


@pytest.fixture
def catalog():
    return [
        ProductSpec(product_id="p1", name="Butter", category="dairy", brand="A", quality=0.8),
        ProductSpec(product_id="p2", name="Milk",   category="dairy", brand="B", quality=0.6),
        ProductSpec(product_id="p3", name="Soap",   category="laundry", brand="A", quality=0.7),
        ProductSpec(product_id="p4", name="Yogurt", category="dairy", brand="A", quality=0.9),
    ]


@pytest.fixture
def schedule():
    return [
        RoundSpec(budget=100.0, listings={
            "p1": ListingEntry(available=1, price=10.0),
            "p2": ListingEntry(available=1, price=15.0),
            "p3": ListingEntry(available=1, price=20.0),
            "p4": ListingEntry(available=0, price=12.0),  # Unavailable
        }),
        RoundSpec(budget=50.0, listings={
            "p1": ListingEntry(available=0, price=10.0),
            "p2": ListingEntry(available=1, price=15.0),
            "p3": ListingEntry(available=1, price=80.0),   # Over Budget
            "p4": ListingEntry(available=1, price=12.0),
        }),
        RoundSpec(budget=100.0, listings={
            "p1": ListingEntry(available=1, price=10.0),
            "p2": ListingEntry(available=1, price=15.0),
            "p3": ListingEntry(available=1, price=20.0),
            "p4": ListingEntry(available=1, price=12.0),
        }),
    ]


@pytest.fixture
def env(catalog, schedule):
    return StoreEnv(catalog=catalog, schedule=schedule) 
# store/schedule.py
from .models import ListingEntry, RoundSpec

# Full schedule for 12 rounds, defined by us.
# Each entry carries: per-round budget + per-product availability & price.
SCHEDULE: list[RoundSpec] = [
    RoundSpec(
        budget=100.0,
        listings={
            "p1": ListingEntry(available=1, price=10.0),
            "p2": ListingEntry(available=1, price=15.0),
            "p3": ListingEntry(available=1, price=20.0),
        },
    ),
    RoundSpec(
        budget=100.0,
        listings={
            "p1": ListingEntry(available=1, price=10.0),
            "p2": ListingEntry(available=0, price=15.0),
            "p3": ListingEntry(available=1, price=20.0),
        },
    ),
    # ... rounds 3 through 12
]
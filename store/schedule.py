# store/schedule.py
from .models import ListingEntry

# فرض: ۳ محصول در کاتالوگ
SCHEDULE: list[dict[str, ListingEntry]] = [
    # راند ۱
    {
        "p1": ListingEntry(available=1, price=10.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=20.0),
    },
    # راند ۲
    {
        "p1": ListingEntry(available=1, price=10.0),
        "p2": ListingEntry(available=0, price=15.0),
        "p3": ListingEntry(available=1, price=20.0),
    },
    # ... تا ۱۲ راند
]
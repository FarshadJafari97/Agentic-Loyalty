# scripts/smoke_test.py
"""End-to-end smoke test for the shopping agent (agent-only, no taxonomy, no DB).

Runs a 10-round episode with a real LLM. All rounds ask for milk.
Three milk products with fictional brands, all with equal quality,
so the only signal is price and prior-purchase loyalty.

This is essentially a single-run "dry run" of the baseline experiment,
without persisting anything to the database. Use it before running
runner.py to make sure the whole pipeline works.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/smoke_test.py
"""
from __future__ import annotations
from dotenv import load_dotenv

import os
import sys
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from store.engine import StoreEnv
from store.models import ProductSpec, RoundSpec, ListingEntry
from orchestrator import run_trajectory


# ── Catalog: 3 milk products, fictional brands, equal quality ──
CATALOG = [
    ProductSpec(product_id="p1", name="Milk", category="dairy",
                brand="Nordvik", quality=0.8),
    ProductSpec(product_id="p2", name="Milk", category="dairy",
                brand="Zephyr",  quality=0.8),
    ProductSpec(product_id="p3", name="Milk", category="dairy",
                brand="Auralis", quality=0.8),
]

# ── Schedule: 10 rounds with price changes ────────────────
SCHEDULE = [
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=16.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=17.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=18.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=19.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=20.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=15.0),
    }),
]

USER_REQUESTS = ["I want milk"] * len(SCHEDULE)

# ── Presentation order ──────────────────────────────────────
# {"order": "schedule"} keeps the listings-dict order above.
# {"order": "shuffle", "seed": N} deterministically shuffles per round
# from seed N (same seed + round => same shown order, reproducible).
PRESENTATION = {
    "order": "shuffle",
    "seed": 412,
}


def main() -> None:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: API_KEY is not set.")
        print("Create a .env file with API_KEY=... next to the project root.")
        sys.exit(1)

    llm = ChatOpenAI(
        model=      "deepseek-v4.1-flash",
        base_url=    os.environ.get("BASE_URL"),
        api_key=     api_key,
        temperature= 1.0
    )
    env = StoreEnv(
        catalog=CATALOG,
        schedule=SCHEDULE,
        order=PRESENTATION.get("order", "schedule"),
        seed=PRESENTATION.get("seed"),
    )
    print(f"Presentation: {PRESENTATION}")

    allowed_categories = sorted({p.category for p in CATALOG})
    print(f"Allowed categories: {allowed_categories}")

    run_trajectory(
        env=env,
        llm=llm,
        user_requests=USER_REQUESTS,
        allowed_categories=allowed_categories,
        max_category_retries=2,
        max_commit_retries=3,
    )

    # ── Report ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PURCHASES")
    print("=" * 60)
    if not env.purchases:
        print("  (none)")
    for p in env.purchases:
        print(
            f"  round {p.round:>2}: "
            f"{p.product_name} / {p.brand} / {p.price_paid} "
            f'— "{p.reason_text}"'
        )

    print("\n" + "=" * 60)
    print("FAILED ROUNDS")
    print("=" * 60)
    if not env.failed_rounds:
        print("  (none)")
    for f in env.failed_rounds:
        print(f"  round {f.round}: {f.reason}")

    print("\n" + "=" * 60)
    print("PRESENTED ORDER PER ROUND")
    print("=" * 60)
    for r in range(1, len(SCHEDULE) + 1):
        print(f"  round {r:>2}: {env.shown_orders.get(r, [])}")

    print("\n" + "=" * 60)
    print("PRICE PER ROUND (for reference)")
    print("=" * 60)
    all_pids = sorted({pid for spec in SCHEDULE for pid in spec.listings})
    header = "       " + "  ".join(f"{pid:>8}" for pid in all_pids)
    print(header)
    for r, spec in enumerate(SCHEDULE, start=1):
        cells = []
        for pid in all_pids:
            entry = spec.listings.get(pid)
            if entry is None or not entry.available:
                cells.append(f"{'--':>8}")
            else:
                cells.append(f"{entry.price:>8.1f}")
        print(f"  r{r:>2}: " + "  ".join(cells))

    # ── Sanity checks ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    total = len(env.purchases) + len(env.failed_rounds)
    ok = True

    check = total == len(SCHEDULE)
    print(f"  every round accounted for: {check} ({total}/{len(SCHEDULE)})")
    ok &= check

    check = all(p.reason_text and p.reason_text.strip() for p in env.purchases)
    print(f"  every purchase has a non-empty reason_text: {check}")
    ok &= check

    check = all(
        p.reason_code is None and p.reason_note is None
        for p in env.purchases
    )
    print(f"  reason_code is unset (to be filled by classifier): {check}")
    ok &= check

    rounds_with_purchase = {p.round for p in env.purchases}
    rounds_failed = {f.round for f in env.failed_rounds}
    check = rounds_with_purchase.isdisjoint(rounds_failed)
    print(f"  no round is both committed and failed: {check}")
    ok &= check

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
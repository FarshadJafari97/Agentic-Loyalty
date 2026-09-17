# scripts/smoke_test.py
"""End-to-end smoke test for the shopping agent.

Runs a small 3-round episode with a real LLM and prints everything:
purchases, failed rounds, and the full event log.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from store.engine import StoreEnv
from store.models import ProductSpec, RoundSpec, ListingEntry
from orchestrator import run_episode


# ── Catalog (fixed for this smoke test) ───────────────────
CATALOG = [
    ProductSpec(product_id="p1", name="Milk",             category="dairy",   brand="A", quality=0.8),
    ProductSpec(product_id="p2", name="Milk",             category="dairy",   brand="B", quality=0.8),
    ProductSpec(product_id="p3", name="Milk",             category="dairy",   brand="A", quality=0.8),
    ProductSpec(product_id="p4", name="Dishwashing Liquid",    category="laundry", brand="A", quality=0.7),
    ProductSpec(product_id="p5", name="Dishwashing Liquid",    category="laundry", brand="B", quality=0.7),
]

# ── Schedule (3 rounds for smoke test) ────────────────────
SCHEDULE = [
    # Round 1
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=12.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=18.0),
        "p4": ListingEntry(available=1, price=25.0),
        "p5": ListingEntry(available=1, price=18.0),
    }),
    # Round 2: milk out of stock, detergent cheaper
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=18.0),
        "p4": ListingEntry(available=1, price=20.0),
        "p5": ListingEntry(available=1, price=18.0),
    }),
    # Round 3: everything available again
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=10.0),
        "p2": ListingEntry(available=1, price=15.0),
        "p3": ListingEntry(available=1, price=12.0),
        "p4": ListingEntry(available=1, price=28.0),
        "p5": ListingEntry(available=1, price=20.0),
    }),
]

# ── User requests (one per round) ─────────────────────────
USER_REQUESTS = [
    "I want milk",
    "Buy some Milk",
    "Buy Dishwashing Liquid",
]

# scripts/smoke_test.py — اضافه کن بعد از import‌ها

def check_reason_consistency(env, schedule, catalog) -> list[str]:
    """Verify that each reason_code is consistent with the actual choice.

    Returns a list of human-readable warnings. Empty list means everything
    is consistent.
    """
    issues: list[str] = []
    catalog_by_id = {p.product_id: p for p in catalog}

    for p in env.purchases:
        r = p.round
        round_spec = schedule[r - 1]
        chosen_spec = catalog_by_id[p.product_id]

        # Alternatives = other products in the SAME category available this round
        alternatives = []
        for pid, entry in round_spec.listings.items():
            if not entry.available:
                continue
            if pid == p.product_id:
                continue
            spec = catalog_by_id[pid]
            if spec.category != chosen_spec.category:
                continue
            alternatives.append((pid, spec, entry))

        if not alternatives:
            continue  # nothing to compare against

        cheapest_alt_price = min(e.price for _, _, e in alternatives)

        # Prior purchases strictly before this round
        prior = [pp for pp in env.purchases if pp.round < r]
        same_product_before = any(pp.product_id == p.product_id for pp in prior)
        same_brand_other_before = any(
            pp.brand == p.brand and pp.product_id != p.product_id
            for pp in prior
        )

        # ── Code 3 / 4 require same product bought before ──
        if p.reason_code in ("3", "4") and not same_product_before:
            issues.append(
                f"r{r}: code {p.reason_code} but this product_id "
                f"({p.product_id}) was never bought before"
            )

        # ── Code 5 / 6 require same brand but DIFFERENT product ──
        if p.reason_code in ("5", "6"):
            if same_product_before:
                issues.append(
                    f"r{r}: code {p.reason_code} but the SAME product_id "
                    f"was bought before (should be 3 or 4)"
                )
            if not same_brand_other_before:
                issues.append(
                    f"r{r}: code {p.reason_code} but no DIFFERENT product "
                    f"from brand {p.brand} was bought before"
                )

        # ── Price-equality rules ──
        if p.reason_code in ("3", "5"):
            if p.price_paid > cheapest_alt_price + 0.01:
                issues.append(
                    f"r{r}: code {p.reason_code} claims EQUAL price, but "
                    f"chosen={p.price_paid} > cheapest_alt={cheapest_alt_price}+0.01"
                )
        if p.reason_code in ("4", "6"):
            if p.price_paid <= cheapest_alt_price + 0.01:
                issues.append(
                    f"r{r}: code {p.reason_code} claims HIGHER price, but "
                    f"chosen={p.price_paid} <= cheapest_alt={cheapest_alt_price}+0.01"
                )

        # ── Code 1 requires chosen to be cheapest among equal-quality ──
        if p.reason_code == "1":
            equal_quality = [
                (pid, spec, e) for pid, spec, e in alternatives
                if abs(spec.quality - chosen_spec.quality) < 0.05
            ]
            if equal_quality:
                min_eq_price = min(e.price for _, _, e in equal_quality)
                if p.price_paid > min_eq_price + 0.01:
                    issues.append(
                        f"r{r}: code 1 claims LOWEST among equal-quality, but "
                        f"chosen={p.price_paid} > min equal-quality alt={min_eq_price}"
                    )
            else:
                issues.append(
                    f"r{r}: code 1 but no equal-quality alternative exists "
                    f"(chosen quality={chosen_spec.quality})"
                )

    return issues


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)

    llm = ChatOpenAI(
        model= "gpt-5.6-luna",
        base_url= "https://api.gapgpt.app/v1",

    )
    env = StoreEnv(catalog=CATALOG, schedule=SCHEDULE)

    allowed_categories = sorted({p.category for p in CATALOG})
    print(f"Allowed categories: {allowed_categories}")

    run_episode(
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
        note = f" — {p.reason_note}" if p.reason_note else ""
        print(
            f"  round {p.round}: "
            f"{p.product_name} / {p.brand} / {p.price_paid} "
            f"[code={p.reason_code}{note}]"
        )

    print("\n" + "=" * 60)
    print("FAILED ROUNDS")
    print("=" * 60)
    if not env.failed_rounds:
        print("  (none)")
    for f in env.failed_rounds:
        print(f"  round {f.round}: {f.reason}")

    print("\n" + "=" * 60)
    print("HISTORY (as the agent sees it)")
    print("=" * 60)
    for h in env.history:
        print(f"  {h.model_dump()}")

    print("\n" + "=" * 60)
    print("EVENT LOG")
    print("=" * 60)
    for e in env.event_log:
        payload = {k: v for k, v in e.payload.items() if k != "reason_note"}
        print(f"  [{e.seq}] r{e.round} {e.event.value}: {payload}")

    # ── Sanity checks ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    total = len(env.purchases) + len(env.failed_rounds)
    ok = True

    check = total == len(SCHEDULE)
    print(f"  every round accounted for: {check} ({total}/{len(SCHEDULE)})")
    ok &= check

    check = all(
        p.reason_code in {"1", "2", "3", "4", "5", "6", "7", "8"}
        for p in env.purchases
    )
    print(f"  all reason_codes are valid: {check}")
    ok &= check

    check = all(
        p.reason_code != "8" or p.reason_note
        for p in env.purchases
    )
    print(f"  OTHER (code 8) purchases have a note: {check}")
    ok &= check

    rounds_with_purchase = {p.round for p in env.purchases}
    rounds_failed = {f.round for f in env.failed_rounds}
    check = rounds_with_purchase.isdisjoint(rounds_failed)
    print(f"  no round is both committed and failed: {check}")
    ok &= check

    # ── Reason-code consistency ───────────────────────────
    print("\n" + "=" * 60)
    print("REASON-CODE CONSISTENCY")
    print("=" * 60)
    issues = check_reason_consistency(env, SCHEDULE, CATALOG)
    if not issues:
        print("  no issues found")
    else:
        for i in issues:
            print(f"  WARNING: {i}")
        ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    if not ok:
        sys.exit(1)

    


if __name__ == "__main__":
    main()
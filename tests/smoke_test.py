# scripts/smoke_test.py
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from store.engine import StoreEnv
from store.models import ProductSpec, RoundSpec, ListingEntry
from orchestrator import run_episode


# ── Catalog: 3 milk products, fictional brands, equal quality ──
CATALOG = [
    ProductSpec(product_id="m_nordvik", name="Milk", category="dairy",
                brand="Nordvik", quality=0.8),
    ProductSpec(product_id="m_zephyr",  name="Milk", category="dairy",
                brand="Zephyr",  quality=0.8),
    ProductSpec(product_id="m_auralis", name="Milk", category="dairy",
                brand="Auralis", quality=0.8),
]

# ── Schedule: 10 rounds with price changes ────────────────
SCHEDULE = [
    # Round 1 — baseline: Nordvik cheapest
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=14.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 2 — same prices: loyalty test
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=14.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 3 — Nordvik becomes expensive
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=14.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 4 — same as round 3
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=15.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 5 — Nordvik cheap again
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=16.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 6 — Zephyr becomes cheapest
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=17.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 7 — same as round 6
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=18.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 8 — Nordvik cheapest again
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=19.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 9 — same as round 8
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=20.0),
        "m_zephyr":  ListingEntry(available=1, price=15.0),
        "m_auralis": ListingEntry(available=1, price=18.0),
    }),
    # Round 10 — Nordvik expensive, Zephyr cheapest
    RoundSpec(budget=100.0, listings={
        "m_nordvik": ListingEntry(available=1, price=21.0),
        "m_zephyr":  ListingEntry(available=1, price=16.0),
        "m_auralis": ListingEntry(available=1, price=14.0),
    }),
]

# ── User requests: milk every round ───────────────────────
USER_REQUESTS = ["I want milk"] * 10


# ── Consistency checker ───────────────────────────────────
def check_reason_consistency(env, schedule, catalog) -> list[str]:
    """Verify that each reason_code is consistent with the actual choice."""
    issues: list[str] = []
    catalog_by_id = {p.product_id: p for p in catalog}

    for p in env.purchases:
        r = p.round
        round_spec = schedule[r - 1]
        chosen_spec = catalog_by_id[p.product_id]

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
            continue

        cheapest_alt_price = min(e.price for _, _, e in alternatives)

        prior = [pp for pp in env.purchases if pp.round < r]
        same_product_before = any(pp.product_id == p.product_id for pp in prior)
        same_brand_other_before = any(
            pp.brand == p.brand and pp.product_id != p.product_id
            for pp in prior
        )

        if p.reason_code in ("3", "4") and not same_product_before:
            issues.append(
                f"r{r}: code {p.reason_code} but this product_id "
                f"({p.product_id}) was never bought before"
            )

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


# ── Main ──────────────────────────────────────────────────
def main() -> None:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: API_KEY is not set.")
        print("Create a .env file with API_KEY=... next to the project root,")
        print("or export it in your shell: export API_KEY=sk-...")
        sys.exit(1)

    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        base_url=os.environ.get("BASE_URL"),
        api_key=api_key,
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
            f"  round {p.round:>2}: "
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
    print("PRICE PER ROUND (for reference)")
    print("=" * 60)
    for r, spec in enumerate(SCHEDULE, start=1):
        row = "  ".join(
            f"{pid.split('_')[1]}={spec.listings[pid].price:>5.1f}"
            for pid in ("m_nordvik", "m_zephyr", "m_auralis")
        )
        print(f"  r{r:>2}: {row}")

    print("\n" + "=" * 60)
    print("HISTORY (as the agent sees it)")
    print("=" * 60)
    for h in env.history:
        print(f"  {h.model_dump()}")

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
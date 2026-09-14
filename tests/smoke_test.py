# scripts/smoke_test.py

from langchain_openai import ChatOpenAI   
from store.engine import StoreEnv
from store.models import ProductSpec, RoundSpec, ListingEntry
from orchestrator import run_episode


CATALOG = [
    ProductSpec(product_id="p1", name="Milk",   category="dairy", brand="A", quality=0.8),
    ProductSpec(product_id="p2", name="Milk",   category="dairy", brand="B", quality=0.8),
]

SCHEDULE = [
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=10.0),
        "p2": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=12.0),
        "p2": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=15.0),
        "p2": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=16.0),
        "p2": ListingEntry(available=1, price=15.0),
    }),
    RoundSpec(budget=100.0, listings={
        "p1": ListingEntry(available=1, price=20.0),
        "p2": ListingEntry(available=1, price=15.0),
    }),
]

USER_REQUESTS = [
    "شیر میخوام",
    "شیر میخوام",
    "شیر میخوام",
    "شیر میخوام",
    "شیر میخوام",
    "شیر میخوام",
    ]


def main():
    llm = ChatOpenAI(
        model = "gemini-3.6-flash",
        base_url= "http://127.0.0.1:31415/v1" ,
        api_key= "freellmapi-704878672ffd732da01727053c869683948f26decc0f8713"
    )
    env = StoreEnv(catalog=CATALOG, schedule=SCHEDULE)

    run_episode(
        env=env,
        llm=llm,
        user_requests=USER_REQUESTS,
        allowed_categories=sorted({p.category for p in CATALOG}),
    )

    print("\n=== PURCHASES ===")
    for p in env.purchases:
        print(f"  round {p.round}: {p.product_name} ({p.brand}) - {p.price_paid} | {p.reason}")

    print("\n=== FAILED ===")
    for f in env.failed_rounds:
        print(f"  round {f.round}: {f.reason}")

    print("\n=== EVENTS ===")
    for e in env.event_log:
        print(f"  [{e.seq}] r{e.round} {e.event.value}: {e.payload}")


if __name__ == "__main__":
    main()
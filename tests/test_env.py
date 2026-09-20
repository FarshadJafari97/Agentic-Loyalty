# tests/test_env.py
import pytest


def test_initial_state(env):
    assert env.round == 1
    assert env.finished is False
    assert env.purchases == []
    assert env.failed_rounds == []
    assert env.history == []


def test_begin_round_loads_budget(env):
    env.begin_round()
    assert env.budget == 100.0
    env.close_round()
    env.begin_round()
    assert env.budget == 50.0


def test_begin_round_twice_raises(env):
    env.begin_round()
    with pytest.raises(RuntimeError, match="round already open"):
        env.begin_round()


def test_close_round_without_begin_raises(env):
    with pytest.raises(RuntimeError, match="round not open"):
        env.close_round()


def test_get_products_filters_category_and_availability(env):
    env.begin_round()
    dairy = env.get_products("dairy")
    ids = {p.product_id for p in dairy}
    assert ids == {"p1", "p2"}          # p4 unavailable, p3 not dairy
    laundry = env.get_products("laundry")
    assert {p.product_id for p in laundry} == {"p3"}
    unknown = env.get_products("nonexistent")
    assert unknown == []


def test_get_products_price_is_round_specific(env):
    env.begin_round()
    p1 = next(p for p in env.get_products("dairy") if p.product_id == "p1")
    assert p1.price == 10.0
    env.close_round()
    env.begin_round()
    # p1 unavailable in round 2
    assert all(p.product_id != "p1" for p in env.get_products("dairy"))


def test_commit_purchase_happy_path(env):
    env.begin_round()
    result = env.commit_purchase("p1", reason_text="cheap")
    assert result.ok is True
    assert result.record.round == 1
    assert result.record.product_name == "Butter"
    assert result.record.price_paid == 10.0
    assert result.record.reason_text == "cheap"
    assert result.record.reason_code is None
    assert result.record.reason_note is None
    assert len(env.purchases) == 1


def test_commit_unknown_product(env):
    env.begin_round()
    result = env.commit_purchase("p999", reason_text="x")
    assert result.ok is False
    assert result.reason == "unknown_product"
    assert env.purchases == []


def test_commit_unavailable_product(env):
    env.begin_round()
    result = env.commit_purchase("p4", reason_text="x")   # p4 unavailable in round 1
    assert result.ok is False
    assert result.reason == "not_available_this_round"


def test_commit_over_budget(env):
    env.begin_round()      # budget 100
    env.close_round()
    env.begin_round()      # budget 50
    result = env.commit_purchase("p3", reason_text="x")   # price 80
    assert result.ok is False
    assert result.reason == "over_budget"


def test_commit_outside_round_raises_validation(env):
    result = env.commit_purchase("p1", reason_text="x")
    assert result.ok is False
    assert result.reason == "round_not_open"


def test_record_failed_round(env):
    env.begin_round()
    env.record_failed_round("agent exhausted retries")
    assert len(env.failed_rounds) == 1
    assert env.failed_rounds[0].round == 1
    assert env.failed_rounds[0].reason == "agent exhausted retries"


def test_record_failed_round_twice_raises(env):
    env.begin_round()
    env.record_failed_round("first")
    with pytest.raises(RuntimeError, match="already marked as failed"):
        env.record_failed_round("second")


def test_record_failed_after_purchase_raises(env):
    env.begin_round()
    env.commit_purchase("p1", reason_text="x")
    with pytest.raises(RuntimeError, match="already has a purchase"):
        env.record_failed_round("x")


def test_history_mixes_committed_and_failed(env):
    # round 1: buy
    env.begin_round()
    env.commit_purchase("p1", reason_text="r1")
    env.close_round()

    # round 2: failed
    env.begin_round()
    env.record_failed_round("no valid product")
    env.close_round()

    # round 3: buy
    env.begin_round()
    env.commit_purchase("p2", reason_text="r3")
    env.close_round()

    h = env.history
    assert len(h) == 3
    assert h[0].round == 1 and h[0].status == "committed"
    assert h[0].product_id == "p1"
    assert h[0].reason_text == "r1"
    assert h[1].round == 2 and h[1].status == "failed"
    assert h[1].reason_text == "no valid product"
    assert h[2].round == 3 and h[2].status == "committed"
    assert h[2].product_id == "p2"


def test_finished_after_all_rounds(env):
    for _ in range(3):
        env.begin_round()
        env.commit_purchase("p1" if env.round == 1 else "p2", reason_text="x")
        env.close_round()
    assert env.finished is True
    with pytest.raises(RuntimeError, match="episode finished"):
        env.begin_round()


def test_max_rounds_property(env):
    assert env.max_rounds == 3


def test_invalid_schedule_unknown_product(catalog):
    from store.engine import StoreEnv
    from store.models import RoundSpec, ListingEntry
    bad_schedule = [RoundSpec(budget=10, listings={"zzz": ListingEntry(available=1, price=1)})]
    with pytest.raises(ValueError, match="unknown product"):
        StoreEnv(catalog=catalog, schedule=bad_schedule)


def test_invalid_schedule_negative_budget(catalog):
    from store.engine import StoreEnv
    from store.models import RoundSpec
    bad = [RoundSpec(budget=-1, listings={})]
    with pytest.raises(ValueError, match="negative budget"):
        StoreEnv(catalog=catalog, schedule=bad)


def test_event_log_sequence(env):
    env.begin_round()
    env.commit_purchase("p1", reason_text="x")
    env.close_round()
    events = [e.event.value for e in env.event_log]
    assert events == ["round_started", "purchase", "round_closed"]
    seqs = [e.seq for e in env.event_log]
    assert seqs == [1, 2, 3]


def test_get_products_logs_products_shown(env):
    env.begin_round()
    env.get_products("dairy")
    events = [e.event.value for e in env.event_log]
    assert events == ["round_started", "products_shown"]
    seqs = [e.seq for e in env.event_log]
    assert seqs == [1, 2]


def _three_product_env(**kwargs):
    from store.engine import StoreEnv
    from store.models import ProductSpec, RoundSpec, ListingEntry
    catalog = [
        ProductSpec(product_id="a", name="Milk", category="dairy", brand="A"),
        ProductSpec(product_id="b", name="Milk", category="dairy", brand="B"),
        ProductSpec(product_id="c", name="Milk", category="dairy", brand="C"),
        ProductSpec(product_id="d", name="Milk", category="dairy", brand="D"),
        ProductSpec(product_id="e", name="Milk", category="dairy", brand="E"),
    ]
    schedule = [RoundSpec(budget=100.0, listings={
        "a": ListingEntry(available=1, price=10.0),
        "b": ListingEntry(available=1, price=11.0),
        "c": ListingEntry(available=1, price=12.0),
        "d": ListingEntry(available=1, price=13.0),
        "e": ListingEntry(available=1, price=14.0),
    })]
    return StoreEnv(catalog=catalog, schedule=schedule, **kwargs)


def test_schedule_order_is_default():
    env = _three_product_env()
    env.begin_round()
    assert [p.product_id for p in env.get_products("dairy")] == ["a", "b", "c", "d", "e"]


def test_shuffle_requires_seed():
    import pytest
    with pytest.raises(ValueError, match="requires a seed"):
        _three_product_env(order="shuffle")


def test_shuffle_is_deterministic_for_same_seed():
    env1 = _three_product_env(order="shuffle", seed=42)
    env1.begin_round()
    order1 = [p.product_id for p in env1.get_products("dairy")]

    env2 = _three_product_env(order="shuffle", seed=42)
    env2.begin_round()
    order2 = [p.product_id for p in env2.get_products("dairy")]

    assert order1 == order2
    assert sorted(order1) == ["a", "b", "c", "d", "e"]


def test_shuffle_differs_across_rounds_with_same_seed():
    # Collect round-1 vs round-2 orders over several seeds; at least one
    # seed must give different orders (shuffling actually varies by round).
    from store.models import ProductSpec, RoundSpec, ListingEntry
    from store.engine import StoreEnv
    catalog = [
        ProductSpec(product_id="a", name="Milk", category="dairy", brand="A"),
        ProductSpec(product_id="b", name="Milk", category="dairy", brand="B"),
        ProductSpec(product_id="c", name="Milk", category="dairy", brand="C"),
        ProductSpec(product_id="d", name="Milk", category="dairy", brand="D"),
        ProductSpec(product_id="e", name="Milk", category="dairy", brand="E"),
    ]
    listings = {
        "a": ListingEntry(available=1, price=10.0),
        "b": ListingEntry(available=1, price=11.0),
        "c": ListingEntry(available=1, price=12.0),
        "d": ListingEntry(available=1, price=13.0),
        "e": ListingEntry(available=1, price=14.0),
    }
    schedule = [RoundSpec(budget=100.0, listings=dict(listings)) for _ in range(2)]
    differed = False
    for seed in range(10):
        env = StoreEnv(catalog=catalog, schedule=schedule, order="shuffle", seed=seed)
        env.begin_round()
        r1 = [p.product_id for p in env.get_products("dairy")]
        env.close_round()
        env.begin_round()
        r2 = [p.product_id for p in env.get_products("dairy")]
        if r1 != r2:
            differed = True
            break
    assert differed


def test_shown_order_logged():
    env = _three_product_env(order="shuffle", seed=7)
    env.begin_round()
    shown = [p.product_id for p in env.get_products("dairy")]
    assert env.shown_orders[1] == shown
    events = [e for e in env.event_log if e.event.value == "products_shown"]
    assert len(events) == 1
    assert events[0].payload["order"] == shown
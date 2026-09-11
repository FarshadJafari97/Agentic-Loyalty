# tests/test_engine.py
import pytest
from store.catalog import build_base_catalog, counterbalance
from store.engine import EngineConfig, StoreEngine
from store.models import Outcome, UserRequest


def make_engine(seed=7, budget=200.0, **kw):
    brands, products = build_base_catalog()
    brands, products, order, _ = counterbalance(brands, products, seed)
    user = UserRequest(need="مایع لباس‌شویی", budget=budget)
    return StoreEngine(products, user, seed, **kw)


def test_successful_purchase_reduces_stock():
    e = make_engine()
    e.begin_round()
    pid = e.visible_products()[0].product_id
    result = e.commit_purchase(pid)
    assert result.ok
    assert e.get_product(pid).stock == 99


def test_over_budget_is_rejected_without_side_effect():
    e = make_engine(budget=10.0)
    e.begin_round()
    pid = e.visible_products()[0].product_id
    result = e.commit_purchase(pid)
    assert not result.ok and result.reason == "over_budget"
    assert e.get_product(pid).stock == 100
    assert e.purchases == []


def test_out_of_stock_is_rejected():
    e = make_engine()
    pid = e.visible_products()[0].product_id
    e.close_round()
    e.apply_stock_shock({pid: 0})
    e.begin_round()
    assert e.commit_purchase(pid).reason == "out_of_stock"
    assert pid not in [p.product_id for p in e.visible_products()]


def test_price_shock_forbidden_mid_round():
    e = make_engine()
    e.begin_round()
    with pytest.raises(RuntimeError):
        e.apply_price_shock({"B0-P1": 50.0})


def test_discount_then_reset_restores_base_price():
    e = make_engine()
    pid = "B0-P1"
    e.apply_discount([pid], 0.15)
    assert e.get_product(pid).current_price == 85.0
    e.close_round()
    e.apply_price_shock({pid: 85.0})  # noop، فقط برای باز بودن دور
    e.reset_prices()
    assert e.get_product(pid).current_price == 100.0


def test_outcome_is_deterministic_given_seed():
    seq = []
    for _ in range(2):
        e = make_engine(seed=42)
        e.begin_round()
        pid = e.visible_products()[0].product_id
        seq.append([e.simulate_outcome(pid) for _ in range(5)])
    assert seq[0] == seq[1]


def test_two_engines_same_seed_produce_identical_logs():
    logs = []
    for _ in range(2):
        e = make_engine(seed=3)
        e.begin_round()
        pid = e.visible_products()[0].product_id
        e.commit_purchase(pid)
        e.simulate_outcome(pid)
        logs.append(e.event_log)
    assert [x.model_dump() for x in logs[0]] == [x.model_dump() for x in logs[1]]


def test_forced_purchase_history_via_stock_shock():
    """موجودی بقیه صفر → عامل فقط یک گزینه دارد = سابقه خرید اجباری."""
    e = make_engine(seed=11)
    e.begin_round()
    target = e.visible_products()[0].product_id
    others = [p.product_id for p in e.visible_products() if p.product_id != target]
    e.close_round()
    e.apply_stock_shock({pid: 0 for pid in others})
    e.begin_round()
    assert [p.product_id for p in e.visible_products()] == [target]
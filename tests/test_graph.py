# tests/test_graph.py
import pytest
from store.engine import StoreEnv
from store.models import ProductSpec, RoundSpec, ListingEntry
from agent.graph import build_graph
from agent.schemas import CategoryChoice, PurchaseChoice


# ── Fake LLM ──────────────────────────────────────────────
class FakeStructuredLLM:
    """Fake LLM that returns pre-canned structured responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.calls = []

    def with_structured_output(self, schema):
        parent = self

        class Bound:
            def invoke(self, messages):
                parent.calls.append((schema.__name__, messages))
                if parent._idx >= len(parent._responses):
                    raise RuntimeError("FakeStructuredLLM: no more responses")
                resp = parent._responses[parent._idx]
                parent._idx += 1
                return resp
        return Bound()


@pytest.fixture
def small_env():
    catalog = [
        ProductSpec(product_id="p1", name="Butter", category="dairy", brand="A", quality=0.8),
        ProductSpec(product_id="p2", name="Milk",   category="dairy", brand="B", quality=0.6),
    ]
    schedule = [
        RoundSpec(budget=100.0, listings={
            "p1": ListingEntry(available=1, price=10.0),
            "p2": ListingEntry(available=1, price=15.0),
        }),
    ]
    return StoreEnv(catalog=catalog, schedule=schedule)


def _invoke(graph, env, *, user_request="x", history=None,
            allowed_categories=("dairy", "laundry")):
    return graph.invoke({
        "user_request": user_request,
        "budget": env.budget,
        "history": history or [],
        "allowed_categories": list(allowed_categories),
        "products": [],
        "category_retries": 0,
        "commit_retries": 0,
        "status": "in_progress",
    })


def test_graph_happy_path(small_env):
    """Happy path: valid category → valid product → commit."""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p1", reason_text="cheap"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy", "laundry"])
    small_env.begin_round()

    final = _invoke(graph, small_env, user_request="I want dairy")

    assert final["status"] == "committed"
    assert len(small_env.purchases) == 1
    assert small_env.purchases[0].product_id == "p1"
    assert small_env.purchases[0].reason_text == "cheap"


def test_graph_category_retry_then_success(small_env):
    """Wrong category → empty list → retry → valid."""
    llm = FakeStructuredLLM([
        CategoryChoice(category="nonexistent"),
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p2", reason_text="ok"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy", "laundry"])
    small_env.begin_round()

    final = _invoke(graph, small_env)

    assert final["status"] == "committed"
    assert small_env.purchases[0].product_id == "p2"


def test_graph_category_exhausted_fails(small_env):
    """Always-wrong category → after max retries → failed."""
    llm = FakeStructuredLLM([
        CategoryChoice(category="bad1"),
        CategoryChoice(category="bad2"),
        CategoryChoice(category="bad3"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"], max_category_retries=2)
    small_env.begin_round()

    final = _invoke(graph, small_env, allowed_categories=("dairy",))

    assert final["status"] == "failed"
    assert len(small_env.failed_rounds) == 1
    assert small_env.purchases == []


def test_graph_commit_retry_then_success(small_env):
    """Bad commit (unknown product) → retry → valid."""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p999", reason_text="bad"),
        PurchaseChoice(product_id="p1",   reason_text="ok"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"])
    small_env.begin_round()

    final = _invoke(graph, small_env, allowed_categories=("dairy",))

    assert final["status"] == "committed"
    assert small_env.purchases[0].product_id == "p1"


def test_graph_commit_exhausted_fails(small_env):
    """Always-bad commit → failed."""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p999", reason_text="bad"),
        PurchaseChoice(product_id="p999", reason_text="bad"),
        PurchaseChoice(product_id="p999", reason_text="bad"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"], max_commit_retries=3)
    small_env.begin_round()

    final = _invoke(graph, small_env, allowed_categories=("dairy",))

    assert final["status"] == "failed"
    assert len(small_env.failed_rounds) == 1


def test_graph_history_passed_to_decide(small_env):
    """history must be embedded in the decision prompt."""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p1", reason_text="ok"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"])
    small_env.begin_round()

    history = [
        {
            "round": 1,
            "status": "committed",
            "product_id": "p2",
            "category": "dairy",
            "brand": "B",
            "product": "Milk",
            "price_paid": 15.0,
            "reason_text": "tried last week",
        },
    ]

    _invoke(graph, small_env, history=history, allowed_categories=("dairy",))

    decide_calls = [c for c in llm.calls if c[0] == "PurchaseChoice"]
    assert len(decide_calls) == 1
    user_msg = decide_calls[0][1][1]["content"]   # role=user
    assert "tried last week" in user_msg
    assert "Milk" in user_msg
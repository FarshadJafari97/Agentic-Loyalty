# tests/test_graph.py
import pytest
from store.engine import StoreEnv
from store.models import ProductSpec, RoundSpec, ListingEntry
from agent.graph import build_graph
from agent.schemas import CategoryChoice, PurchaseChoice


# ── Fake LLM ────────────────────────────────────────────────
class FakeStructuredLLM:
    """LLM جعلی که خروجی ساختارمند از پیش تعیین‌شده می‌دهد."""
    def __init__(self, responses):
        # responses: لیست از پاسخ‌ها به ترتیب (چرخشی یا بر اساس نوع)
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


def test_graph_happy_path(small_env):
    """مسیر موفق: category درست → product درست → commit"""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p1", reason="cheap"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy", "laundry"])
    small_env.begin_round()

    final = graph.invoke({
        "user_request": "I want dairy",
        "budget": small_env.budget,
        "history": [],
        "allowed_categories": ["dairy", "laundry"],
        "products": [],
        "category_retries": 0,
        "commit_retries": 0,
        "status": "in_progress",
    })

    assert final["status"] == "committed"
    assert len(small_env.purchases) == 1
    assert small_env.purchases[0].product_id == "p1"


def test_graph_category_retry_then_success(small_env):
    """category اشتباه → لیست خالی → retry → درست"""
    llm = FakeStructuredLLM([
        CategoryChoice(category="nonexistent"),   # بار اول اشتباه
        CategoryChoice(category="dairy"),         # retry
        PurchaseChoice(product_id="p2", reason="ok"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy", "laundry"])
    small_env.begin_round()

    final = graph.invoke({
        "user_request": "x", "budget": small_env.budget, "history": [],
        "allowed_categories": ["dairy", "laundry"],
        "products": [], "category_retries": 0, "commit_retries": 0,
        "status": "in_progress",
    })

    assert final["status"] == "committed"
    assert small_env.purchases[0].product_id == "p2"


def test_graph_category_exhausted_fails(small_env):
    """category همیشه اشتباه → بعد از max retries → failed"""
    llm = FakeStructuredLLM([
        CategoryChoice(category="bad1"),
        CategoryChoice(category="bad2"),
        CategoryChoice(category="bad3"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"], max_category_retries=2)
    small_env.begin_round()

    final = graph.invoke({
        "user_request": "x", "budget": small_env.budget, "history": [],
        "allowed_categories": ["dairy"],
        "products": [], "category_retries": 0, "commit_retries": 0,
        "status": "in_progress",
    })

    assert final["status"] == "failed"
    assert len(small_env.failed_rounds) == 1
    assert small_env.purchases == []


def test_graph_commit_retry_then_success(small_env):
    """commit اشتباه (محصول over budget) → retry → موفق"""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p999", reason="bad"),   # unknown → رد
        PurchaseChoice(product_id="p1",   reason="ok"),    # درست
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"])
    small_env.begin_round()

    final = graph.invoke({
        "user_request": "x", "budget": small_env.budget, "history": [],
        "allowed_categories": ["dairy"],
        "products": [], "category_retries": 0, "commit_retries": 0,
        "status": "in_progress",
    })

    assert final["status"] == "committed"
    assert small_env.purchases[0].product_id == "p1"


def test_graph_commit_exhausted_fails(small_env):
    """commit همیشه اشتباه → failed"""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p999", reason="bad"),
        PurchaseChoice(product_id="p999", reason="bad"),
        PurchaseChoice(product_id="p999", reason="bad"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"], max_commit_retries=3)
    small_env.begin_round()

    final = graph.invoke({
        "user_request": "x", "budget": small_env.budget, "history": [],
        "allowed_categories": ["dairy"],
        "products": [], "category_retries": 0, "commit_retries": 0,
        "status": "in_progress",
    })

    assert final["status"] == "failed"
    assert len(small_env.failed_rounds) == 1


def test_graph_history_passed_to_decide(small_env):
    """history باید در prompt تصمیم‌گیری به LLM برسد."""
    llm = FakeStructuredLLM([
        CategoryChoice(category="dairy"),
        PurchaseChoice(product_id="p1", reason="ok"),
    ])
    graph = build_graph(small_env, llm, allowed_categories=["dairy"])
    small_env.begin_round()

    history = [
        {"round": 1, "status": "committed", "category": "dairy",
         "brand": "A", "product": "Butter", "reason": "cheap"},
    ]

    graph.invoke({
        "user_request": "x", "budget": small_env.budget, "history": history,
        "allowed_categories": ["dairy"],
        "products": [], "category_retries": 0, "commit_retries": 0,
        "status": "in_progress",
    })

    # second call = decide، باید history در promptش باشد
    decide_calls = [c for c in llm.calls if c[0] == "PurchaseChoice"]
    assert len(decide_calls) == 1
    user_msg = decide_calls[0][1][1]["content"]   # role=user
    assert "Butter" in user_msg
    assert "cheap" in user_msg
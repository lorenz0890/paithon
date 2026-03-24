import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self, implementations=None, repairs=None):
        self.implementations = implementations or {}
        self.repairs = repairs or {}
        self.calls = []

    def implement_function(self, request, model):
        self.calls.append(("implement", request.snapshot.qualname, model))
        return self.implementations[request.snapshot.name]

    def repair_function(self, request, model):
        self.calls.append(("repair", request.snapshot.qualname, request.error_type, model))
        return self.repairs[request.snapshot.name]


class StatefulMethodProvider(LLMProvider):
    def __init__(self):
        self.implementation_snapshots = []
        self.repair_snapshots = []

    def implement_function(self, request, model):
        self.implementation_snapshots.append(request.snapshot)
        if request.snapshot.name == "deposit":
            return (
                "def deposit(self, amount):\n"
                "    if amount < 0:\n"
                "        raise ValueError('amount must be non-negative')\n"
                "    self.balance += amount\n"
                "    self.history.append(('deposit', amount))\n"
                "    self.status = 'active'\n"
                "    return self.balance\n"
            )
        raise KeyError(request.snapshot.name)

    def repair_function(self, request, model):
        self.repair_snapshots.append(request.snapshot)
        if request.snapshot.name == "withdraw":
            return (
                "def withdraw(self, amount):\n"
                "    if amount < 0:\n"
                "        raise ValueError('amount must be non-negative')\n"
                "    if amount > self.balance:\n"
                "        raise ValueError('insufficient funds')\n"
                "    self.balance -= amount\n"
                "    self.history.append(('withdraw', amount))\n"
                "    self.status = 'active'\n"
                "    return self.balance\n"
            )
        raise KeyError(request.snapshot.name)


class EscalationProvider(LLMProvider):
    def __init__(self):
        self.calls = []
        self.outer_requests = []

    def implement_function(self, request, model):
        raise AssertionError("Unexpected implementation request")

    def repair_function(self, request, model):
        self.calls.append(("repair", request.snapshot.name, request.error_type, model))
        if request.snapshot.name == "fetch_tax_rate":
            raise RuntimeError("needs caller-level fallback")
        if request.snapshot.name == "compute_total":
            self.outer_requests.append(request)
            return (
                "def compute_total(subtotal, country):\n"
                "    rates = {'AT': 0.2, 'US': 0.0}\n"
                "    rate = rates.get(country, 0.0)\n"
                "    return subtotal * (1 + rate)\n"
            )
        raise KeyError(request.snapshot.name)


def make_runtime_add(engine):
    @engine.runtime_implemented
    def add(x, y):
        """Return the sum of two numbers."""
        raise NotImplementedError

    return add


def make_broken_mean(engine):
    @engine.self_healing
    def mean(values):
        """Return the arithmetic mean of a non-empty iterable of numbers."""
        return sum(values) / len(value)

    return mean


def make_nested_total(engine):
    @engine.self_healing
    def fetch_tax_rate(country):
        """Return the tax rate for a country code."""
        rates = {"AT": 0.2, "US": 0.0}
        return rates[country]

    @engine.self_healing
    def compute_total(subtotal, country):
        """Return subtotal with country-specific tax; fall back to subtotal if the tax rate is unavailable."""
        return subtotal * (1 + fetch_tax_rate(country))

    return fetch_tax_rate, compute_total


def make_wallet_class(engine):
    class Wallet:
        def __init__(self, balance):
            self.balance = balance
            self.status = "new"
            self.history = []

        @engine.runtime_implemented(state_fields=["balance", "status", "history"])
        def deposit(self, amount):
            """Increase balance by amount, append a history entry, and return the new balance."""
            raise NotImplementedError

        @engine.self_healing(state_fields=["balance", "status", "history"])
        def withdraw(self, amount):
            """Decrease balance by amount if funds are available, append a history entry, and return the new balance."""
            self.balance -= amunt
            self.history.append(("withdraw", amount))
            self.status = "active"
            return self.balance

    return Wallet


def test_runtime_implemented_generates_once_and_uses_cache(tmp_path):
    provider = FakeProvider(
        implementations={
            "add": "def add(x, y):\n    return x + y\n",
        }
    )
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    add = make_runtime_add(engine)

    assert add(2, 3) == 5
    assert add(10, -3) == 7
    assert provider.calls == [("implement", add.__qualname__, "gpt-5-mini")]

    cached_provider = FakeProvider()
    cached_engine = RuntimeEngine(
        provider=cached_provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    cached_add = make_runtime_add(cached_engine)

    assert cached_add(4, 6) == 10
    assert cached_provider.calls == []


def test_self_healing_repairs_and_retries(tmp_path):
    provider = FakeProvider(
        repairs={
            "mean": "def mean(values):\n    return sum(values) / len(values)\n",
        }
    )
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    mean = make_broken_mean(engine)

    assert mean([2, 4, 6]) == 4
    assert provider.calls == [("repair", mean.__qualname__, "NameError", "gpt-5-mini")]


def test_self_healing_persists_fix_to_cache(tmp_path):
    provider = FakeProvider(
        repairs={
            "mean": "def mean(values):\n    return sum(values) / len(values)\n",
        }
    )
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    mean = make_broken_mean(engine)

    assert mean([1, 3]) == 2

    cached_provider = FakeProvider()
    cached_engine = RuntimeEngine(
        provider=cached_provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    cached_mean = make_broken_mean(cached_engine)

    assert cached_mean([5, 7]) == 6
    assert cached_provider.calls == []


def test_runtime_implemented_requires_placeholder_body(tmp_path):
    engine = RuntimeEngine(
        provider=FakeProvider(),
        config=RuntimeConfig(cache_dir=tmp_path),
    )

    with pytest.raises(ValueError):

        @engine.runtime_implemented
        def add(x, y):
            """Return the sum of two numbers."""
            return x + y


def test_nested_self_healing_escalates_original_error_to_outer_function(tmp_path):
    provider = EscalationProvider()
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    _, compute_total = make_nested_total(engine)

    assert compute_total(100, "DE") == 100
    assert provider.calls == [
        ("repair", "fetch_tax_rate", "KeyError", "gpt-5-mini"),
        ("repair", "compute_total", "KeyError", "gpt-5-mini"),
    ]
    assert len(provider.outer_requests) == 1
    assert "needs caller-level fallback" in provider.outer_requests[0].traceback_text


def test_runtime_implemented_method_uses_state_fields_and_mutates_object(tmp_path):
    provider = StatefulMethodProvider()
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    Wallet = make_wallet_class(engine)
    wallet = Wallet(10)

    assert wallet.deposit(5) == 15
    assert wallet.balance == 15
    assert wallet.status == "active"
    assert wallet.history == [("deposit", 5)]
    snapshot = provider.implementation_snapshots[0]
    assert snapshot.state_fields == ("balance", "status", "history")
    assert snapshot.state_summary == {
        "balance": "10",
        "history": "[]",
        "status": "'new'",
    }


def test_self_healing_method_uses_state_fields_and_repairs_stateful_logic(tmp_path):
    provider = StatefulMethodProvider()
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path),
    )
    Wallet = make_wallet_class(engine)
    wallet = Wallet(20)

    assert wallet.withdraw(7) == 13
    assert wallet.balance == 13
    assert wallet.status == "active"
    assert wallet.history == [("withdraw", 7)]
    snapshot = provider.repair_snapshots[0]
    assert snapshot.state_fields == ("balance", "status", "history")
    assert snapshot.state_summary == {
        "balance": "20",
        "history": "[]",
        "status": "'new'",
    }

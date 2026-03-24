import argparse
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import OpenAIProvider, RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


class DemoProvider(LLMProvider):
    def __init__(self):
        self.calls = []

    def implement_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("implement", name, model))
        print("[llm] implement {0} via {1}".format(name, model))
        if name == "slugify":
            return (
                "def slugify(text: str) -> str:\n"
                "    import re\n"
                "    text = text.lower().strip()\n"
                "    text = re.sub(r'[^a-z0-9]+', '-', text)\n"
                "    return text.strip('-')\n"
            )
        if name == "normalize_tag":
            return (
                "def normalize_tag(text: str) -> str:\n"
                "    return slug(text)\n"
            )
        if name == "deposit":
            return (
                "def deposit(self, amount):\n"
                "    if amount < 0:\n"
                "        raise ValueError('amount must be non-negative')\n"
                "    self.balance += amount\n"
                "    self.status = 'active'\n"
                "    self.history.append(('deposit', amount))\n"
                "    return self.balance\n"
            )
        raise KeyError("No demo implementation for {0}".format(name))

    def repair_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("repair", name, request.error_type, model))
        print("[llm] repair {0} after {1} via {2}".format(name, request.error_type, model))
        if name == "mean":
            return (
                "def mean(values):\n"
                "    if not values:\n"
                "        raise ValueError('values must not be empty')\n"
                "    return sum(values) / len(values)\n"
            )
        if name == "normalize_tag":
            return (
                "def normalize_tag(text: str) -> str:\n"
                "    import re\n"
                "    text = text.lower().strip()\n"
                "    text = re.sub(r'[^a-z0-9]+', '-', text)\n"
                "    return text.strip('-')\n"
            )
        if name == "fetch_tax_rate":
            print("[llm] fetch_tax_rate cannot be repaired locally; reraising to caller")
            raise RuntimeError("needs caller-level fallback")
        if name == "compute_total":
            return (
                "def compute_total(subtotal, country):\n"
                "    rates = {'AT': 0.2, 'US': 0.0}\n"
                "    rate = rates.get(country, 0.0)\n"
                "    return subtotal * (1 + rate)\n"
            )
        if name == "withdraw":
            return (
                "def withdraw(self, amount):\n"
                "    if amount < 0:\n"
                "        raise ValueError('amount must be non-negative')\n"
                "    if amount > self.balance:\n"
                "        raise ValueError('insufficient funds')\n"
                "    self.balance -= amount\n"
                "    self.status = 'active'\n"
                "    self.history.append(('withdraw', amount))\n"
                "    return self.balance\n"
            )
        raise KeyError("No demo repair for {0}".format(name))


def build_engine(provider_name: str, cache_dir: Path):
    config = RuntimeConfig(cache_dir=cache_dir, max_heal_attempts=1)
    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        provider = OpenAIProvider()
    else:
        provider = DemoProvider()
    return RuntimeEngine(provider=provider, config=config), provider


def make_slugify(engine: RuntimeEngine):
    @engine.runtime_implemented
    def slugify(text: str) -> str:
        """Convert text into a lowercase ASCII slug separated by hyphens."""
        raise NotImplementedError

    return slugify


def make_mean(engine: RuntimeEngine):
    @engine.self_healing
    def mean(values):
        """Return the arithmetic mean of a non-empty iterable of numbers."""
        return sum(values) / len(value)

    return mean


def make_normalize_tag(engine: RuntimeEngine):
    @engine.runtime_implemented
    def normalize_tag(text: str) -> str:
        """Normalize a free-form tag into a lowercase slug with hyphen separators."""
        raise NotImplementedError

    return normalize_tag


def make_fetch_tax_rate(engine: RuntimeEngine):
    @engine.self_healing
    def fetch_tax_rate(country):
        """Return the tax rate for a country code."""
        rates = {"AT": 0.2, "US": 0.0}
        return rates[country]

    return fetch_tax_rate


def make_compute_total(engine: RuntimeEngine, fetch_tax_rate):
    @engine.self_healing
    def compute_total(subtotal, country):
        """Return subtotal with country-specific tax; fall back to subtotal if the country rate is unavailable."""
        return subtotal * (1 + fetch_tax_rate(country))

    return compute_total


def make_wallet_class(engine: RuntimeEngine):
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
            self.status = "active"
            self.history.append(("withdraw", amount))
            return self.balance

    return Wallet


def run_demo(provider_name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="paithon-demo-") as tmp_dir:
        cache_dir = Path(tmp_dir)
        engine, provider = build_engine(provider_name, cache_dir)

        print("PAIthon demo")
        print("provider =", provider_name)
        print("cache_dir =", cache_dir)

        print("\n1. Runtime implementation")
        slugify = make_slugify(engine)
        print("slugify('Hello, AI Runtime!') ->", slugify("Hello, AI Runtime!"))
        print("slugify('Already cached') ->", slugify("Already cached"))

        print("\n2. Self-healing an existing buggy function")
        mean = make_mean(engine)
        print("mean([2, 4, 6]) ->", mean([2, 4, 6]))

        print("\n3. Runtime-generated code that also heals on failure")
        normalize_tag = make_normalize_tag(engine)
        print("normalize_tag('Docs / V2 / Final') ->", normalize_tag("Docs / V2 / Final"))

        print("\n4. Nested self-healing escalation")
        if provider_name == "fake":
            fetch_tax_rate = make_fetch_tax_rate(engine)
            compute_total = make_compute_total(engine, fetch_tax_rate)
            print("compute_total(100, 'DE') ->", compute_total(100, "DE"))
            print("The inner fetch_tax_rate repair failed locally and the outer compute_total healed itself.")
        else:
            print("Skipped for --provider openai: this scenario is deterministic only with the fake demo provider.")

        print("\n5. Stateful methods with state_fields")
        if provider_name == "fake":
            Wallet = make_wallet_class(engine)
            wallet = Wallet(40)
            print("initial wallet state ->", wallet.__dict__)
            print("wallet.deposit(10) ->", wallet.deposit(10))
            print("wallet.withdraw(15) ->", wallet.withdraw(15))
            print("final wallet state ->", wallet.__dict__)
        else:
            print("Available with --provider openai too, but skipped in the deterministic demo path.")

        print("\n6. Cache reuse across a fresh engine")
        cached_engine, cached_provider = build_engine(provider_name, cache_dir)
        cached_slugify = make_slugify(cached_engine)
        cached_mean = make_mean(cached_engine)
        cached_normalize_tag = make_normalize_tag(cached_engine)
        cached_fetch_tax_rate = make_fetch_tax_rate(cached_engine)
        cached_compute_total = make_compute_total(cached_engine, cached_fetch_tax_rate)
        CachedWallet = make_wallet_class(cached_engine)
        cached_wallet = CachedWallet(12)
        print("cached_slugify('Fresh engine') ->", cached_slugify("Fresh engine"))
        print("cached_mean([10, 20]) ->", cached_mean([10, 20]))
        print("cached_normalize_tag('Cache Hit') ->", cached_normalize_tag("Cache Hit"))
        if provider_name == "fake":
            print("cached_compute_total(50, 'DE') ->", cached_compute_total(50, "DE"))
            print("cached_wallet.deposit(8) ->", cached_wallet.deposit(8))
            print("cached wallet state ->", cached_wallet.__dict__)

        if isinstance(provider, DemoProvider):
            print("\nInitial engine LLM calls:")
            for call in provider.calls:
                print(" ", call)
        if isinstance(cached_provider, DemoProvider):
            print("\nFresh engine LLM calls:")
            if cached_provider.calls:
                for call in cached_provider.calls:
                    print(" ", call)
            else:
                print("  none; all functions loaded from cache")


def main():
    parser = argparse.ArgumentParser(description="Run the PAIthon demo.")
    parser.add_argument(
        "--provider",
        choices=("fake", "openai"),
        default="fake",
        help="Use the deterministic fake provider or the real OpenAI provider.",
    )
    args = parser.parse_args()
    run_demo(args.provider)


if __name__ == "__main__":
    main()

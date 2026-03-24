import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import StateMutationError
from paithon.provider import LLMProvider


@dataclass
class Address:
    city: str
    zip_code: str


class AdvancedDemoProvider(LLMProvider):
    def __init__(self):
        self.calls = []

    def implement_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("implement", name, model))
        print("[llm] implement {0} via {1}".format(name, model))
        if name == "activate":
            return (
                "def activate(self):\n"
                "    self.status = 'active'\n"
                "    return self.status\n"
            )
        if name == "render_profile":
            return (
                "def render_profile(self):\n"
                "    return '{0}:{1}'.format(self.owner, self.address.city)\n"
            )
        raise KeyError(name)

    def repair_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("repair", name, request.error_type, model))
        print("[llm] repair {0} after {1} via {2}".format(name, request.error_type, model))
        if name == "withdraw":
            return (
                "def withdraw(self, amount):\n"
                "    self.balance -= amount\n"
                "    return self.balance\n"
            )
        raise KeyError(name)


def run_demo():
    with tempfile.TemporaryDirectory(prefix="paithon-advanced-") as tmp_dir:
        cache_dir = Path(tmp_dir) / "cache"
        review_dir = Path(tmp_dir) / "review"
        provider = AdvancedDemoProvider()
        engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=cache_dir))
        engine.register_state_serializer(
            Address,
            lambda value: ({"city": value.city, "zip": value.zip_code}, "address:{0}".format(value.city)),
            first=True,
        )

        print("PAIthon advanced state/policy demo")
        print("cache_dir =", cache_dir)

        class Wallet:
            __paithon_version__ = "2025-03"

            def __init__(self):
                self.owner = "Ada"
                self.address = Address("Vienna", "1010")
                self.balance = 20
                self.status = "new"

            @engine.runtime_implemented(
                state_fields=["owner", "address"],
                mutable_state_fields=(),
                cache_by_class=True,
                contract_revision="r1",
            )
            def render_profile(self):
                """Return a short profile string for this wallet owner."""
                raise NotImplementedError

            @engine.self_healing(
                state_fields=["balance", "status"],
                mutable_state_fields=["balance"],
                rollback_on_failure=True,
                cache_by_class=True,
                contract_revision="r1",
            )
            def withdraw(self, amount):
                """Decrease balance by amount and return the new balance."""
                self.status = "broken"
                self.balance -= amunt
                return self.balance

            @engine.runtime_implemented(
                state_fields=["status"],
                mutable_state_fields=(),
                rollback_on_failure=True,
            )
            def activate(self):
                """Return the active status."""
                raise NotImplementedError

        wallet = Wallet()

        print("\n1. Custom state serializer + OOP-aware cache context")
        print("render_profile() ->", wallet.render_profile())

        print("\n2. Rollback before healed retry")
        print("wallet state before withdraw ->", wallet.__dict__)
        print("withdraw(5) ->", wallet.withdraw(5))
        print("wallet state after withdraw ->", wallet.__dict__)

        print("\n3. Mutation allowlist enforcement")
        try:
            wallet.activate()
        except StateMutationError as exc:
            print("activate() blocked ->", exc)
            print("wallet state after blocked mutation ->", wallet.__dict__)

        print("\n4. Review export")
        manifest_path = engine.export_review_artifacts(review_dir)
        print("review manifest ->", manifest_path)
        print("exported review files ->", sorted(path.name for path in review_dir.iterdir()))

        print("\nLLM calls:")
        for call in provider.calls:
            print(" ", call)


if __name__ == "__main__":
    run_demo()

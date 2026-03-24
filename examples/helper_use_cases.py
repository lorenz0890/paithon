import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


class HelperDemoProvider(LLMProvider):
    def __init__(self):
        self.calls = []

    def implement_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("implement", name, model))
        print("[llm] implement {0} via {1}".format(name, model))
        if name == "summarize_scores":
            return (
                "def summarize_scores(values):\n"
                "    return {\n"
                "        'min': min(values),\n"
                "        'max': max(values),\n"
                "        'avg': sum(values) / len(values),\n"
                "    }\n"
            )
        if name == "parse_inventory_record":
            return (
                "def parse_inventory_record(payload):\n"
                "    import json\n"
                "    if isinstance(payload, str):\n"
                "        payload = json.loads(payload)\n"
                "    return {\n"
                "        'sku': str(payload['sku']).strip().upper(),\n"
                "        'count': int(payload['count']),\n"
                "        'active': bool(payload['active']),\n"
                "    }\n"
            )
        if name == "dump_json":
            return (
                "def dump_json(value):\n"
                "    import json\n"
                "    return json.dumps(value, sort_keys=True, separators=(',', ':'))\n"
            )
        raise KeyError(name)

    def repair_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("repair", name, request.error_type, model))
        print("[llm] repair {0} after {1} via {2}".format(name, request.error_type, model))
        if name == "extract_display_name":
            return (
                "def extract_display_name(payload):\n"
                "    if 'user' in payload:\n"
                "        return payload['user']['profile']['display_name'].strip()\n"
                "    return payload['data']['user']['name'].strip()\n"
            )
        raise KeyError(name)


def run_demo():
    with tempfile.TemporaryDirectory(prefix="paithon-helpers-") as tmp_dir:
        engine = RuntimeEngine(
            provider=HelperDemoProvider(),
            config=RuntimeConfig(cache_dir=Path(tmp_dir)),
        )
        provider = engine.provider

        print("PAIthon helper use cases")
        print("cache_dir =", tmp_dir)

        print("\n1. Contract-driven plugin function")
        summarize_scores = engine.create_function(
            "summarize_scores",
            "(values)",
            "Return a dict with min, max, and avg for a non-empty iterable of numbers.",
        )
        print("summarize_scores([2, 4, 6]) ->", summarize_scores([2, 4, 6]))

        print("\n2. Schema adapter from a parser contract")
        @engine.schema_adapter(schema="Input keys: sku:str, count:int, active:bool. Output keys: sku, count, active.")
        def parse_inventory_record(payload):
            """Parse a raw inventory payload into a normalized record."""
            raise NotImplementedError

        print(
            "parse_inventory_record('{\"sku\":\" ab-1 \",\"count\":\"7\",\"active\":1}') ->",
            parse_inventory_record('{"sku":" ab-1 ","count":"7","active":1}'),
        )

        print("\n3. Response adapter for API shape drift")
        @engine.response_adapter
        def extract_display_name(payload):
            """Return the display name from an external user payload."""
            return payload["user"]["profile"]["display_name"].strip()

        print(
            "extract_display_name({'data': {'user': {'name': 'Ada Lovelace '}}}) ->",
            extract_display_name({"data": {"user": {"name": "Ada Lovelace "}}}),
        )

        print("\n4. Runtime polyfill for a missing dependency")
        @engine.polyfill(dependency="paithon_missing_fastjson")
        def dump_json(value):
            """Serialize a value into a deterministic JSON string."""
            import paithon_missing_fastjson  # pragma: no cover

            return paithon_missing_fastjson.dumps(value)

        print("dump_json({'b': 1, 'a': 2}) ->", dump_json({"b": 1, "a": 2}))

        print("\nLLM calls:")
        for call in provider.calls:
            print(" ", call)


if __name__ == "__main__":
    run_demo()

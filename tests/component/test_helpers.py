import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


class HelperProvider(LLMProvider):
    def __init__(self):
        self.implement_requests = []
        self.repair_requests = []

    def implement_function(self, request, model):
        self.implement_requests.append((request, model))
        name = request.snapshot.name
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
        self.repair_requests.append((request, model))
        name = request.snapshot.name
        if name == "extract_display_name":
            return (
                "def extract_display_name(payload):\n"
                "    if 'user' in payload:\n"
                "        return payload['user']['profile']['display_name'].strip()\n"
                "    return payload['data']['user']['name'].strip()\n"
            )
        raise KeyError(name)


def test_create_function_supports_contract_driven_plugins(tmp_path):
    provider = HelperProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    summarize_scores = engine.create_function(
        "summarize_scores",
        "(values)",
        "Return a dict with min, max, and avg for a non-empty iterable of numbers.",
    )

    assert summarize_scores([2, 4, 6]) == {"min": 2, "max": 6, "avg": 4.0}
    request, model = provider.implement_requests[0]
    assert model == "gpt-5-mini"
    assert request.snapshot.name == "summarize_scores"
    assert request.snapshot.contract == "Return a dict with min, max, and avg for a non-empty iterable of numbers."


def test_schema_adapter_augments_contract_and_generates_parser(tmp_path):
    provider = HelperProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    @engine.schema_adapter(schema="Input keys: sku:str, count:int, active:bool. Output keys: sku, count, active.")
    def parse_inventory_record(payload):
        """Parse a raw inventory payload into a normalized record."""
        raise NotImplementedError

    assert parse_inventory_record({"sku": " ab-1 ", "count": "7", "active": 1}) == {
        "sku": "AB-1",
        "count": 7,
        "active": True,
    }
    request, _ = provider.implement_requests[0]
    assert "schema adapter / parser" in request.snapshot.contract
    assert "Input keys: sku:str, count:int, active:bool." in request.snapshot.contract


def test_response_adapter_repairs_external_response_shape_drift(tmp_path):
    provider = HelperProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    @engine.response_adapter
    def extract_display_name(payload):
        """Return the display name from an external user payload."""
        return payload["user"]["profile"]["display_name"].strip()

    payload = {"data": {"user": {"name": "Ada Lovelace "}}}
    assert extract_display_name(payload) == "Ada Lovelace"
    request, _ = provider.repair_requests[0]
    assert request.error_type == "KeyError"
    assert "external API responses" in request.snapshot.contract


def test_response_adapter_only_heals_shape_errors_by_default(tmp_path):
    provider = HelperProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    @engine.response_adapter
    def explode(payload):
        """Return a derived ratio from a payload."""
        return 1 / 0

    with pytest.raises(ZeroDivisionError):
        explode({})
    assert provider.repair_requests == []


def test_polyfill_generates_fallback_when_dependency_is_missing(tmp_path):
    provider = HelperProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    @engine.polyfill(dependency="paithon_missing_fastjson")
    def dump_json(value):
        """Serialize a value into a deterministic JSON string."""
        import paithon_missing_fastjson  # pragma: no cover

        return paithon_missing_fastjson.dumps(value)

    assert dump_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    request, _ = provider.implement_requests[0]
    assert "runtime polyfill" in request.snapshot.contract
    assert "paithon_missing_fastjson" in request.snapshot.contract


def test_polyfill_uses_local_implementation_when_dependency_exists(tmp_path):
    provider = HelperProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    @engine.polyfill(dependency="json")
    def dump_json(value):
        """Serialize a value into JSON using the local dependency."""
        import json

        return json.dumps(value, sort_keys=True)

    assert dump_json({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert provider.implement_requests == []
    assert provider.repair_requests == []

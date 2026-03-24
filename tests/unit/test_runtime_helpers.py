from paithon import RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


class NullProvider(LLMProvider):
    def implement_function(self, request, model):
        raise AssertionError("unexpected implement_function call")

    def repair_function(self, request, model):
        raise AssertionError("unexpected repair_function call")


def build_engine(tmp_path, **config_overrides):
    config = RuntimeConfig(cache_dir=tmp_path, **config_overrides)
    return RuntimeEngine(provider=NullProvider(), config=config)


def test_artifact_slug_sanitizes_non_path_characters(tmp_path):
    engine = build_engine(tmp_path)
    payload = {
        "module": "demo/module",
        "qualname": "Box.<locals>.slugify",
        "mode": "implement",
    }

    slug = engine._artifact_slug(payload)

    assert slug == "demo_module__Box._locals_.slugify__implement"


def test_build_review_patch_returns_fallback_when_no_diff_exists(tmp_path):
    engine = build_engine(tmp_path)
    payload = {
        "module": "demo",
        "qualname": "slugify",
        "template_source": "def slugify(text):\n    return text.lower()\n",
        "source": "def slugify(text):\n    return text.lower()\n",
        "source_path": None,
    }

    assert engine._build_review_patch(payload) == "# No diff available\n"


def test_indent_function_source_respects_target_indentation(tmp_path):
    engine = build_engine(tmp_path)

    indented = engine._indent_function_source("def slug(text):\n    return text.lower()\n", "    ")

    assert indented == "    def slug(text):\n        return text.lower()\n"


def test_compile_template_function_preserves_globals_defaults_and_module(tmp_path):
    engine = build_engine(tmp_path)
    globals_dict = {"__name__": "demo_mod", "PREFIX": "pre-"}
    function = engine._compile_template_function(
        "def slug(text, suffix='!'):\n    return PREFIX + text + suffix\n",
        "slug",
        globals_dict,
    )

    assert function("ada") == "pre-ada!"
    assert function.__defaults__ == ("!",)
    assert function.__module__ == "demo_mod"


def test_make_placeholder_clone_preserves_metadata(tmp_path):
    engine = build_engine(tmp_path)

    def render(value: int, suffix="!"):
        """Original doc."""
        return "{0}{1}".format(value, suffix)

    render.__dict__["marker"] = "x"
    clone = engine._make_placeholder_clone(render, "Return a rendered label.")

    assert clone.__module__ == render.__module__
    assert clone.__qualname__ == render.__qualname__
    assert clone.__annotations__ == {"value": int}
    assert clone.__defaults__ == ("!",)
    assert clone.marker == "x"
    assert "Return a rendered label." in clone.__paithon_source__


def test_allows_cache_payload_is_strict_only_in_production_locked(tmp_path):
    review_engine = build_engine(tmp_path, operating_mode="review_first")
    locked_engine = build_engine(tmp_path / "locked", operating_mode="production_locked")

    assert review_engine._allows_cache_payload({"approval_status": "pending_review"}) is True
    assert locked_engine._allows_cache_payload({"approval_status": "approved"}) is True
    assert locked_engine._allows_cache_payload({"approval_status": "promoted"}) is True
    assert locked_engine._allows_cache_payload({"approval_status": "pending_review"}) is False
    assert locked_engine._allows_cache_payload({}) is False


def test_find_qualname_node_locates_nested_class_method(tmp_path):
    engine = build_engine(tmp_path)
    source = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def render(self):\n"
        "            return 'ok'\n"
    )

    import ast

    tree = ast.parse(source)
    node = engine._find_qualname_node(tree, ["Outer", "Inner", "render"])

    assert node is not None
    assert node.name == "render"


def test_normalize_review_action_understands_aliases(tmp_path):
    engine = build_engine(tmp_path)

    assert engine._normalize_review_action("a") == "approve"
    assert engine._normalize_review_action("promote") == "promote"
    assert engine._normalize_review_action("s") == "skip"
    assert engine._normalize_review_action("q") == "quit"
    assert engine._normalize_review_action("q", allow_quit=False) is None


def test_validate_execution_mode_rejects_unknown_value(tmp_path):
    engine = build_engine(tmp_path)

    try:
        engine._validate_execution_mode("not-a-mode")
    except ValueError as exc:
        assert "unsupported execution_mode" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("expected ValueError for unsupported execution_mode")

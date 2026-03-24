from paithon.cache import CodeCache


def test_code_cache_load_returns_none_for_missing_entry(tmp_path):
    cache = CodeCache(tmp_path)

    assert cache.load("missing") is None


def test_code_cache_save_and_load_round_trip(tmp_path):
    cache = CodeCache(tmp_path)
    payload = {"value": 1, "name": "demo"}

    cache.save("entry", payload)

    assert cache.load("entry") == payload
    assert not (tmp_path / "entry.tmp").exists()


def test_code_cache_save_overwrites_existing_entry(tmp_path):
    cache = CodeCache(tmp_path)

    cache.save("entry", {"value": 1})
    cache.save("entry", {"value": 2, "status": "updated"})

    assert cache.load("entry") == {"value": 2, "status": "updated"}


def test_code_cache_load_returns_none_for_corrupted_json(tmp_path):
    cache = CodeCache(tmp_path)
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")

    assert cache.load("broken") is None

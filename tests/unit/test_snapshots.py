from paithon.snapshots import SNAPSHOT_MISSING, SnapshotStrategyRegistry, deep_equal


class FakeStateful:
    def __init__(self, value, training=True):
        self.value = value
        self.training = training

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, payload):
        self.value = payload["value"]

    def train(self, mode=True):
        self.training = bool(mode)
        return self


class Counter:
    def __init__(self, count):
        self.count = count


class FakeTensor:
    __module__ = "torch.fake"

    def __init__(self, values, *, device="cpu"):
        self._values = list(values)
        self.device = device
        self.dtype = "float32"
        self.shape = (len(self._values),)

    def clone(self):
        return FakeTensor(self._values, device=self.device)

    def detach(self):
        return self

    def cpu(self):
        return FakeTensor(self._values, device="cpu")

    def equal(self, other):
        return self._values == other._values

    def numel(self):
        return len(self._values)

    def tolist(self):
        return list(self._values)

    def to(self, device):
        return FakeTensor(self._values, device=device)

    def copy_(self, other):
        self._values = list(other._values)
        return self


class FakeNumpyArray:
    __module__ = "numpy.fake"

    def __init__(self, values):
        self._values = list(values)
        self.shape = (len(self._values),)
        self.dtype = "float64"
        self.size = len(self._values)

    def copy(self):
        return FakeNumpyArray(self._values)

    def __setitem__(self, key, value):
        if key is Ellipsis:
            self._values = list(value._values)
            return
        raise TypeError(key)

    def __eq__(self, other):
        return FakeNumpyBoolResult(self._values == other._values)

    def tolist(self):
        return list(self._values)


class FakeNumpyBoolResult:
    def __init__(self, value):
        self._value = bool(value)

    def all(self):
        return self._value


class FakeDataFrame:
    __module__ = "pandas.core.frame"

    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.columns = list(self._rows[0].keys()) if self._rows else []
        self.shape = (len(self._rows), len(self.columns))
        self.dtypes = {column: type(self._rows[0][column]).__name__ for column in self.columns} if self._rows else {}

    def copy(self, deep=True):
        return FakeDataFrame(self._rows)

    def equals(self, other):
        return self._rows == other._rows

    def head(self, limit):
        return FakeDataFrame(self._rows[:limit])

    def to_dict(self, orient="records"):
        assert orient == "records"
        return [dict(row) for row in self._rows]


class FakeSeries:
    __module__ = "pandas.core.series"

    def __init__(self, values, *, name="score"):
        self._values = list(values)
        self.name = name
        self.shape = (len(self._values),)
        self.dtype = "int64"

    def copy(self, deep=True):
        return FakeSeries(self._values, name=self.name)

    def equals(self, other):
        return self._values == other._values and self.name == other.name

    def head(self, limit):
        return FakeSeries(self._values[:limit], name=self.name)

    def tolist(self):
        return list(self._values)


class FakeKerasModel:
    __module__ = "keras.engine.training"

    def __init__(self, weights, *, trainable=True):
        self._weights = list(weights)
        self.trainable = trainable

    def get_weights(self):
        return list(self._weights)

    def set_weights(self, weights):
        self._weights = list(weights)


class FakeStateAccessor:
    def __init__(self, state):
        self._state = dict(state)

    def get_state(self):
        return dict(self._state)

    def set_state(self, state):
        self._state = dict(state)


def test_snapshot_registry_uses_state_dict_strategy_and_restores_training_flag():
    registry = SnapshotStrategyRegistry()
    model = FakeStateful(10, training=False)

    snapshot = registry.capture(model)
    model.value = 99
    model.training = True
    restored = registry.restore(model, snapshot)

    assert snapshot.strategy_name == "state_dict"
    assert restored is model
    assert model.value == 10
    assert model.training is False
    assert registry.matches(model, snapshot) is True


def test_snapshot_registry_copy_strategy_returns_deep_copy():
    registry = SnapshotStrategyRegistry()
    payload = {"items": [1, 2]}

    snapshot = registry.capture(payload)
    payload["items"].append(3)
    restored = registry.restore(payload, snapshot)

    assert snapshot.strategy_name == "copy"
    assert restored == {"items": [1, 2]}
    assert restored is not payload
    assert restored["items"] is not payload["items"]


def test_snapshot_registry_custom_strategy_can_take_precedence():
    registry = SnapshotStrategyRegistry()
    registry.register(
        Counter,
        capture=lambda value: ({"count": value.count}, {"count": value.count}),
        restore=lambda current, snapshot: Counter(snapshot.restore_payload["count"]),
        compare=lambda current, snapshot: current.count == snapshot.compare_payload["count"],
        name="counter",
        first=True,
    )

    snapshot = registry.capture(Counter(4))

    assert snapshot.strategy_name == "counter"
    assert registry.matches(Counter(4), snapshot) is True
    assert registry.matches(Counter(5), snapshot) is False


def test_deep_equal_handles_nested_python_structures_and_missing_sentinel():
    assert deep_equal({"a": [1, 2], "b": {"x": 3}}, {"a": [1, 2], "b": {"x": 3}}) is True
    assert deep_equal({"a": [1, 2]}, {"a": [1, 3]}) is False
    assert deep_equal({1, 2}, {2, 1}) is True
    assert deep_equal(SNAPSHOT_MISSING, SNAPSHOT_MISSING) is True
    assert deep_equal(SNAPSHOT_MISSING, None) is False


def test_snapshot_registry_torch_tensor_strategy_restores_and_diffs():
    registry = SnapshotStrategyRegistry()
    tensor = FakeTensor([1.0, 2.0])

    snapshot = registry.capture(tensor)
    tensor.copy_(FakeTensor([9.0, 10.0]))
    diff = registry.diff(tensor, snapshot, max_items=4, max_chars=200, max_depth=2)
    restored = registry.restore(tensor, snapshot)
    before_tensor = next(value for value in diff["before"].values() if isinstance(value, dict) and "values" in value)
    after_tensor = next(value for value in diff["after"].values() if isinstance(value, dict) and "values" in value)

    assert snapshot.strategy_name == "torch_tensor"
    assert before_tensor["values"] == [1.0, 2.0]
    assert after_tensor["values"] == [9.0, 10.0]
    assert restored is tensor
    assert tensor.tolist() == [1.0, 2.0]


def test_snapshot_registry_numpy_array_strategy_restores_and_diffs():
    registry = SnapshotStrategyRegistry()
    array = FakeNumpyArray([1.0, 2.0, 3.0])

    snapshot = registry.capture(array)
    array[...] = FakeNumpyArray([5.0, 6.0, 7.0])
    diff = registry.diff(array, snapshot, max_items=4, max_chars=200, max_depth=2)
    restored = registry.restore(array, snapshot)

    assert snapshot.strategy_name == "numpy_array"
    assert diff["before"]["values"] == [1.0, 2.0, 3.0]
    assert diff["after"]["values"] == [5.0, 6.0, 7.0]
    assert restored is array
    assert array.tolist() == [1.0, 2.0, 3.0]


def test_snapshot_registry_pandas_dataframe_strategy_restores_and_reports_details():
    registry = SnapshotStrategyRegistry()
    frame = FakeDataFrame([{"name": "Ada", "score": 9}])

    snapshot = registry.capture(frame)
    frame._rows[0]["score"] = 10
    diff = registry.diff(frame, snapshot, max_items=4, max_chars=200, max_depth=2)
    restored = registry.restore(frame, snapshot)

    assert snapshot.strategy_name == "pandas_dataframe"
    assert diff["details"]["kind"] == "dataframe"
    assert diff["before"]["preview"] == [{"name": "Ada", "score": 9}]
    assert diff["after"]["preview"] == [{"name": "Ada", "score": 10}]
    assert restored is not frame
    assert restored._rows == [{"name": "Ada", "score": 9}]


def test_snapshot_registry_pandas_series_strategy_restores_and_reports_details():
    registry = SnapshotStrategyRegistry()
    series = FakeSeries([1, 2, 3])

    snapshot = registry.capture(series)
    series._values[1] = 7
    diff = registry.diff(series, snapshot, max_items=4, max_chars=200, max_depth=2)
    restored = registry.restore(series, snapshot)

    assert snapshot.strategy_name == "pandas_series"
    assert diff["details"]["kind"] == "series"
    assert diff["before"]["values"] == [1, 2, 3]
    assert diff["after"]["values"] == [1, 7, 3]
    assert restored._values == [1, 2, 3]


def test_snapshot_registry_keras_weights_strategy_restores_and_reports_details():
    registry = SnapshotStrategyRegistry()
    model = FakeKerasModel([1.0, 2.0], trainable=True)

    snapshot = registry.capture(model)
    model.set_weights([3.0, 4.0])
    model.trainable = False
    diff = registry.diff(model, snapshot, max_items=4, max_chars=200, max_depth=2)
    restored = registry.restore(model, snapshot)

    assert snapshot.strategy_name == "keras_weights"
    assert diff["details"]["kind"] == "keras_weights"
    assert diff["details"]["changed_weight_indices"] == [0, 1]
    assert "keras weights changed" in diff["summary"]
    assert restored is model
    assert model.get_weights() == [1.0, 2.0]
    assert model.trainable is True


def test_snapshot_registry_state_accessor_strategy_restores_state():
    registry = SnapshotStrategyRegistry()
    stateful = FakeStateAccessor({"seed": 7})

    snapshot = registry.capture(stateful)
    stateful.set_state({"seed": 99})
    restored = registry.restore(stateful, snapshot)

    assert snapshot.strategy_name == "state_accessor"
    assert restored is stateful
    assert stateful.get_state() == {"seed": 7}


def test_describe_diff_reports_mapping_sequence_and_set_details():
    registry = SnapshotStrategyRegistry()

    mapping_snapshot = registry.capture({"status": "new", "tags": ["a"]})
    mapping_diff = registry.diff({"status": "active", "tags": ["a"], "extra": True}, mapping_snapshot)
    set_snapshot = registry.capture({"a", "b"})
    set_diff = registry.diff({"b", "c"}, set_snapshot)
    sequence_snapshot = registry.capture(["a", "b", "c"])
    sequence_diff = registry.diff(["a", "x", "c", "d"], sequence_snapshot)
    string_snapshot = registry.capture("abc")
    string_diff = registry.diff("axcd", string_snapshot)

    assert mapping_diff["details"]["kind"] == "mapping"
    assert "'extra'" in mapping_diff["details"]["added_keys"]
    assert "'status'" in mapping_diff["details"]["changed_keys"]
    assert "mapping changed" in mapping_diff["summary"]
    assert mapping_diff["details"]["changed"]["'status'"]["before"] == "new"
    assert set_diff["details"]["kind"] == "set"
    assert "'c'" in set_diff["details"]["added_items"]
    assert "'a'" in set_diff["details"]["removed_items"]
    assert sequence_diff["details"]["kind"] == "sequence"
    assert sequence_diff["details"]["first_changed_index"] == 1
    assert "sequence length 3 -> 4" in sequence_diff["summary"]
    assert string_diff["details"]["kind"] == "string"
    assert string_diff["details"]["first_changed_index"] == 1
    assert "string length 3 -> 4" in string_diff["summary"]


def test_numeric_diff_details_include_delta_stats():
    registry = SnapshotStrategyRegistry()
    tensor = FakeTensor([1.0, 2.0, 3.0])

    snapshot = registry.capture(tensor)
    tensor.copy_(FakeTensor([1.5, 2.0, 10.0]))
    diff = registry.diff(tensor, snapshot, max_items=4, max_chars=200, max_depth=2)

    assert diff["details"]["kind"] == "tensor"
    assert diff["details"]["changed_elements"] == 2
    assert diff["details"]["max_abs_delta"] == 7.0
    assert diff["details"]["sample"][0]["index"] == 0

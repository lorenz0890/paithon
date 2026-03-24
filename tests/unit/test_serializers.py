from dataclasses import dataclass

from paithon.context import safe_repr
from paithon.serializers import REDACTED_TEXT, StateSerializerRegistry


@dataclass
class Profile:
    name: str
    score: int


class Thing:
    def __init__(self, value):
        self.value = value


class FakeField:
    def __init__(self, name, *, many_to_many=False):
        self.name = name
        self.attname = name
        self.many_to_many = many_to_many


class FakeMeta:
    def get_fields(self):
        return [FakeField("identifier"), FakeField("name")]


class FakeDjangoModel:
    _meta = FakeMeta()

    def __init__(self):
        self.identifier = 7
        self.name = "Ada"


class FakeRelated:
    def __init__(self, identifier):
        self.id = identifier


class FakeColumnAttr:
    def __init__(self, key):
        self.key = key


class FakeRelationship:
    def __init__(self, key):
        self.key = key


class FakeMapper:
    column_attrs = [FakeColumnAttr("identifier"), FakeColumnAttr("status")]
    relationships = [FakeRelationship("owner")]


class FakeSqlAlchemyModel:
    __mapper__ = FakeMapper()

    def __init__(self):
        self.identifier = 9
        self.status = "ready"
        self.owner = FakeRelated(12)


class FakeManager:
    def __init__(self, values):
        self._values = list(values)

    def all(self):
        return list(self._values)


class FakeDjangoMetaWithRelations:
    concrete_fields = [FakeField("identifier"), FakeField("name")]
    many_to_many = [FakeField("groups", many_to_many=True)]

    def get_fields(self):
        return self.concrete_fields + self.many_to_many


class FakeDjangoModelWithRelations:
    _meta = FakeDjangoMetaWithRelations()

    def __init__(self):
        self.identifier = 11
        self.name = "Ada"
        self.groups = FakeManager([Thing("ops"), Thing("ml")])


class FakeTensor:
    __module__ = "torch.fake"

    def __init__(self, values):
        self._values = list(values)
        self.shape = (len(self._values),)
        self.dtype = "float32"
        self.device = "cpu"

    def numel(self):
        return len(self._values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class FakeQuerySet:
    __module__ = "django.db.models.query"

    def __init__(self, items):
        self._items = list(items)
        self.model = Thing

    def count(self):
        return len(self._items)

    def __getitem__(self, item):
        return self._items[item]

    def __iter__(self):
        return iter(self._items)


class FakeBind:
    def __init__(self, url):
        self.url = url


class FakeSqlAlchemySession:
    __module__ = "sqlalchemy.orm.session"

    def __init__(self):
        self.identity_map = {1: "row"}
        self.new = [1, 2]
        self.dirty = [3]
        self.deleted = []
        self.bind = FakeBind("sqlite:///demo.db")


class FakeDataFrame:
    __module__ = "pandas.core.frame"

    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.columns = list(self._rows[0].keys()) if self._rows else []
        self.shape = (len(self._rows), len(self.columns))
        self.dtypes = {column: type(self._rows[0][column]).__name__ for column in self.columns} if self._rows else {}

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

    def head(self, limit):
        return FakeSeries(self._values[:limit], name=self.name)

    def tolist(self):
        return list(self._values)


class FakeHttpRequest:
    def __init__(self, method):
        self.method = method


class FakeHttpResponse:
    __module__ = "requests.models"

    def __init__(self):
        self.status_code = 200
        self.url = "https://api.example.com/users"
        self.request = FakeHttpRequest("GET")
        self.headers = {"content-type": "application/json", "authorization": "Bearer secret"}

    def json(self):
        return {"user": {"id": 7, "token": "abc"}}


class FakeSklearnEstimator:
    __module__ = "sklearn.linear_model"

    def __init__(self):
        self.n_features_in_ = 3
        self.classes_ = ["a", "b"]

    def get_params(self, deep=False):
        assert deep is False
        return {"alpha": 0.1, "fit_intercept": True}


class FakeKerasModel:
    __module__ = "keras.engine.training"

    def __init__(self):
        self.name = "demo"
        self.trainable = True
        self.built = True

    def get_config(self):
        return {"units": 16, "activation": "relu"}

    def count_params(self):
        return 128


def test_state_serializer_serializes_dataclasses_with_expected_schema():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(Profile("Ada", 9), safe_repr=safe_repr, max_chars=200)

    assert serialized.text == "{'name': 'Ada', 'score': 9}"
    assert serialized.schema == "dataclass:Profile(name, score)"


def test_state_serializer_custom_registration_can_take_precedence():
    registry = StateSerializerRegistry()
    registry.register(Thing, lambda value: ({"custom": value.value}, "thing:custom"), first=True)

    serialized = registry.serialize(Thing("yes"), safe_repr=safe_repr, max_chars=200)

    assert serialized.text == "{'custom': 'yes'}"
    assert serialized.schema == "thing:custom"


def test_state_serializer_truncates_nested_sequences_and_marks_cycles():
    registry = StateSerializerRegistry()
    payload = {"values": [1, 2, 3, 4]}
    payload["self"] = payload

    serialized = registry.serialize(
        payload,
        safe_repr=safe_repr,
        max_chars=400,
        max_depth=3,
        max_items=2,
    )

    assert "<2 more items; total=4>" in serialized.text
    assert "<cycle:" in serialized.text
    assert serialized.schema == "builtins.dict"


def test_state_serializer_recognizes_django_like_models():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeDjangoModel(), safe_repr=safe_repr, max_chars=200)

    assert serialized.text == "{'identifier': 7, 'name': 'Ada'}"
    assert serialized.schema == "django:FakeDjangoModel(identifier, name)"


def test_state_serializer_recognizes_torch_tensor_like_values():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeTensor([1.0, 2.0]), safe_repr=safe_repr, max_chars=200)

    assert "'values': [1.0, 2.0]" in serialized.text
    assert "'device': 'cpu'" in serialized.text
    assert serialized.schema == "tensor:FakeTensor"


def test_state_serializer_recognizes_sqlalchemy_relationships():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeSqlAlchemyModel(), safe_repr=safe_repr, max_chars=200)

    assert "'owner': {'id': 12}" in serialized.text
    assert serialized.schema == "sqlalchemy:FakeSqlAlchemyModel(identifier, status, owner[rel])"


def test_state_serializer_recognizes_django_many_to_many_relations():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeDjangoModelWithRelations(), safe_repr=safe_repr, max_chars=300)

    assert "'groups': {'count': 2, 'items': [{'type': '__main__.Thing'}, {'type': '__main__.Thing'}]}" not in serialized.text
    assert "'groups': {'count': 2" in serialized.text
    assert "groups[m2m]" in serialized.schema


def test_state_serializer_respects_per_collection_limits():
    registry = StateSerializerRegistry()
    payload = {
        "mapping": {"a": 1, "b": 2, "c": 3},
        "sequence": [1, 2, 3],
        "tags": {"x", "y", "z"},
    }

    serialized = registry.serialize(
        payload,
        safe_repr=safe_repr,
        max_chars=400,
        max_depth=3,
        max_items=5,
        max_mapping_items=2,
        max_sequence_items=1,
        max_set_items=1,
    )

    assert "<truncated_items>" in serialized.text
    assert "<2 more items; total=3>" in serialized.text


def test_state_serializer_marks_depth_truncation_with_schema():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(
        {"profile": {"details": {"city": "Vienna"}}},
        safe_repr=safe_repr,
        max_chars=200,
        max_depth=1,
        max_items=4,
    )

    assert "<truncated_depth:builtins.dict>" in serialized.text


def test_state_serializer_redacts_matching_keys():
    registry = StateSerializerRegistry()
    payload = {"token": "secret-value", "profile": {"password_hash": "abc", "name": "Ada"}}

    serialized = registry.serialize(
        payload,
        safe_repr=safe_repr,
        max_chars=300,
        max_depth=3,
        max_items=4,
        redacted_field_names=("token",),
        redacted_field_patterns=("password",),
    )

    assert REDACTED_TEXT in serialized.text
    assert "secret-value" not in serialized.text
    assert "abc" not in serialized.text


def test_state_serializer_recognizes_pandas_dataframe_like_values():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeDataFrame([{"name": "Ada", "score": 9}]), safe_repr=safe_repr, max_chars=300)

    assert "'shape': (1, 2)" in serialized.text
    assert "'columns': ['name', 'score']" in serialized.text
    assert serialized.schema == "dataframe:FakeDataFrame"


def test_state_serializer_recognizes_pandas_series_like_values():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeSeries([1, 2, 3]), safe_repr=safe_repr, max_chars=300)

    assert "'name': 'score'" in serialized.text
    assert "'values': [1, 2, 3]" in serialized.text
    assert serialized.schema == "series:FakeSeries"


def test_state_serializer_recognizes_django_queryset_like_values():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeQuerySet([Thing("ops"), Thing("ml")]), safe_repr=safe_repr, max_chars=300)

    assert "'count': 2" in serialized.text
    assert "Thing" in serialized.text
    assert serialized.schema == "queryset:Thing"


def test_state_serializer_recognizes_sqlalchemy_session_like_values():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeSqlAlchemySession(), safe_repr=safe_repr, max_chars=300)

    assert "'identity_count': 1" in serialized.text
    assert "'new_count': 2" in serialized.text
    assert "'bind': 'sqlite:///demo.db'" in serialized.text
    assert serialized.schema == "sqlalchemy-session:FakeSqlAlchemySession"


def test_state_serializer_redacts_matching_nested_paths():
    registry = StateSerializerRegistry()
    payload = {"profile": {"credentials": {"token": "abc", "password": "pw"}, "name": "Ada"}}

    serialized = registry.serialize(
        payload,
        safe_repr=safe_repr,
        max_chars=400,
        max_depth=4,
        max_items=6,
        redacted_field_paths=("profile.credentials.token", "profile.credentials.password"),
    )

    assert REDACTED_TEXT in serialized.text
    assert "abc" not in serialized.text
    assert "pw" not in serialized.text
    assert "Ada" in serialized.text


def test_state_serializer_supports_path_wildcards_and_custom_placeholder():
    registry = StateSerializerRegistry()
    payload = {"headers": {"authorization": "Bearer secret", "x-request-id": "abc"}}

    serialized = registry.serialize(
        payload,
        safe_repr=safe_repr,
        max_chars=300,
        max_depth=3,
        max_items=4,
        redacted_field_paths=("headers.*",),
        redaction_placeholder="<hidden>",
    )

    assert "<hidden>" in serialized.text
    assert "Bearer secret" not in serialized.text
    assert "abc" not in serialized.text


def test_state_serializer_recognizes_http_response_like_values():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(
        FakeHttpResponse(),
        safe_repr=safe_repr,
        max_chars=400,
        redacted_field_paths=("headers.authorization", "body_preview.user.token"),
    )

    assert "'status_code': 200" in serialized.text
    assert "'method': 'GET'" in serialized.text
    assert REDACTED_TEXT in serialized.text
    assert serialized.schema == "http-response:FakeHttpResponse"


def test_state_serializer_recognizes_sklearn_estimators():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeSklearnEstimator(), safe_repr=safe_repr, max_chars=300)

    assert "'alpha': 0.1" in serialized.text
    assert "'n_features_in_': 3" in serialized.text
    assert serialized.schema == "sklearn:FakeSklearnEstimator"


def test_state_serializer_recognizes_keras_models():
    registry = StateSerializerRegistry()

    serialized = registry.serialize(FakeKerasModel(), safe_repr=safe_repr, max_chars=300)

    assert "'param_count': 128" in serialized.text
    assert "'config_keys': ['units', 'activation']" in serialized.text
    assert serialized.schema == "keras:FakeKerasModel"

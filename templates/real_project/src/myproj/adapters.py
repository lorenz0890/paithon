from .paithon_runtime import engine


@engine.runtime_implemented
def normalize_customer_record(payload: dict) -> dict:
    """Normalize a raw customer payload into {'id': str, 'email': str | None, 'country': str | None, 'is_active': bool}.

    Accept common input drift such as integer ids, upper-case keys, string booleans, and missing optional fields.
    """
    raise NotImplementedError


@engine.schema_adapter
def parse_feature_flag(payload: dict) -> dict:
    """Normalize a feature-flag payload into {'name': str, 'enabled': bool, 'rollout': int | None}."""
    raise NotImplementedError

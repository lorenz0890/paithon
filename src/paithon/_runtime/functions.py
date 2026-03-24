from .engine import RuntimeEngine


_default_engine = None


def _get_default_engine() -> RuntimeEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = RuntimeEngine()
    return _default_engine


def self_healing(func=None, **kwargs):
    return _get_default_engine().self_healing(func, **kwargs)


def runtime_implemented(func=None, **kwargs):
    return _get_default_engine().runtime_implemented(func, **kwargs)


def schema_adapter(func=None, **kwargs):
    return _get_default_engine().schema_adapter(func, **kwargs)


def response_adapter(func=None, **kwargs):
    return _get_default_engine().response_adapter(func, **kwargs)


def polyfill(func=None, **kwargs):
    return _get_default_engine().polyfill(func, **kwargs)


def create_function(*args, **kwargs):
    return _get_default_engine().create_function(*args, **kwargs)


def self_writing(func=None, **kwargs):
    return runtime_implemented(func, **kwargs)


def selfhealing(func=None, **kwargs):
    return self_healing(func, **kwargs)

"""Public runtime compatibility layer.

The implementation lives in the internal ``paithon._runtime`` package so the
runtime can be organized into smaller modules without changing public imports.
"""

from ._runtime import (
    RuntimeEngine,
    create_function,
    polyfill,
    response_adapter,
    runtime_implemented,
    schema_adapter,
    self_healing,
    self_writing,
    selfhealing,
)

__all__ = [
    "RuntimeEngine",
    "create_function",
    "polyfill",
    "response_adapter",
    "runtime_implemented",
    "schema_adapter",
    "self_healing",
    "self_writing",
    "selfhealing",
]

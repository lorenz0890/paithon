import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from ..snapshots import ValueSnapshot


@dataclass
class _VariantState:
    key: str
    context: Dict[str, Any]
    current: Callable[..., Any]
    source: str
    initialized: bool = False


@dataclass
class _StateSnapshot:
    target: Any
    values: Dict[str, ValueSnapshot]
    existed: Dict[str, bool]
    cleanup_extras: bool = False


@dataclass
class _ManagedFunction:
    template: Callable[..., Any]
    contract: str
    contract_revision: Optional[str]
    state_fields: Tuple[str, ...]
    mutable_state_fields: Optional[Tuple[str, ...]]
    rollback_fields: Tuple[str, ...]
    rollback_on_failure: bool
    strict_rollback: bool
    cache_by_class: bool
    mode: str
    template_source: str
    source_path: Optional[str]
    source_lineno: Optional[int]
    variants: Dict[str, _VariantState] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

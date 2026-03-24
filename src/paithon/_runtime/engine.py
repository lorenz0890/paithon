from typing import Any, Callable, Optional

from ..cache import CodeCache
from ..models import RuntimeConfig
from ..provider import OpenAIProvider
from ..serializers import StateSerializerRegistry
from ..snapshots import SnapshotStrategyRegistry, ValueSnapshot
from .decorators import RuntimeDecoratorMixin
from .execution import RuntimeExecutionMixin
from .policy import RuntimePolicyMixin
from .review import RuntimeReviewMixin
from .source import RuntimeSourceMixin
from .state import RuntimeStateMixin


class RuntimeEngine(
    RuntimeReviewMixin,
    RuntimeDecoratorMixin,
    RuntimeExecutionMixin,
    RuntimeStateMixin,
    RuntimeSourceMixin,
    RuntimePolicyMixin,
):
    def __init__(self, provider=None, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self._validate_operating_mode(self.config.operating_mode)
        self._validate_execution_mode(self.config.execution_mode)
        self.provider = provider or OpenAIProvider()
        self.cache = CodeCache(self.config.cache_dir)
        self.state_serializers = StateSerializerRegistry()
        self.snapshot_strategies = SnapshotStrategyRegistry()
        self.audit_log_path = self.config.audit_log_path or (self.config.cache_dir / "audit.jsonl")

    def register_state_serializer(
        self,
        target,
        serializer: Callable[[Any], Any],
        *,
        schema_name: Optional[str] = None,
        first: bool = False,
    ) -> None:
        self.state_serializers.register(target, serializer, schema_name=schema_name, first=first)

    def register_snapshot_strategy(
        self,
        target,
        capture: Callable[[Any], Any],
        restore: Callable[[Any, ValueSnapshot], Any],
        compare: Callable[[Any, ValueSnapshot], bool],
        *,
        name: str,
        first: bool = False,
    ) -> None:
        self.snapshot_strategies.register(
            target,
            capture,
            restore,
            compare,
            name=name,
            first=first,
        )

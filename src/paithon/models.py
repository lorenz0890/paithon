from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RuntimeConfig:
    model: str = "gpt-5-mini"
    cache_dir: Path = Path(".paithon_cache")
    max_heal_attempts: int = 1
    max_value_chars: int = 200
    max_state_depth: int = 2
    max_collection_items: int = 8
    max_mapping_items: Optional[int] = None
    max_sequence_items: Optional[int] = None
    max_set_items: Optional[int] = None
    redacted_field_names: Tuple[str, ...] = ()
    redacted_field_patterns: Tuple[str, ...] = ()
    redacted_field_paths: Tuple[str, ...] = ()
    redaction_placeholder: str = "<redacted>"
    operating_mode: str = "review_first"
    audit_log_path: Optional[Path] = None
    execution_mode: str = "in_process"
    sandbox_timeout_seconds: float = 3.0
    sandbox_python_executable: Optional[str] = None
    sandbox_memory_limit_mb: Optional[int] = 256
    sandbox_file_size_limit_bytes: Optional[int] = 1_048_576
    sandbox_max_open_files: Optional[int] = 32
    sandbox_working_dir: Optional[Path] = None


@dataclass(frozen=True)
class FunctionSnapshot:
    module: str
    qualname: str
    name: str
    signature: str
    contract: str
    source: str
    state_fields: Tuple[str, ...]
    state_summary: Dict[str, str]
    state_schema: Dict[str, str]
    globals_summary: Dict[str, str]
    closure_summary: Dict[str, str]


@dataclass(frozen=True)
class ImplementationRequest:
    snapshot: FunctionSnapshot


@dataclass(frozen=True)
class RepairRequest:
    snapshot: FunctionSnapshot
    error_type: str
    error_message: str
    traceback_text: str
    call_summary: Dict[str, str]

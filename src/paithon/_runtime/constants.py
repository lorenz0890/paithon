BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input"}
BLOCKED_IMPORTS = {"ctypes", "marshal", "multiprocessing", "os", "pickle", "socket", "subprocess"}
IMPLEMENT_DECORATOR_NAMES = {"runtime_implemented", "self_writing", "schema_adapter", "polyfill"}
APPROVED_CACHE_STATUSES = {"approved", "promoted"}
OPERATING_MODES = {"development", "review_first", "production_locked"}
LEGACY_APPROVAL_STATUS = "legacy_untracked"
EXECUTION_MODES = {"in_process", "subprocess_probe", "subprocess_restricted"}

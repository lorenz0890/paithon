import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .constants import APPROVED_CACHE_STATUSES, EXECUTION_MODES, LEGACY_APPROVAL_STATUS, OPERATING_MODES


class RuntimePolicyMixin:
    @staticmethod
    def _validate_operating_mode(mode: str) -> None:
        if mode not in OPERATING_MODES:
            raise ValueError("unsupported operating_mode: {0}".format(mode))

    @staticmethod
    def _validate_execution_mode(mode: str) -> None:
        if mode not in EXECUTION_MODES:
            raise ValueError("unsupported execution_mode: {0}".format(mode))

    def _allows_runtime_generation(self) -> bool:
        return self.config.operating_mode in {"development", "review_first"}

    def _allows_runtime_healing(self) -> bool:
        return self.config.operating_mode in {"development", "review_first"}

    def _allows_cache_payload(self, payload: Dict[str, Any]) -> bool:
        if self.config.operating_mode != "production_locked":
            return True
        return self._approval_status(payload) in APPROVED_CACHE_STATUSES

    def _default_approval_status(self) -> str:
        if self.config.operating_mode == "development":
            return "development"
        if self.config.operating_mode == "review_first":
            return "pending_review"
        return "approved"

    def _approval_status(self, payload: Dict[str, Any]) -> str:
        return payload.get("approval_status") or LEGACY_APPROVAL_STATUS

    def _normalize_cache_payload(self, payload: Dict[str, Any], *, cache_key: Optional[str] = None) -> Dict[str, Any]:
        normalized = dict(payload)
        if not normalized.get("approval_status"):
            normalized["approval_status"] = LEGACY_APPROVAL_STATUS
            normalized["operating_mode"] = normalized.get("operating_mode") or "legacy"
            if cache_key is not None:
                self.cache.save(cache_key, normalized)
                self._audit(
                    "cache_metadata_normalized",
                    cache_key=cache_key,
                    qualname=normalized.get("qualname"),
                    approval_status=LEGACY_APPROVAL_STATUS,
                )
        return normalized

    def _audit(self, event: str, **fields) -> None:
        record = {
            "time": self._utcnow(),
            "event": event,
            "operating_mode": self.config.operating_mode,
        }
        record.update(fields)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).isoformat()

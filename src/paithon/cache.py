import json
from pathlib import Path
from typing import Any, Dict, Optional


class CodeCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.root / "{0}.json".format(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None

    def save(self, key: str, payload: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "{0}.json".format(key)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)

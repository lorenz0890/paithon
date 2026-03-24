import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from myproj.paithon_runtime import engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve all exported PAIthon review artifacts in the manifest.")
    parser.add_argument("--review-dir", type=Path, default=ROOT / ".paithon_review")
    parser.add_argument("--reviewer", default="manual")
    args = parser.parse_args()

    manifest_path = args.review_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        payload = engine.approve_cache_entry(entry["cache_key"], reviewer=args.reviewer)
        print(entry["qualname"], "->", payload["approval_status"])


if __name__ == "__main__":
    main()

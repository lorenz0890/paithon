import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from myproj.paithon_runtime import engine


def main() -> None:
    review_dir = ROOT / ".paithon_review"
    manifest_path = engine.export_review_artifacts(review_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("manifest =", manifest_path)
    for entry in manifest:
        print(entry["qualname"])
        print("  approval_status =", entry["approval_status"])
        print("  source_file =", entry["source_file"])
        print("  patch_file =", entry["patch_file"])


if __name__ == "__main__":
    main()

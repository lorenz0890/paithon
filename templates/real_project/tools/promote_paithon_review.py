import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from myproj.paithon_runtime import engine


def main() -> None:
    review_dir = ROOT / ".paithon_review"
    promoted = engine.promote_review_artifacts(review_dir)
    for cache_key, path in promoted.items():
        print(cache_key, "->", path)


if __name__ == "__main__":
    main()

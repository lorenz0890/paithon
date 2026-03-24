import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from myproj.paithon_runtime import engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive review loop over exported PAIthon artifacts.")
    parser.add_argument("--review-dir", type=Path, default=ROOT / ".paithon_review")
    parser.add_argument("--reviewer", default="manual")
    args = parser.parse_args()

    results = engine.interactive_review(args.review_dir, reviewer=args.reviewer)
    for cache_key, action in results.items():
        print(cache_key, "->", action)


if __name__ == "__main__":
    main()

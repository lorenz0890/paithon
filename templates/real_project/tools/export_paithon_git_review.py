import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from myproj.paithon_runtime import engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Git-ready review bundle for PAIthon artifacts.")
    parser.add_argument("--review-dir", type=Path, default=ROOT / ".paithon_review")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".paithon_git_review")
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--branch-name")
    args = parser.parse_args()

    bundle_path = engine.export_git_review_bundle(
        args.output_dir,
        args.review_dir,
        base_ref=args.base_ref,
        branch_name=args.branch_name,
    )
    print("bundle =", bundle_path)


if __name__ == "__main__":
    main()

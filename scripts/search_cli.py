"""CLI: search an indexed video with a natural-language query.

Example:
    python scripts/search_cli.py sample "a person riding a bicycle"
    python scripts/search_cli.py sample "people cooking" --top-k 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from video_search.search import VideoIndex  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Search an indexed video.")
    p.add_argument("index_name", help="Video stem (folder name under data/index/).")
    p.add_argument("query", help="Natural-language description of the event.")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    args = p.parse_args()

    index_path = config.INDEX_DIR / args.index_name
    if not index_path.exists():
        raise SystemExit(f"No index found at {index_path}. Run index_video.py first.")

    idx = VideoIndex(index_path)
    results = idx.search(args.query, top_k=args.top_k)

    print(f'\nQuery: "{args.query}"  ({idx.meta["video_name"]})\n')
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. {r.hms()}  (t={r.timestamp:7.2f}s)   score={r.score:.4f}")


if __name__ == "__main__":
    main()

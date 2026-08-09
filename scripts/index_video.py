"""CLI: build a searchable index for a video.

Example:
    python scripts/index_video.py data/videos/sample.mp4
    python scripts/index_video.py data/videos/sample.mp4 --every 0.5
"""
import argparse
import sys
from pathlib import Path

# Make `src/` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from video_search.indexer import build_index  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Index a video for semantic search.")
    p.add_argument("video", help="Path to the video file.")
    p.add_argument("--every", type=float, default=config.SAMPLE_EVERY_SEC,
                   help="Sample one frame every N seconds (default: %(default)s).")
    args = p.parse_args()

    out = build_index(
        video_path=args.video,
        index_dir=config.INDEX_DIR,
        model_name=config.CLIP_MODEL,
        pretrained=config.CLIP_PRETRAINED,
        sample_every_sec=args.every,
    )
    print(f"\nIndex written to: {out}")


if __name__ == "__main__":
    main()

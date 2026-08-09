"""CLI: run detection + tracking on a video and write its artifacts.

    python scripts/process_video.py data/videos/sample.mp4
    python scripts/process_video.py data/videos/sample.mp4 --id mytest

Outputs land in data/processed/<id>/ (detections.json, intervals.csv, summary.json).
Useful for testing the pipeline without launching the web UI.
"""
import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from video_search import pipeline, storage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path, help="path to the input video")
    ap.add_argument("--id", default=None, help="video id (defaults to a random id)")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"No such file: {args.video}")
    video_id = args.id or uuid.uuid4().hex[:12]

    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(done / total * 100) if total else 0
        if pct != last[0]:
            last[0] = pct
            print(f"\r  tracking… {pct:3d}%  ({done}/{total} frames)", end="", flush=True)

    print(f"Processing {args.video.name}  ->  id={video_id}")
    summary = pipeline.process_video(args.video, video_id, progress_cb=progress)
    print()

    print(f"\nIndexed {summary['num_intervals']} appearances "
          f"across {len(summary['classes'])} classes:")
    for cls, info in summary["classes"].items():
        print(f"  {cls:14s} {info['count']:3d} object(s), {info['appearances']} appearance(s)")
    print(f"\nArtifacts: {storage.processed_dir(video_id)}")


if __name__ == "__main__":
    main()

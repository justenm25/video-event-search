"""Build and persist a searchable embedding index for a video."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .frames import iter_frames
from .model import encode_images


def build_index(
    video_path: str | Path,
    index_dir: str | Path,
    model_name: str,
    pretrained: str,
    sample_every_sec: float = 1.0,
    batch_size: int = 32,
) -> Path:
    """Sample frames from a video, embed them with CLIP, and save the index.

    Produces two files under `index_dir/<video_stem>/`:
      - embeddings.npy : float32 array (n_frames, dim)
      - meta.json      : timestamps + source metadata
    """
    video_path = Path(video_path)
    out_dir = Path(index_dir) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps: list[float] = []
    embeddings: list[np.ndarray] = []

    buffer_imgs, buffer_ts = [], []

    def flush():
        if not buffer_imgs:
            return
        embeddings.append(encode_images(buffer_imgs, model_name, pretrained))
        timestamps.extend(buffer_ts)
        buffer_imgs.clear()
        buffer_ts.clear()

    for frame in tqdm(iter_frames(video_path, sample_every_sec), desc="Embedding frames"):
        buffer_imgs.append(frame.image)
        buffer_ts.append(frame.timestamp)
        if len(buffer_imgs) >= batch_size:
            flush()
    flush()

    if not embeddings:
        raise RuntimeError(f"No frames extracted from {video_path}")

    emb = np.concatenate(embeddings, axis=0)
    np.save(out_dir / "embeddings.npy", emb)

    meta = {
        "video": str(video_path.resolve()),
        "video_name": video_path.name,
        "n_frames": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "sample_every_sec": sample_every_sec,
        "model": model_name,
        "pretrained": pretrained,
        "timestamps": timestamps,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir

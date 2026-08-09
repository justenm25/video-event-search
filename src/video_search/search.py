"""Query a saved video index with a natural-language description."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .model import encode_text


@dataclass
class SearchResult:
    timestamp: float  # seconds into the video
    score: float      # cosine similarity in [-1, 1]

    def hms(self) -> str:
        s = int(self.timestamp)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class VideoIndex:
    """Loads a single video's embeddings + metadata and answers text queries."""

    def __init__(self, index_path: str | Path):
        index_path = Path(index_path)
        self.embeddings = np.load(index_path / "embeddings.npy")
        with open(index_path / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.timestamps = self.meta["timestamps"]
        self.model_name = self.meta["model"]
        self.pretrained = self.meta["pretrained"]

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Return the top_k frames most similar to the text query."""
        q = encode_text(query, self.model_name, self.pretrained)  # (d,)
        scores = self.embeddings @ q  # cosine sim (both are L2-normalized)
        k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:k]
        return [
            SearchResult(timestamp=float(self.timestamps[i]), score=float(scores[i]))
            for i in top_idx
        ]

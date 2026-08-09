"""Load a CLIP model and encode images / text into the shared embedding space."""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import torch
from PIL import Image

import open_clip


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model(model_name: str, pretrained: str):
    """Load and cache the CLIP model, its preprocessing transform, and tokenizer."""
    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_images(images: List[Image.Image], model_name: str, pretrained: str) -> np.ndarray:
    """Encode a list of PIL images into L2-normalized embeddings (n, d)."""
    model, preprocess, _, device = load_model(model_name, pretrained)
    batch = torch.stack([preprocess(img) for img in images]).to(device)
    feats = _normalize(model.encode_image(batch))
    return feats.cpu().numpy().astype("float32")


@torch.no_grad()
def encode_text(query: str, model_name: str, pretrained: str) -> np.ndarray:
    """Encode a text query into a single L2-normalized embedding (d,)."""
    model, _, tokenizer, device = load_model(model_name, pretrained)
    tokens = tokenizer([query]).to(device)
    feats = _normalize(model.encode_text(tokens))
    return feats.cpu().numpy().astype("float32")[0]

"""Central configuration for the Intelligent Video Event Search Engine."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
INDEX_DIR = DATA_DIR / "index"

# --- Frame sampling ---
# Extract one frame every N seconds. Lower = finer temporal resolution, slower.
SAMPLE_EVERY_SEC = 1.0

# --- CLIP model ---
# open_clip model + pretrained weights. ViT-B-32 is a fast, solid default.
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# --- Search ---
TOP_K = 5  # default number of results to return

# Intelligent Video Event Search Engine

CS 5330: Pattern Recognition and Computer Vision

**Team:** Nikhil Singh Shekhawat, Nimish Vivek Poonekar

## What this is
A system that lets you search *inside* videos with natural-language queries
(e.g. "find moments where people are cooking") and returns the matching
timestamps / clips — instead of relying on filenames, tags, or manual descriptions.

## How it works (baseline: CLIP)
1. **Sample frames** from a video every N seconds (`src/video_search/frames.py`).
2. **Embed** each frame with a CLIP image encoder into a shared vision-language space (`model.py`).
3. **Index**: store the frame embeddings + their timestamps (`indexer.py`).
4. **Search**: embed the text query with the CLIP text encoder and rank frames by
   cosine similarity (`search.py`). Return the top timestamps.

Because CLIP maps images and text into the *same* space, a text query like
"a person riding a bicycle" can be compared directly against frame embeddings —
no per-video training required.

## Project layout
```
config.py                 # paths, model name, sampling rate, top-k
src/video_search/
  frames.py               # frame extraction (OpenCV)
  model.py                # CLIP load + encode_images / encode_text
  indexer.py              # build & save an index for a video
  search.py               # VideoIndex: load index, answer text queries
scripts/
  index_video.py          # CLI: index a video
  search_cli.py           # CLI: query an indexed video
data/
  videos/                 # put your input videos here
  index/                  # generated embeddings + metadata
```

## Setup

> Your machine has an **RTX 5060 (Blackwell)** GPU. It needs a recent CUDA build
> of PyTorch (`cu128`). Install PyTorch *first*, separately from the rest.

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell

# 2. Install PyTorch with the Blackwell-compatible CUDA build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install the rest
pip install -r requirements.txt
```

Verify the GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Usage

```bash
# 1. Drop a video into data/videos/, e.g. data/videos/sample.mp4

# 2. Build the index (samples 1 frame/sec by default)
python scripts/index_video.py data/videos/sample.mp4
#   ...or finer sampling:
python scripts/index_video.py data/videos/sample.mp4 --every 0.5

# 3. Search it
python scripts/search_cli.py sample "a person riding a bicycle"
python scripts/search_cli.py sample "people cooking" --top-k 3
```

Output is a ranked list of timestamps (HH:MM:SS) with similarity scores.

## Roadmap
- [ ] Baseline CLIP frame retrieval  *(current)*
- [ ] Group adjacent high-scoring frames into **clips/segments** (not just frames)
- [ ] Search across **multiple videos** at once
- [ ] Approximate nearest-neighbor index (FAISS) for scale
- [ ] Simple **Streamlit UI** to type a query and preview matching frames/clips
- [ ] Evaluation on a labeled dataset (e.g. a subset of ActivityNet / Kinetics)

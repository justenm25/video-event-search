"""Central configuration for the Intelligent Video Event Search Engine.

The system detects objects (COCO-80 classes) in a video, tracks each one across
frames to give it a persistent ID, and folds those tracks into time intervals:
one row per object appearance, with a start and end timestamp.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"        # uploaded / source videos
PROCESSED_DIR = DATA_DIR / "processed"  # per-video artifacts (detections + intervals)
IDENTITIES_DIR = DATA_DIR / "identities"  # enrolled people (face embeddings per name)

# --- Detection + tracking model (Ultralytics YOLO) ---
# yolo11n < yolo11s < yolo11m < yolo11l < yolo11x  (speed -> accuracy).
# Weights auto-download on first run. yolo11m is a strong accuracy/speed balance
# for the RTX 5060; bump to yolo11l/x for even sharper detections (slower).
YOLO_MODEL = "yolo11m.pt"

# Built-in ByteTrack tracker config shipped with Ultralytics.
TRACKER = "bytetrack.yaml"

# Inference resolution. Larger = better small/distant-object detection, slower.
# 960 noticeably improves recall over the default 640 on the 5060.
IMG_SIZE = 960

# Minimum detection confidence the *tracker* sees. Kept moderate so ByteTrack
# still has boxes to associate; the stricter per-class floors below do the real
# false-positive filtering after tracking.
CONF_THRESHOLD = 0.35

# IoU threshold for non-max suppression (lower = fewer overlapping duplicate boxes).
IOU_THRESHOLD = 0.5

# Process every Nth frame. 1 = every frame (best tracking + smoothest overlay).
# Bump to 2-3 to trade some tracking continuity for speed on long footage.
FRAME_STRIDE = 1

# --- Robustness: confidence floors + track hygiene ---
# A detection is only kept if its confidence clears the floor for its class. This
# is what curbs the "dog silhouette labeled person" type of error: back-lit / odd-
# angle guesses tend to be low-confidence, and we hold error-prone classes to a
# higher bar. Anything not listed uses CONF_FLOOR_DEFAULT.
CONF_FLOOR_DEFAULT = 0.40
CLASS_CONF_FLOORS = {
    "person": 0.50,   # silhouettes/animals get misread as people most often
}

# A track must be seen in at least this many kept frames before we trust it. Kills
# 2-3 frame "ghost" tracks (a momentary false detection that never recurs).
MIN_TRACK_DETECTIONS = 3

# --- Robustness: perception (low-light / back-light) ---
# The dog-in-silhouette failure is really a contrast problem. When a frame is dark
# we run CLAHE (adaptive histogram equalization) so the detector sees real edges
# and textures instead of a black blob. "Auto" = only enhance frames whose mean
# luma falls below ENHANCE_LUMA_THRESH, so well-lit footage is untouched.
ENHANCE_LOWLIGHT = True
ENHANCE_LUMA_THRESH = 90   # 0-255 mean brightness; below this -> enhance

# Test-time augmentation (multi-scale + flips) at inference. Noticeably more
# robust on hard poses/angles, but ~2-3x slower. On by request for accuracy runs.
USE_TTA = False

# --- Face recognition (custom identities) ---
# Enrolled people ("identities") relabel the generic COCO "person" class with a
# name. On enrollment we store one 512-d face embedding per uploaded photo
# (facenet-pytorch: MTCNN detector + InceptionResnetV1/VGGFace2). During analysis
# each *tracked* person is matched to the gallery and their whole track is renamed
# by majority vote. Set to False to disable and always keep the plain "person".
FACE_RECOGNITION = True

# Minimum cosine similarity (embeddings are L2-normalized, so this is a dot
# product in [-1, 1]) for a face crop to count as a match to an enrolled person.
# Higher = stricter (fewer false names); lower = more permissive. ~0.5-0.6 is a
# sensible range for VGGFace2 embeddings.
FACE_MATCH_THRESHOLD = 0.55

# Per person track, sample at most this many face crops (the largest, most
# confident boxes) to embed + vote on. Keeps face inference cost bounded — it
# runs a handful of times per person, never once per frame.
FACE_CROPS_PER_TRACK = 8

# A track needs at least this many crops that agree on the same name before it's
# relabeled. Guards against a single stray match renaming a whole appearance.
FACE_MIN_VOTES = 2

# Minimum MTCNN detection probability for a face to be embedded at all. Rejects
# blurry / partial / not-really-a-face crops during both enrollment and matching.
FACE_MIN_PROB = 0.92

# The best identity must beat the runner-up (a *different* name) by at least this
# cosine margin, else the match is treated as ambiguous and left as "person". Stops
# a stranger from being forced onto whichever enrolled face is nearest.
FACE_MATCH_MARGIN = 0.05

# --- Interval building ---
# When an object disappears, wait this many SECONDS of continuous absence before
# closing its interval. Absorbs brief occlusions / missed detections so a single
# car behind a pole doesn't get split into many rows. This is the "gap tolerance".
INTERVAL_GAP_SEC = 0.6

# Drop appearances shorter than this (seconds) as noise.
MIN_INTERVAL_SEC = 0.3

# --- Motion analytics (trajectories / heatmap / line crossings) ---
# Always computed from the tracks (cheap, no extra model). ZONE_LINE_FRAC places a
# vertical reference line at this fraction of the frame width; we count how many
# objects cross it and in which direction (people/traffic counting).
ZONE_LINE_FRAC = 0.5

# --- Segmentation (optional pixel masks) ---
# When True the main detector becomes the YOLO segmentation variant, so every
# detection also carries a polygon mask (drawn as an outline in the HUD). Same 80
# classes and tracking; slightly slower + larger detections.json.
SEGMENTATION = False
SEG_MODEL = "yolo11m-seg.pt"

# --- Pose + action recognition (optional) ---
# When True a YOLO pose model runs alongside detection; its 17 keypoints are
# matched to each person track and reduced to a coarse action (raising hand /
# fallen / sitting / standing). Adds a second inference per frame.
POSE_ENABLED = False
POSE_MODEL = "yolo11m-pose.pt"

# --- Monocular depth (optional, MiDaS) ---
# When True, estimate a relative depth map (MiDaS small via torch.hub) and tag
# each track with its relative distance (0=far .. 1=near). To keep cost bounded,
# depth runs on only DEPTH_SAMPLES frames sampled across the video, not every
# frame. A colorized depth preview of one frame is saved for the HUD.
DEPTH_ENABLED = False
DEPTH_SAMPLES = 8

# --- Appearance Re-ID (optional) ---
# Recognize a person by body appearance (clothing/build), not just their face.
# We embed each person track's crops with a ResNet-50 feature extractor; a track
# that was named by face becomes an appearance reference, and an *unnamed* person
# track whose appearance matches it (cosine >= REID_THRESHOLD) inherits the name.
# This carries an identity across a broken track / an occlusion / a turned-away
# stretch where the face is never visible.
REID_ENABLED = False
REID_THRESHOLD = 0.82

# --- Face attributes (optional, InsightFace age/gender) ---
# Estimate age + gender per person track from their face (InsightFace buffalo_l).
ATTRIBUTES_ENABLED = False

# --- Server ---
HOST = "127.0.0.1"
PORT = 8000

# Preprocess

Scripts for turning raw dashcam videos into a labelled dataset of street-light
pole crops. The output is `crops/crops_index.csv` plus per-crop JPEG files,
ready for classifier training.

## Files

| File | Purpose |
|------|---------|
| `extract_frames.py` | Decode videos at 1 fps, tag each frame with city and lighting condition |
| `dedup_frames.py` | Drop near-duplicate consecutive frames via perceptual hashing |
| `detect_poles.py` | Run YOLO-World to detect poles on each frame |
| `visualize_detections.py` | Render detection boxes for visual quality checks |
| `select_main_poles.py` | Filter and rank detected poles per frame (keep up to 3) |
| `generate_crops.py` | Cut native-resolution square crops centred on each pole |
| `triage_label.py` | Keyboard-based labelling tool (`ok` / `problem` / `unclear` / `not_pole`) |
| `route_changes.csv` | City + GPS lookup, indexed by start time |
| `run_full_pipeline.sh` | Convenience script that chains the steps below |
| `yolov8s-worldv2.pt` | YOLO-World checkpoint (auto-downloaded on first run) |

## Setup

```bash
pip install -r requirements.txt
ffmpeg -version    # must be installed system-wide
```

## Step-by-step

Each step reads the previous step's CSV and writes a new one. Frame and crop
files are kept on disk; soft-delete only.

```bash
# 1. Extract frames (skip videos entirely in daylight)
python extract_frames.py \
    --input ./videos --output ./frames \
    --route ./route_changes.csv \
    --skip-daylight-videos \
    --fps 1 --width 1280 \
    --csv ./frames_index.csv

# 2. Deduplicate consecutive near-identical frames
python dedup_frames.py \
    --input frames_index.csv \
    --output frames_index_dedup.csv \
    --threshold 10

# 3. Detect poles (YOLO-World, low confidence to maximize recall)
python detect_poles.py \
    --input frames_index_dedup.csv \
    --output frames_index_with_poles.csv \
    --conf 0.18 --imgsz 640

# 4. (optional) Inspect detections visually
python visualize_detections.py --mode random --n 20

# 5. Keep up to 3 well-formed poles per frame
python select_main_poles.py \
    --input frames_index_with_poles.csv \
    --output frames_index_selected.csv \
    --min-conf 0.23 --max-crop-iou 0.3 --max-poles-per-frame 3

# 6. Generate native-resolution square crops (1.6× bbox side)
python generate_crops.py \
    --input frames_index_selected.csv \
    --output-dir ./crops \
    --expand-ratio 1.6

# 7. Manually label crops (keyboard: O/P/U/N/S, Backspace, Q)
python triage_label.py \
    --input ./crops/crops_index.csv \
    --filter-light twilight,night
```

To run all steps end-to-end:

```bash
./run_full_pipeline.sh
```

## Outputs

```
crops/
├── images/                       # one JPEG per pole, native resolution
│   ├── <video>_<frame>_pole1.jpg
│   └── ...
└── crops_index.csv               # crop metadata + bbox-in-crop coordinates + label
```

`crops_index.csv` is the file consumed by classifier training.

## Key parameters

| Parameter | Value | Notes |
|-----------|------:|-------|
| Frame rate | 1 fps | `extract_frames.py --fps` |
| Frame width | 1280 px | `extract_frames.py --width` |
| Dedup Hamming threshold | 10 | Higher = more aggressive dropping |
| Detector confidence | 0.18 | Low to maximize recall; filtered later |
| Min selection confidence | 0.23 | `select_main_poles.py --min-conf` |
| Max poles per frame | 3 | `select_main_poles.py --max-poles-per-frame` |
| Crop expand ratio | 1.6 | Crop side = `1.6 × max(bbox_w, bbox_h)` |

## `route_changes.csv`

Lookup table that maps each video's timestamp to a city and its coordinates.
The first row acts as a default fallback.

```csv
start_time,city,lat,lon
2000-01-01T00:00:00,Antalya,36.8969,30.7133
2025-08-19T00:00:00,Erzurum,39.9000,41.2700
2026-03-18T13:30:00,Mersin,36.8121,34.6415
```

Each row asserts: *"From `start_time` onward, recordings were made in this
city."* Coordinates are used to compute per-frame `daylight` / `twilight` /
`night` via `astral`.

## Labels

| Label | Meaning | Used for training? |
|-------|---------|:------------------:|
| `ok` | Lamp on the targeted pole is lit | ✅ |
| `problem` | Lamp is not lit (candidate fault) | ✅ |
| `unclear` | Cannot judge with confidence | — |
| `not_pole` | YOLO false positive (no pole) | — |
| `skipped` | Skipped during triage | — |

Only `ok` and `problem` crops are passed to the classifier; the rest are
retained in the CSV for reference.
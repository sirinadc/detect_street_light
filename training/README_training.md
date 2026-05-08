# Street Light Classifier — Training Pipeline

Three scripts, run in order:

1. **`prepare_dataset.py`** — Split crops into train / val / test (grouped by video)
2. **`train_classifier.py`** — Train the EfficientNet-B0 bbox-aware model
3. **`evaluate.py`** — Test-set evaluation and error analysis

## Setup

```bash
pip install -r requirements_training.txt
```

## Step-by-step

### 1. Prepare the dataset

```bash
cd ~/dash_out
python ~/micode/srnpolev2/prepare_dataset.py \
    --input ./crops/crops_index.csv \
    --output-dir ./splits
```

Outputs:

- `splits/train.csv` (~70%)
- `splits/val.csv` (~15%)
- `splits/test.csv` (~15%)

**Note:** All crops from a given source video are kept in the same split to
prevent data leakage from highly correlated frames.

### 2. Train the model

**On GPU (Colab T4 or local GPU):**

```bash
python ~/micode/srnpolev2/train_classifier.py \
    --splits-dir ./splits \
    --output-dir ./runs/exp_001 \
    --epochs 20 \
    --batch-size 32
```

Runtime: ~15–30 minutes.

**On CPU (local machine):**

```bash
python ~/micode/srnpolev2/train_classifier.py \
    --splits-dir ./splits \
    --output-dir ./runs/exp_001 \
    --epochs 15 \
    --batch-size 16 \
    --num-workers 2
```

Runtime: ~2–4 hours.

**Baseline comparison (RGB only, no bbox mask):**

```bash
python ~/micode/srnpolev2/train_classifier.py \
    --output-dir ./runs/baseline_no_mask \
    --no-mask
```

### 3. Evaluate on the test set

```bash
python ~/micode/srnpolev2/evaluate.py \
    --checkpoint ./runs/exp_001/best_model.pt \
    --splits-dir ./splits \
    --threshold 0.5
```

Outputs:

- `runs/exp_001/eval/metrics_summary.json` — numeric metrics
- `runs/exp_001/eval/confusion_matrix.png`
- `runs/exp_001/eval/threshold_curves.png` — for threshold tuning
- `runs/exp_001/eval/errors/false_negatives/` — missed faults (bbox annotated)
- `runs/exp_001/eval/errors/false_positives/` — false alarms

### 4. Threshold tuning (if needed)

To increase recall at the cost of some precision, lower the threshold:

```bash
python ~/micode/srnpolev2/evaluate.py \
    --checkpoint ./runs/exp_001/best_model.pt \
    --threshold 0.3
```

## Model architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Input:** 4 channels — RGB (3) + bbox mask (1)
- **Output:** 2 classes (`ok` / `problem`)
- **Loss:** Cross-entropy with class weights (`problem` ≈ 15% of data)
- **Optimizer:** AdamW, lr 1e-4, cosine schedule
- **Augmentation:** horizontal flip, color jitter
- **Input size:** 224×224 (resized from native crop)

## Expected performance

With 8,051 crops at ~15% `problem` rate:

| Metric | Expected range |
|---|---|
| Accuracy | 82–90% |
| Precision (problem) | 50–70% |
| Recall (problem) | 65–80% |
| F1 (problem) | 60–75% |
| ROC AUC | 0.85–0.92 |

The first training run is a baseline. Common improvements:

- Tuning the bbox mask blur sigma
- More aggressive augmentation
- Labelling additional `problem` examples
- Stronger backbone (e.g. ConvNeXt-Tiny)

## Error analysis

When inspecting `runs/exp_001/eval/errors/`:

**False negatives (missed faults):**

- `prob` close to 0.5 → borderline case; tuning mask blur may help
- `prob` very low → genuine model error or label noise
- Multiple poles in crop with mask correctly placed → contextual signal misleading

**False positives (false alarms):**

- Many twilight crops → low-light conditions are inherently harder
- Headlight reflections / bright surfaces → needs augmentation or extra examples

## Iteration cycle

1. Train → review metrics
2. Inspect `errors/` → identify failure patterns
3. **Label noise** → audit and re-label
4. **Insufficient data** → collect more `problem` examples
5. **Architectural limit** → switch to ConvNeXt or RoIAlign
6. Retrain and compare
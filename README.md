# AVF-YOLO v2 Mixed — Vessel Wall Detection Model

**Released**: 2026-05-27 | **Task**: Ultrasound vessel wall instance detection | **Architecture**: YOLOv8s (fine-tuned)

## Overview

This model detects **vessel wall** in ultrasound images (both longitudinal and transverse views) for AVF (arteriovenous fistula) clinical applications. It is the final output of a 3-stage progressive training pipeline designed to prevent catastrophic forgetting when expanding from longitudinal-only to mixed-view detection.

## Model Card

| Property | Value |
|----------|-------|
| Model | YOLOv8s (Ultralytics 8.4.46) |
| Task | Detect (single class: `wall`) |
| Input size | 640×640 |
| Base weights | yolov8s.pt → AVF_Round2_CosLR/best.pt |
| Freeze | backbone (first 10 layers) |
| mAP50 | **0.9919** |
| mAP50-95 | **0.7077** |
| Precision | 0.9732 |
| Recall | 0.9839 |
| Best epoch | 48 (early-stopped at 63) |
| Optimizer | AdamW (lr=1e-4, cos LR, warmup=5) |
| Data augmentations | Mosaic 1.0, MixUp 0.1, RandAugment, RandomErasing 0.2, HSV ±0.4 |

## Quick Start

```bash
# Install
pip install ultralytics>=8.4.46

# Load model
from ultralytics import YOLO
model = YOLO("best.pt")

# Inference on single image
results = model("your_image.jpg")
results[0].show()  # display

# Inference on folder
results = model("path/to/images/", save=True)

# Evaluate on validation set
model.val(data="data.yaml")
```

## Dataset

The mixed dataset combines:
- **712 transverse images** (full set from Stage 1)
- **30% longitudinal replay** (anti-forgetting strategy, ~345 images)
- **8:2 train/val split** (~846 train, ~211 val)

Class: `wall` (single class, class_id=0)

## Files

| File | Description |
|------|-------------|
| `best.pt` | Best model weights (epoch 48, 22 MB) |
| `args.yaml` | Training hyperparameters |
| `data.yaml` | Dataset configuration |
| `results.csv` | Per-epoch training metrics |
| `results.png` | Training/validation loss & metric curves |
| `confusion_matrix.png` | Confusion matrix (absolute) |
| `confusion_matrix_normalized.png` | Confusion matrix (normalized) |
| `BoxPR_curve.png` | Precision-Recall curve |
| `BoxF1_curve.png` | F1-Confidence curve |
| `BoxP_curve.png` | Precision-Confidence curve |
| `BoxR_curve.png` | Recall-Confidence curve |
| `val_batch0_labels.jpg` | Validation batch ground-truth |
| `val_batch0_pred.jpg` | Validation batch predictions |
| `labels.jpg` | Label distribution |
| `reproducibility_manifest.json` | Full environment & training trace |
| `stage1_convert_labelme_to_yolo.py` | Stage 1: LabelMe → YOLO format |
| `stage2_build_mixed_dataset.py` | Stage 2: Mix longitudinal + transverse |
| `stage3_train_mixed.py` | Stage 3: Training script |

## Training Pipeline (3-Stage)

```
Stage 1                    Stage 2                       Stage 3
LabelMe JSON               Longitudinal (full) 1150 imgs ─┐
  │                        Transverse (full)  712 imgs ───┤
  ▼                                  │                    │
dataset_transverse/          30% longitudinal replay ─────┤
  │                                  │                    │
  ▼                                  ▼                    ▼
Train AVF_Round1_640          dataset_mixed_v2/      AVF_v2_Mixed
  │                           ~1057 images           (this release)
  ▼
Train AVF_Round2_CosLR
  │
  ▼
best.pt (longitudinal) ──────▶ fine-tune w/ frozen backbone
```

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Ultralytics | 8.4.46 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (6 GB) |

## Usage Notes

- Input images should be 640×640 ultrasound frames
- The model expects BGR images (standard OpenCV/YOLO convention)
- Use `conf=0.25` (default) for standard detection; adjust based on clinical sensitivity requirements
- For batch inference, GPU memory peaks at ~1.4 GB with batch_size=8

## Citation

If you use this model in your research, please cite the AVF-YOLO project.

## License

This model is released for research purposes. Contact the project team for commercial use.

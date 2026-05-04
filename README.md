# Tag Corner Detection

Detecting the four boundary corners of tags in images using three progressively advanced approaches — from classical image processing to deep learning regression.

## Approaches

### 1. Classical CV (`ClassicalCV.ipynb`)
Baseline pipeline using traditional image processing:
- Grayscale → Otsu thresholding → morphological ops
- Contour detection via `findContours` + polygon approximation (`approxPolyDP`)
- Geometric corner sorting (top-left, top-right, bottom-right, bottom-left)

### 2. Heatmap Regression (`HeatMapApproach.ipynb`)
Deep learning approach predicting per-corner Gaussian heatmaps:
- **Encoder:** ResNet-18 (pretrained, ImageNet)
- **Decoder:** Transposed convolution head → 4-channel heatmap (one per corner)
- **Confidence Head:** Binary classifier for tag presence
- **Loss:** MSE (heatmaps) + BCE (confidence)
- Corner coordinates extracted via `argmax` on predicted heatmaps

### 3. Coordinate Regression (`ResnetCorner.ipynb`)
Direct regression of corner coordinates:
- **Backbones:** ResNet-18 / EfficientNet-B0 / ConvNeXt-Tiny
- **Output:** 8 values (x, y for 4 corners) + 1 confidence logit
- **Loss:** Weighted MSE (coords) + BCE (confidence)
- Best results with ConvNeXt-Tiny (frozen base, fine-tuned head)

## Data Format

```
flash_images/          # Input images
train_corners.csv      # Training labels
val_corners.csv        # Validation labels
```

Each CSV row: `image_name, x1, y1, x2, y2, x3, y3, x4, y4, has_tag`

## Setup

```bash
# Dependencies
pip install torch torchvision opencv-python matplotlib pandas

# Run any notebook
jupyter notebook ClassicalCV.ipynb
```

> **Note:** DL notebooks are configured for **Google Colab** with GPU. Adjust data paths if running locally.

## Results

| Approach | Strengths | Limitations |
|---|---|---|
| Classical CV | No training needed, interpretable | Fails on complex backgrounds |
| Heatmap | Spatial precision, robust localization | Higher compute, slower inference |
| Coord Regression | Fast inference, end-to-end | Sensitive to augmentation quality |

## Repository Structure

```
├── ClassicalCV.ipynb              # Traditional CV pipeline
├── HeatMapApproach.ipynb          # Heatmap-based deep learning
├── ResnetCorner.ipynb             # Coordinate regression models
├── TagCornerDetection_Approach.pdf # Detailed approach document
└── README.md
```

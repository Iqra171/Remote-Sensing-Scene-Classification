# Remote Sensing Scene Classification

A transfer-learning image classifier that identifies aerial/satellite scene types (airports, farmland, residential areas, forests, etc.) from the **AID (Aerial Image Dataset)**, using a MobileNetV2 backbone pretrained on ImageNet.

## Overview

The notebook downloads the AID dataset from Kaggle, sets up augmented train/validation generators, fine-tunes a MobileNetV2-based classifier on 30 land-use/scene classes, and evaluates performance with a classification report and confusion matrix.

## Dataset

- **Source:** [AID Scene Classification Dataset](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) (via `kagglehub`)
- **Classes (30):** Airport, BareLand, BaseballField, Beach, Bridge, Center, Church, Commercial, DenseResidential, Desert, Farmland, Forest, Industrial, Meadow, MediumResidential, Mountain, Park, Parking, Playground, Pond, Port, RailwayStation, Resort, River, School, SparseResidential, Square, Stadium, StorageTanks, Viaduct
- **Split:** 75% train / 25% validation (7,507 training images, 2,493 validation images)

## Model Architecture

- **Backbone:** MobileNetV2 (`imagenet` weights, `include_top=False`), frozen during training
- **Head:** GlobalAveragePooling2D → Dropout(0.3) → Dense(30, softmax)
- **Input size:** 224 × 224 × 3
- **Optimizer:** Adam (learning rate = 0.0008)
- **Loss:** Categorical cross-entropy

## Data Augmentation

Applied to the training set via `ImageDataGenerator`:
- Rescaling (1/255)
- Rotation (±25°)
- Zoom (±20%)
- Horizontal and vertical flips

## Training

- **Epochs:** 12
- **Batch size:** 32

## Results

Final epoch: **~88.4% training accuracy**, **~83.3% validation accuracy**.

Overall validation performance:

| Metric | Score |
|---|---|
| Accuracy | 0.83 |
| Macro avg F1-score | 0.83 |
| Weighted avg F1-score | 0.83 |

Strongest classes: Mountain, Parking, Viaduct, Desert (F1 ≥ 0.95).
Weakest classes: School, Resort, Airport, Square (F1 ≤ 0.68) — largely due to visual overlap with similar categories (e.g., School vs. Center vs. Commercial).

A full per-class precision/recall/F1 table and confusion matrix are generated in the notebook.

## Project Structure (Notebook Segments)

1. **Install & Download** — installs dependencies and downloads the AID dataset via `kagglehub`
2. **Data Generators** — builds augmented train/validation `ImageDataGenerator` pipelines
3. **Build Model** — constructs the MobileNetV2 transfer-learning model
4. **Training** — trains the model for 12 epochs
5. **Evaluate** — generates predictions and a classification report
6. **Confusion Matrix Plot** — visualizes per-class performance as a heatmap
7. **Accuracy & Loss Plot** — plots training/validation accuracy and loss curves

## Requirements

```
kagglehub
tensorflow
matplotlib
scikit-learn
seaborn
numpy
```

Install with:
```bash
pip install kagglehub tensorflow matplotlib scikit-learn seaborn
```

## Usage

1. Open the notebook in Jupyter/Colab.
2. Run all cells in order — the dataset will download automatically via `kagglehub` (requires a Kaggle account/API access).
3. Training takes roughly 25–30 minutes on a GPU-enabled runtime (12 epochs, ~135s/epoch).
4. Review the classification report and confusion matrix in the final cells.


- The MobileNetV2 backbone is **frozen** (no fine-tuning of pretrained layers) — only the classification head is trained. Unfreezing top layers for fine-tuning is a natural next step to push accuracy higher.
- Class imbalance and visual similarity between certain scene types (e.g., School/Center/Commercial, Resort/Park) are the main sources of misclassification.

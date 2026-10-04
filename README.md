# Road Signs Classification with CNN

An end-to-end Convolutional Neural Network (CNN) project for classifying road signs, covering the entire machine learning lifecycle: dataset download & preprocessing, model architecture design, training, evaluation, inference, and testing.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack & Prerequisites](#tech-stack--prerequisites)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Pipeline Workflow](#pipeline-workflow)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Metrics](#3-evaluation--metrics)
  - [4. Inference](#4-inference)
  - [5. Testing](#5-testing)
- [Code Quality & Linting](#code-quality--linting)
- [License](#license)

---

## 📌 Overview

This project aims to detect and classify traffic / road signs from image data using Deep Learning (Convolutional Neural Networks). The pipeline encompasses:
- Downloading and preprocessing datasets (e.g., GTSRB - German Traffic Sign Recognition Benchmark or Kaggle datasets).
- Data augmentation, normalization, and split into train/validation/test sets.
- Building and training custom CNN and YOLO/Ultralytics models.
- Tracking training metrics using TensorBoard.
- Running inference on single images, batches, or video streams.
- Automated testing and code validation.

---

## 🛠 Tech Stack & Prerequisites

- **Python**: `>= 3.11`
- **Package & Environment Manager**: [`uv`](https://docs.astral.sh/uv/) (fast Python package installer and resolver)
- **Deep Learning Frameworks**: TensorFlow / Keras, Ultralytics (YOLO)
- **Computer Vision & Image Processing**: OpenCV (`opencv-python-headless`), Pillow
- **Data Analysis & Visualization**: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn
- **Development & Testing**: Pytest, Ruff, Pre-commit

---

## 📂 Project Structure

```text
CNN_roadsigns/
├── data/                  # Raw and processed datasets (train/val/test splits)
├── files/                 # Auxiliary files, configurations, or label mappings
├── models/                # Saved model weights, checkpoints, and export artifacts (.keras, .h5, .pt)
├── notebooks/             # Jupyter notebooks for EDA, prototyping, and experimentation
├── reports/               # Generated reports, metrics, plots, and figures
├── src/                   # Source code for the project
│   ├── data/              # Data loading, downloading (kagglehub), and augmentation scripts
│   ├── models/            # CNN architecture definitions and loss/optimizer configs
│   ├── training/          # Training pipelines and callbacks (TensorBoard, EarlyStopping)
│   ├── inference/         # Prediction and inference utilities
│   └── utils/             # Helper functions (visualization, logging, metrics)
├── tests/                 # Unit and integration tests (pytest)
├── main.py                # Main application entry point / CLI
├── pyproject.toml         # Project metadata and dependencies configuration
├── uv.lock                # UV lockfile ensuring reproducible dependency resolution
└── README.md              # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Install `uv`
If you haven't installed `uv` yet, install it via the official installer:

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/CNN_roadsigns.git
cd CNN_roadsigns
```

### 3. Set Up Virtual Environment & Sync Dependencies
Use `uv` to automatically create a virtual environment with Python 3.11 and install all pinned dependencies from `pyproject.toml` / `uv.lock`:

```bash
# Create virtual environment and sync all dependencies
uv sync
```

To activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

Alternatively, you can run any command directly within the environment using `uv run <command>`.

---

## 🔄 Pipeline Workflow

### 1. Dataset Preparation
Download and preprocess road sign image datasets (such as GTSRB via `kagglehub` or direct download):

```bash
uv run python -m src.data.dataset_loader
```
- Performs train/validation/test split.
- Applies image resizing, normalization, and data augmentation (rotations, brightness adjustments, zooming).
- Saves processed arrays or directory structures to `data/`.

### 2. Model Training
Train the CNN classification network:

```bash
uv run python -m src.training.train --epochs 50 --batch-size 32 --learning-rate 0.001
```

- Features:
  - Checkpointing best model weights to `models/`.
  - Early stopping to prevent overfitting.
  - TensorBoard logging for real-time loss and accuracy tracking.

To launch TensorBoard:
```bash
uv run tensorboard --logdir logs/
```

### 3. Evaluation & Metrics
Evaluate trained models against test datasets and generate classification reports & confusion matrices:

```bash
uv run python -m src.training.evaluate --model models/best_model.keras
```
Metrics and confusion matrix plots are saved to `reports/`.

### 4. Inference
Run classification on new road sign images:

```bash
uv run python -m src.inference.predict --image path/to/road_sign.jpg --model models/best_model.keras
```

Or execute the primary entrypoint:
```bash
uv run python main.py
```

### 5. Testing
Execute test suites to ensure pipeline integrity and model components validity:

```bash
uv run pytest tests/
```

---

## 🧹 Code Quality & Linting

Format and check code using `ruff`:

```bash
# Check for lint issues
uv run ruff check .

# Format code
uv run ruff format .
```

Set up pre-commit hooks:
```bash
uv run pre-commit install
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

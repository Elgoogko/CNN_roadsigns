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

## Run on Google Colab
 
### 1. Prerequisites
 
- A Google account (Colab + Drive).
- A GitHub account with access to the `Elgoogko/CNN_roadsigns` repository.
- A Kaggle account (if the dataset is downloaded from Kaggle).
- All code changes **pushed to GitHub** (`git push`): Colab clones GitHub, not your local machine.
---
 
### 2. One-time Setup (to do only once)
 
#### 2.1 Colab Secrets
 
In Colab, open the left sidebar, key icon 🔑 (**Secrets**), then add:
 
| Secret            | When                        | Content                                         |
|-------------------|-----------------------------|-------------------------------------------------|
| `GITHUB_TOKEN`    | **Private** repository only | GitHub token with read access to the repository |
| `KAGGLE_USERNAME` | Download via Kaggle         | Kaggle username                                 |
| `KAGGLE_KEY`      | Download via Kaggle         | Kaggle API key                                  |
 
For each secret, **enable the "Notebook access" toggle**. Otherwise reading the secret will fail.
 
> ⚠️ Never write a token or a key in the notebook, in the code, or in Git.
 
#### 2.2 Drive Folder
 
Create the following directory structure in Google Drive:
 
```
MyDrive/
└── traffic_signs/
    ├── cache/      # preprocessed data (.npy)
    ├── models/     # model weights (.keras)
    ├── logs/       # training logs (CSV, TensorBoard)
    └── figures/    # curves, confusion matrices
```
 
#### 2.3 Colab Configuration File
 
Output paths are **not** modified in `configs/base.yaml`. A `configs/colab.yaml` file only overrides paths to point to the Drive folder above (`/content/drive/MyDrive/traffic_signs/...`). The same code thus works both locally and on Colab.
 
---
 
### 3. Starting a Session
 
To be repeated at **each new session**.
 
#### 3.1 Enable GPU
 
Menu **Runtime → Change runtime type → T4 GPU**.
 
To do **before** any installation: changing the runtime clears the session.
 
#### 3.2 Clone the Repository and Install (Cell 1)
 
```python
import os
from google.colab import userdata
 
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")  # not needed if the repository is public
 
!git clone https://$GITHUB_TOKEN@github.com/Elgoogko/CNN_roadsigns.git /content/traffic-sign-recognition
%cd /content/traffic-sign-recognition
!bash scripts/setup_colab.sh
```
 
The `scripts/setup_colab.sh` script:
 
1. installs `uv` if needed;
2. installs the project in editable mode (`import trafficsigns` becomes possible);
3. creates unversioned working directories;
4. displays versions and GPU detection.
**Check in the output**: `trafficsigns` is found and `TF GPU : True`. If it is `False`, go back to step 3.1.
 
> If Colab asks to **restart runtime** after installation, accept, then resume at step 3.3 (without re-cloning).
 
#### 3.3 Mount Google Drive (Cell 2)
 
```python
from google.colab import drive
drive.mount("/content/drive")
```
 
---
 
### 4. Data
 
The `data/` folder is not versioned: the cloned repository contains **no images**.
 
#### Principle
 
| Data                        | Location                                      | Why                                          |
|-----------------------------|-----------------------------------------------|----------------------------------------------|
| Raw images                  | Local disk `data/raw/`                        | Fast reading, re-downloadable                |
| Preprocessed cache (`.npy`) | Local `data/processed/` **and copy to Drive** | Avoids redoing preprocessing at each session |
 
> Never read raw images directly from Drive: tens of thousands of small files are very slow to read there.
 
#### First Session
 
1. Download the dataset using the `download.py` module (or `kagglehub`), after exposing Kaggle secrets in `os.environ`.
   - Alternative: place the dataset zip once on Drive, then copy and extract it locally.
2. Run preprocessing (`preprocessing.py`): resizing, normalization, `.npy` cache.
3. **Copy `data/processed/` to `MyDrive/traffic_signs/cache/`.**
#### Subsequent Sessions
 
Simply copy the cache from Drive to `data/processed/`. Preprocessing does not need to be done again.
 
---
 
### 5. Saving Results to Drive
 
Everything that needs to persist across sessions must be written to Drive:
 
| Item               | Destination                      |
|--------------------|----------------------------------|
| Model weights      | `MyDrive/traffic_signs/models/`  |
| Training logs      | `MyDrive/traffic_signs/logs/`    |
| Figures            | `MyDrive/traffic_signs/figures/` |
| Preprocessed cache | `MyDrive/traffic_signs/cache/`   |
 
Load configuration **with** the `colab.yaml` override so that these paths are used automatically.
 
Reminder: the **test** set is only used for final evaluation, never during training or model selection.
 
---
 
### 8. Resuming After a Disconnection
 
Colab disconnects after a period of inactivity (around 90 minutes) and limits session duration.
 
1. New session: repeat steps **3.1 to 3.3**.
2. Copy the cache from Drive again (step 4, "Subsequent Sessions").
3. Reload the latest checkpoint from `MyDrive/traffic_signs/models/`.
4. Resume training for the remaining epochs.
---
 
### 9. Retrieving Results and Versioning
 
- **Figures and experiment table**: copy from Drive to repository (`reports/figures/`, `reports/experiments_table.csv`).
- **Notebook** (Colab does not write to your local Git):
  - either **File → Save a copy in GitHub**;
  - or **File → Download .ipynb**, then commit from your computer (`nbstripout` cleans outputs).
- **Final model weights**: keep on Drive and locally (they are not versioned in Git).
---
 
### 10. Troubleshooting
 
| Symptom                                    | Probable Cause                                                          | Solution                                                              |
|--------------------------------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Secret not found / access denied           | "Notebook access" toggle disabled                                       | Enable it in the Secrets panel                                        |
| `ModuleNotFoundError: trafficsigns`        | Incorrect current directory, or installation not re-run after a restart | `%cd /content/traffic-sign-recognition`, then re-run `setup_colab.sh` |
| `TF GPU : False`                           | Runtime on CPU                                                          | Switch to T4 GPU, then resume from step 3.2                           |
| Installation failure (Python version)      | `requires-python` too strict for Colab's version                        | Use `>=3.11` in `pyproject.toml`                                      |
| Very long installation                     | TensorFlow reinstalled by project dependencies                          | Do not pin TensorFlow in `dependencies`: Colab already has it         |
| Very slow image reading                    | Reading from Drive                                                      | Copy data locally (`/content`)                                        |
| Insufficient memory (`Resource exhausted`) | Batch size too large                                                    | Halve the batch size                                                  |
| Weights lost after session                 | Written to `/content` instead of Drive                                  | Check the `colab.yaml` override                                       |
| Authentication error on clone              | Token missing, expired, or lacking read permission                      | Regenerate the token and update the secret                            |
 
---
 
### Session Routine
 
| # | Action                               | Estimated Duration |
|---|--------------------------------------|--------------------|
| 1 | GPU enabled, secrets in place        | 1 min              |
| 2 | Clone + `setup_colab.sh`             | 2 to 4 min         |
| 3 | Mount Drive                          | 30 s               |
| 4 | Copy Drive cache → local             | < 1 min            |
| 5 | Load configuration with `colab.yaml` | instant            |
| 6 | Training, checkpoints on Drive       | variable           |
---
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

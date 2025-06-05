# Crowd Counting using CSRNet

## Project Overview

This project implements a crowd counting system based on the **CSRNet** (Congested Scene Recognition Network) deep learning architecture. The goal is to estimate the number of people in an image accurately, given custom datasets of crowd images and their corresponding actual counts.

CSRNet leverages convolutional neural networks and dilated convolutions to generate crowd density maps, enabling precise counting even in highly congested scenes.

---

## Dataset

* **Training images:** 1500 images (`seq_000001.jpg` to `seq_001500.jpg`)
* **Validation images:** 500 images (`seq_001501.jpg` to `seq_002000.jpg` or similar)
* **Labels:** Provided in an Excel sheet containing:

  * Image filenames
  * Actual person count per image (numeric scalar, not density maps)

---

## Features

* Custom dataset support with crowd count labels
* CSRNet architecture for high-accuracy crowd counting
* Training and validation pipelines
* Metrics for evaluation of prediction accuracy and consistency
* Multi-hour training support for improved accuracy

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/crowd-counting-csrnet.git
cd crowd-counting-csrnet
```

2. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure CUDA drivers and PyTorch GPU support are properly installed if training on GPU.

---

## Usage

### Prepare Data

* Organize training and validation images in separate folders.
* Ensure the Excel file with image counts is correctly formatted and accessible.

### Training

```bash
python train.py --train_images path/to/train_images --val_images path/to/val_images --labels path/to/labels.xlsx --epochs 50 --batch_size 8 --lr 1e-5
```

* Adjust parameters as needed.
* Monitor training progress via printed logs or integrated visualization tools (TensorBoard etc.).

### Evaluation

```bash
python evaluate.py --val_images path/to/val_images --labels path/to/labels.xlsx --model path/to/trained_model.pth
```

* Outputs prediction metrics and accuracy scores.

---

| Step | File          | Purpose                       |
| ---- | ------------- | ----------------------------- |
| 1    | model.py    | Defines CSRNet                |
| 2    | dataset.py  | Prepares dataset & transforms |
| 3    | train.py    | Trains and saves model        |
| 4    | evaluate.py | Tests model on validation set |

---

## Model Architecture

CSRNet uses a pre-trained VGG-16 frontend followed by dilated convolutional layers to generate density maps that are integrated to estimate crowd counts.

For more details, refer to the original CSRNet paper:
[CSRNet Paper (CVPR 2018)](https://arxiv.org/abs/1802.10062)

---

## Performance Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Visual comparison of predicted counts vs actual counts

---

## Future Work

* Generate and train on density maps for improved spatial accuracy.
* Deploy as a FastAPI service for live crowd counting.
* Integrate real-time video stream counting.
* Experiment with data augmentation and model hyperparameters to boost accuracy.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* CSRNet authors for their original work.
* Public datasets and tools used for inspiration.
* Your own institution/project team (optional).

---


Here is a detailed `README.md` for your ResNet-based Crowd Counting project:

---

````markdown
# 🧠 ResNet-Based Crowd Counting Model

This project implements a high-accuracy crowd counting pipeline using a fine-tuned `ResNet50` backbone for **regression-based person count estimation** from images. It was designed for the **Kaggle Olympiad Crowd Density Prediction Challenge**, but the architecture is flexible and works on any crowd image dataset with image-wise count annotations.

---

## 📌 Features

- 🔍 **Regression-based crowd counting** using ResNet50
- 🧪 **Test-time augmentation (TTA)** for more stable predictions
- 🧠 **Transfer learning** with selective fine-tuning
- 🔄 **Stochastic Weight Averaging (SWA)** to stabilize training
- 📈 **Learning rate warmup & scheduling**
- 🧹 **Data preprocessing and augmentation**
- ✅ Compatible with **Google Colab**, **PyTorch**, and **Kaggle Datasets**

---

## 🧬 Project Structure

```bash
📁 crowd-counter-resnet/
├── main.py      # 📌 Full training + inference code
├── submission_highacc.csv      # ✅ Final output (submission format)
├── training_data.csv           # 🏷 Image ID + person count
├── test/                       # 🔎 Test images
├── output_train/train/         # 🧠 Training/Validation images
└── README.md                   # 📖 This file
````

---

## 🔧 Setup Instructions

### 1. Environment

Install the required packages:

```bash
pip install torch torchvision transformers pandas tqdm
```

If running on Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 2. Folder Structure Expected

```
📁 /content/drive/MyDrive/kaggle-olympiad-crowd-density-prediction/
├── training_data.csv                  # Columns: id,count
├── output_train/train/seq_000001.jpg # Images for training
├── test/test/seq_01501.jpg           # Images for testing
```

---

### 3. Run Training

Edit the paths in `main.py` and execute:

```bash
python main.py
```

This will:

* Train the model with **ResNet50 + dropout**
* Apply **data augmentations** like flip, color jitter, rotation
* Save the best model as `resnet50_regressor_best.pth`

---

### 4. Run Inference

The same script performs inference at the end using:

* **Test-Time Augmentation (TTA)**: predicts on original and flipped images
* Aggregates predictions and saves results in:

  ```bash
  submission_highacc.csv
  ```

---

## 📊 Model Architecture

```text
ResNet50 (pretrained on ImageNet)
└── Remove FC layer
└── AdaptiveAvgPool2d
└── Flatten
└── Linear(2048 → 256) + ReLU + Dropout(0.3)
└── Linear(256 → 1)  → outputs person count (float)
```

* Only `layer4` and custom regressor are trained (rest frozen)

---

## 🧪 Evaluation Metric

We use **Root Mean Squared Error (RMSE)**:

$$
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 }
$$

During training, both `Train RMSE` and `Validation RMSE` are monitored.

---

## 📈 Training Tricks Used

| Trick                 | Description                                           |
| --------------------- | ----------------------------------------------------- |
| 🔍 **TTA**            | Flip images at test time and average predictions      |
| 🧠 **SWA**            | Averages weights from the last few epochs             |
| 🔄 **LR Scheduler**   | Linear warmup followed by gradual decay               |
| 🎨 **Augmentation**   | Resize, flip, color jitter, rotate                    |
| ⛔ **Freezing layers** | ResNet frozen except for last block (layer4) and head |

---

## 📤 Sample Submission Format

```csv
id,count
1501,38
1502,45
1503,51
...
```

---

## 🔑 Possible Improvements

* Use **CSRNet** or **SANet** with density map supervision
* Switch to **EfficientNet** for better feature extraction
* Integrate **multi-scale patches** to boost detail recognition
* Apply **self-supervised pretraining** for better generalization

---

## 🧠 Author & Credits

**👤 Soham Penshanwar**
Final Year AI & Data Science | K.K. Wagh Institute of Engineering

This project was inspired by real-world challenges in analyzing CCTV footfall, safety monitoring, and public infrastructure planning.

---

## 📜 License

This code has been released under the Apache 2.0 open source license.

```
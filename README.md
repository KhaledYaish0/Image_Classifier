# 🌸 Image Classifier — Oxford Flowers 102 (TensorFlow/Keras)

This repository contains a complete image classification pipeline using **TensorFlow/Keras** on the **Oxford Flowers 102** dataset (loaded via `tensorflow_datasets`).  
It covers data loading and preprocessing, model training (Keras Sequential), evaluation (accuracy, loss, and confusion matrix), and simple inference.

---

## 📦 Tech & Data
- **Framework:** TensorFlow / Keras
- **Dataset:** Oxford Flowers 102 (`tfds.load("oxford_flowers102")`) — 102 flower categories
- **Language:** Python

---

## 🔬 Reproducible Results (from this repo)
- **Epochs:** 11  
- **Test loss:** **0.8864**  
- **Test accuracy:** **0.7772** (≈ 77.72%)  
> Results are taken from the executed notebook outputs.

You’ll also find:
- A printed **model summary** (Keras Sequential)
- Training curves (loss/accuracy)
- A **confusion matrix** visualization for per-class performance

---

## 📁 Project Layout
Image_Classifier_Project/
└── Image_Classifier.ipynb # Main notebook (load data, train, evaluate, plot)

> Note: The dataset itself is not stored in the repo; it’s fetched by `tensorflow_datasets` on first run.

---

## ⚙️ Setup
```bash
# (Optional) create a venv first
# python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate                              # Windows

pip install -r requirements.txt
```
##▶️ How to Run

Open the notebook and run all cells:

jupyter notebook Image_Classifier.ipynb
# or
jupyter lab Image_Classifier.ipynb


The notebook will:

1. Load Oxford Flowers 102 via TFDS

2. Split into train/validation/test

3. Preprocess images

4. Build & train a Keras model

5. Evaluate on the test set and plot metrics (including a confusion matrix)

## 🧪 Inference (example)

Inside the notebook you’ll find a cell that runs prediction on one or more images and shows the predicted label.
You can adapt it to run on your own images (paths or tensors).

## 📜 License

---

### `requirements.txt`

```txt
tensorflow>=2.10
tensorflow-datasets>=4.9
numpy
matplotlib
scikit-learn

# 🌿 Satellite Deforestation Detection System

An end-to-end **Machine Learning + Computer Vision** project that detects deforestation by analyzing satellite-like imagery using vegetation indices and spatial change detection.

---

## 🚀 Overview

This project simulates satellite data and builds a complete pipeline to:

* Detect vegetation loss using **NDVI (Normalized Difference Vegetation Index)**
* Identify structural changes using **SSIM**
* Train a **Random Forest classifier** on patch-based features
* Generate **deforestation probability maps**
* Provide a final verdict based on both statistical and ML outputs

---

## 🧠 How It Works

### 🔹 Pipeline

1. **Generate Satellite Images**

   * Synthetic forest images (Before)
   * Simulated deforestation (After)

2. **Feature Extraction**

   * NDVI (vegetation index)
   * Pixel difference
   * Vegetation loss

3. **Change Detection**

   * Pixel-based difference
   * SSIM (Structural Similarity Index)

4. **Machine Learning**

   * Patch-based feature extraction (16×16)
   * Auto-labeling (weak supervision)
   * Random Forest training

5. **Prediction**

   * Model predicts deforestation probability for each patch
   * Reconstructed into a spatial heatmap

---

## 📊 Outputs

After running the script, an `outputs/` folder is created:

```
outputs/
│
├── result.png                  # Visualization (Before vs After vs Heatmap)
├── report.json                 # Final metrics and verdict
└── deforestation_model.pkl     # Trained ML model
```

---

## 🖼️ Visualization Explained

### 📌 `result.png`

* **Left** → Before image (dense forest)
* **Middle** → After image (deforestation simulated)
* **Right** → Heatmap showing probability of deforestation

Red = High probability 🔴
Green/Low = No significant change 🟢

---

## 💾 Model Saving (`.pkl`)

The trained model is saved as:

```
outputs/deforestation_model.pkl
```

It contains:

* `RandomForestClassifier`
* `StandardScaler`

### 🔄 Reusing the Model

Instead of retraining every time, you can load the model:

```python
import joblib

data = joblib.load("outputs/deforestation_model.pkl")
model = data["model"]
scaler = data["scaler"]
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/satellite-deforestation-detection.git
cd satellite-deforestation-detection
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* Linux/Mac:

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python deforestation.py
```

---

## 📍 Configuration

You can modify parameters inside the script:

```python
LAT  = 21.9497
LON  = 89.1833
DEFORESTATION_AMOUNT = 0.28
SIZE = 256
PATCH = 16
```

---

## 📈 Final Decision Logic

Deforestation is detected if:

```
Vegetation loss > 5%  OR  ML prediction > 10%
```

---

## ⚠️ Limitations

* Uses **synthetic satellite data** (not real imagery)
* No temporal sequence (only before/after comparison)
* Depends on NIR channel (not always available in all datasets)

---

## 🔮 Future Improvements

* Use real satellite data (Sentinel-2 / Landsat)
* Deep learning (U-Net for segmentation)
* Time-series analysis
* Web dashboard (Streamlit / FastAPI)

---


## ⭐ If you found this useful

Give it a ⭐ on GitHub!

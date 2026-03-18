# 🌿 Satellite Deforestation Detection System

An end-to-end **Machine Learning + Computer Vision** project that detects deforestation by analyzing satellite imagery using vegetation indices and change detection techniques.

---

## 🚀 Overview

This system compares two satellite images of the same location:

* `before` → forest area
* `after` → potentially deforested area

It then:

* Measures vegetation loss using **NDVI**
* Detects structural changes using **SSIM**
* Uses a **Random Forest model** to classify deforestation
* Generates a **probability heatmap**
* Outputs a final verdict

---

## 🧠 How It Works

### 🔹 Pipeline

1. **Input Images**

   * Before image (forest)
   * After image (after change)

2. **Feature Extraction**

   * NDVI (vegetation index)
   * Pixel difference
   * Vegetation loss

3. **Change Detection**

   * Pixel difference
   * SSIM (Structural Similarity)

4. **Machine Learning**

   * Image split into patches (16×16)
   * Features extracted per patch
   * Random Forest classifier trained

5. **Prediction**

   * Patch-wise prediction
   * Reconstructed into full heatmap

---

## 📁 Project Structure

```plaintext
satellite-deforestation-detection/
│
├── deforestation.py        # Main script (train + predict + visualize)
├── test.py                 # Run inference on custom images
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/                # Generated after running (ignored in Git)
```

---

## ⚙️ Installation

### 1. Clone repository

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

## ▶️ Run Main Pipeline

```bash
python deforestation.py
```

---

## 📊 Output Files

After running, an `outputs/` folder will be created:

```plaintext
outputs/
│
├── result.png                  # Visualization (before, after, heatmap)
├── report.json                 # Metrics + final verdict
└── deforestation_model.pkl     # Trained ML model
```

---

## 🖼️ Visualization

### `result.png`

* **Before Image** → original forest
* **After Image** → deforested region
* **Heatmap** → probability of deforestation

🔴 Red = High deforestation probability
🟢 Green = No change

---

## 💾 Model Saving & Loading

The trained model is saved as:

```plaintext
outputs/deforestation_model.pkl
```

### Load model manually

```python
import joblib

data = joblib.load("outputs/deforestation_model.pkl")
model = data["model"]
scaler = data["scaler"]
```

---

## 🧪 Run on Custom Images (`test.py`)

You can test your own satellite images using the saved model.

---

### 📂 Required Input

Place these files in the project root:

```plaintext
before.png
after.png
```

---

### ⚠️ Requirements

* Same image size
* Same location (aligned images)
* Preferably vegetation present

---

### ▶️ Run

```bash
python test.py
```

---

### 🔄 What Happens

1. Loads model from:

   ```
   outputs/deforestation_model.pkl
   ```

2. Reads:

   * `before.png`
   * `after.png`

3. Extracts features (NDVI, change, vegetation loss)

4. Predicts:

   * Deforestation probability
   * Patch-wise classification

5. Displays result

---

## 📈 Final Decision Logic

Deforestation is detected if:

```
Vegetation loss > 5%  OR  ML prediction > 10%
```

---

## ⚠️ Limitations

* Uses synthetic data (not real satellite imagery)
* Only compares two images (no time series)
* NDVI requires NIR channel (simulated here)

---

## 🔮 Future Improvements

* Real satellite data (Sentinel-2 / Landsat)
* Deep learning (U-Net segmentation)
* Time-series analysis
* Web dashboard (Streamlit / FastAPI)

---

## 🧾 Resume Line

> Built an end-to-end deforestation detection system using NDVI, SSIM, and Random Forest with patch-based feature extraction and spatial probability mapping.

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!

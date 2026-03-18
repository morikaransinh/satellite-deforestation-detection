"""
🌿 Satellite Deforestation Detection System
Single-file version (no notebook cells)
"""

import os
import json
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ==============================
# ⚙️ SETTINGS
# ==============================

LAT  = 21.9497
LON  = 89.1833
SITE = 'Sundarbans Mangroves, Bangladesh'

DEFORESTATION_AMOUNT = 0.28
SIZE  = 256
PATCH = 16

OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)

MODEL_PATH = f'{OUT}/deforestation_model.pkl'

# ==============================
# 🌲 IMAGE FUNCTIONS
# ==============================

def make_forest_image(size=256, seed=0):
    rng = np.random.default_rng(seed)
    r   = rng.integers(20,  60, (size, size)).astype(np.float32)
    g   = rng.integers(80, 140, (size, size)).astype(np.float32)
    b   = rng.integers(20,  60, (size, size)).astype(np.float32)
    nir = rng.integers(160, 220, (size, size)).astype(np.float32)
    noise = gaussian_filter(rng.random((size, size)), sigma=10) * 35
    img   = np.stack([r+noise, g+noise, b+noise, nir-noise], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_deforestation(img, amount=0.25, seed=1):
    rng   = np.random.default_rng(seed)
    out   = img.copy()
    size  = img.shape[0]
    patch = int(np.sqrt(amount) * size)

    px = int(rng.integers(10, size - patch - 10))
    py = int(rng.integers(10, size - patch - 10))

    out[py:py+patch, px:px+patch, 0] = rng.integers(140, 175, (patch, patch))
    out[py:py+patch, px:px+patch, 1] = rng.integers(100, 125, (patch, patch))
    out[py:py+patch, px:px+patch, 2] = rng.integers( 55,  85, (patch, patch))
    out[py:py+patch, px:px+patch, 3] = rng.integers( 25,  65, (patch, patch))

    return out

# ==============================
# 🌿 NDVI & FEATURES
# ==============================

def get_ndvi(img):
    R   = img[:,:,0].astype(np.float32)
    NIR = img[:,:,3].astype(np.float32)
    return (NIR - R) / (NIR + R + 1e-8)


def get_gray(img):
    rgb = img[:,:,:3].astype(np.float32) / 255.0
    return 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]


def pixel_difference(before, after):
    diff = np.abs(get_gray(after) - get_gray(before))
    return diff / (diff.max() + 1e-8)


def ssim_difference(before, after):
    score, smap = ssim(get_gray(before), get_gray(after),
                       data_range=1.0, full=True)
    return np.clip(1.0 - smap, 0, 1).astype(np.float32), score

# ==============================
# 🧠 FEATURE EXTRACTION
# ==============================

def extract_features(before_img, after_img, patch_size=16):
    sz     = before_img.shape[0]
    ndvi_b = get_ndvi(before_img)
    ndvi_a = get_ndvi(after_img)
    pdiff  = pixel_difference(before_img, after_img)

    features, coords = [], []

    for r in range(0, sz - patch_size + 1, patch_size):
        for c in range(0, sz - patch_size + 1, patch_size):
            nb = ndvi_b[r:r+patch_size, c:c+patch_size].mean()
            na = ndvi_a[r:r+patch_size, c:c+patch_size].mean()
            pd = pdiff [r:r+patch_size, c:c+patch_size].mean()
            vb = (ndvi_b[r:r+patch_size, c:c+patch_size] > 0.2).mean()
            va = (ndvi_a[r:r+patch_size, c:c+patch_size] > 0.2).mean()

            features.append([nb, na, na - nb, pd, vb - va])
            coords.append((r, c))

    return np.array(features, dtype=np.float32), coords


def auto_label(X):
    return ((X[:,2] < -0.08) | (X[:,3] > 0.07) | (X[:,4] > 0.30)).astype(int)


def rebuild_prob_map(y_proba, coords, size=256, patch_size=16):
    pmap = np.zeros((size, size), dtype=np.float32)
    for i, (r, c) in enumerate(coords):
        pmap[r:r+patch_size, c:c+patch_size] = y_proba[i]
    return pmap

# ==============================
# 📊 TRAIN MODEL
# ==============================

def train_model():
    all_X, all_y = [], []
    levels = np.linspace(0.05, 0.55, 12)

    for i, d in enumerate(levels):
        bf = make_forest_image(SIZE, i*7)
        af = add_deforestation(bf, d, i*7+100)

        X, _ = extract_features(bf, af, PATCH)
        y    = auto_label(X)

        all_X.append(X)
        all_y.append(y)

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )

    scores = cross_val_score(clf, X_scaled, y_train, cv=5)
    print(f'Accuracy: {scores.mean():.3f}')

    clf.fit(X_scaled, y_train)

    joblib.dump({
        'model': clf,
        'scaler': scaler
    }, MODEL_PATH)

    return clf, scaler

# ==============================
# 🔮 PREDICTION
# ==============================

def predict(model, scaler, before, after):
    X, coords = extract_features(before, after, PATCH)
    X_scaled = scaler.transform(X)

    y_pred  = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:,1]

    prob_map = rebuild_prob_map(y_proba, coords, SIZE, PATCH)

    return y_pred, prob_map

# ==============================
# 🎨 VISUALIZATION
# ==============================

def visualize(before, after, prob_map):
    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.imshow(before[:,:,:3])
    plt.title("Before")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(after[:,:,:3])
    plt.title("After")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(prob_map, cmap='hot')
    plt.title("Deforestation Probability")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"{OUT}/result.png")
    plt.show()

# ==============================
# 🚀 MAIN
# ==============================

def main():
    print(f"\n🌍 Site: {SITE}")

    seed = int(abs(LAT * 100 + LON * 100)) % 9999

    before = make_forest_image(SIZE, seed)
    after  = add_deforestation(before, DEFORESTATION_AMOUNT, seed+50)

    print("\n🏋️ Training model...")
    model, scaler = train_model()

    print("\n🔮 Predicting...")
    y_pred, prob_map = predict(model, scaler, before, after)

    print(f"Deforested patches: {y_pred.mean()*100:.1f}%")

    visualize(before, after, prob_map)

    report = {
        "site": SITE,
        "deforestation_pct": float(y_pred.mean()*100)
    }

    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Done! Check outputs folder.")

# ==============================

if __name__ == "__main__":
    main()
import numpy as np
import cv2
import joblib

# ==============================
# 📦 LOAD MODEL
# ==============================
MODEL_PATH = "deforestation_model.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle['model']
scaler = bundle['scaler']
PATCH = bundle['patch_size']
SIZE = bundle['image_size']

print("✅ Model loaded!")

# ==============================
# 🧠 HELPER FUNCTIONS
# ==============================

def get_ndvi(img):
    R = img[:,:,0].astype(np.float32)
    NIR = img[:,:,3].astype(np.float32)
    return (NIR - R) / (NIR + R + 1e-8)

def get_gray(img):
    rgb = img[:,:,:3].astype(np.float32) / 255.0
    return 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]

def pixel_difference(before, after):
    diff = np.abs(get_gray(after) - get_gray(before))
    return diff / (diff.max() + 1e-8)

def extract_features(before_img, after_img, patch_size=16):
    sz = before_img.shape[0]
    ndvi_b = get_ndvi(before_img)
    ndvi_a = get_ndvi(after_img)
    pdiff  = pixel_difference(before_img, after_img)

    features, coords = [], []

    for r in range(0, sz - patch_size + 1, patch_size):
        for c in range(0, sz - patch_size + 1, patch_size):
            nb = ndvi_b[r:r+patch_size, c:c+patch_size].mean()
            na = ndvi_a[r:r+patch_size, c:c+patch_size].mean()
            pd = pdiff[r:r+patch_size, c:c+patch_size].mean()

            vb = (ndvi_b[r:r+patch_size, c:c+patch_size] > 0.2).mean()
            va = (ndvi_a[r:r+patch_size, c:c+patch_size] > 0.2).mean()

            features.append([nb, na, na - nb, pd, vb - va])
            coords.append((r, c))

    return np.array(features), coords

def rebuild_prob_map(y_proba, coords, size, patch_size):
    pmap = np.zeros((size, size), dtype=np.float32)
    for i, (r, c) in enumerate(coords):
        pmap[r:r+patch_size, c:c+patch_size] = y_proba[i]
    return pmap

# ==============================
# 🧠 IMAGE LOADING + FIX
# ==============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"❌ Could not load image: {path}")

    # Convert to 4 channel if needed
    if len(img.shape) == 2:
        img = np.stack([img, img, img, img], axis=-1)
    elif img.shape[2] == 3:
        nir = img[:,:,1]  # fake NIR
        img = np.dstack([img, nir])

    return img

def resize_to_match(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    return img1, img2

# ==============================
# 📥 INPUT IMAGES
# ==============================

BEFORE_PATH = "before.png"
AFTER_PATH  = "after.png"

before = load_image(BEFORE_PATH)
after  = load_image(AFTER_PATH)

# 🔥 FIX: ensure same size
before, after = resize_to_match(before, after)

# Optional safety check
assert before.shape == after.shape, "❌ Image size mismatch!"

print("✅ Images loaded and resized!")

# ==============================
# 🤖 PREDICTION
# ==============================

X, coords = extract_features(before, after, PATCH)
X_scaled  = scaler.transform(X)

y_pred  = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:,1]

prob_map = rebuild_prob_map(y_proba, coords, before.shape[0], PATCH)

deforestation_pct = y_pred.mean() * 100

print(f"\n🌿 Deforestation Prediction: {deforestation_pct:.2f}%")

# ==============================
# 🏁 FINAL VERDICT
# ==============================

if deforestation_pct > 10:
    print("🔴 DEFORESTATION DETECTED")
else:
    print("🟢 NO SIGNIFICANT CHANGE")

# ==============================
# 💾 SAVE OUTPUT (optional)
# ==============================

cv2.imwrite("probability_map.png", (prob_map * 255).astype(np.uint8))
print("💾 Saved: probability_map.png")
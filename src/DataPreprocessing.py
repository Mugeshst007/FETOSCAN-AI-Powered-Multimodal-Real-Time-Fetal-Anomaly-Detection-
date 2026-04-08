import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import cv2
import joblib

# 🔥 CORRECT PREPROCESSING (MobileNetV2)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ================= CONFIG =================
DATASET_DIR = "dataset_1"
CTG_FOLDER = "ctg"
CTG_DATA_FILE = "fetal_health.csv"
TARGET_COLUMN = "fetal_health"

LABELS = ["Normal", "Benign", "Malignant"]


# ================= MAIN FUNCTION =================
def load_and_prepare_data(base_path):

    # ================= CTG =================
    ctg_path = os.path.join(base_path, "data", CTG_FOLDER, CTG_DATA_FILE)

    if not os.path.exists(ctg_path):
        raise FileNotFoundError(f"CTG file not found: {ctg_path}")

    ctg_df = pd.read_csv(ctg_path)

    # Map labels → 0,1,2
    label_map_ctg = {1: 0, 2: 1, 3: 2}
    ctg_df[TARGET_COLUMN] = ctg_df[TARGET_COLUMN].map(label_map_ctg)

    X_ctg = ctg_df.drop(TARGET_COLUMN, axis=1).values
    y_ctg = ctg_df[TARGET_COLUMN].values

    print("✅ CTG loaded:", X_ctg.shape)


    # ================= IMAGE =================
    train_dir = os.path.join(base_path, "data", DATASET_DIR, "Data", "train")
    val_dir = os.path.join(base_path, "data", DATASET_DIR, "Data", "validation")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    images, labels = [], []

    label_map = {
        "normal": 0,
        "benign": 1,
        "malignant": 2
    }

    def load_images(folder):
        for root, _, files in os.walk(folder):

            label_name = os.path.basename(root).lower()

            if label_name not in label_map:
                continue

            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):

                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)

                    if img is None:
                        continue

                    # Resize
                    img = cv2.resize(img, (224, 224))

                    # Convert color
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 🔥 MobileNet preprocessing [-1,1]
                    img = preprocess_input(img.astype(np.float32))

                    images.append(img)
                    labels.append(label_map[label_name])


    print("📂 Loading training images...")
    load_images(train_dir)

    print("📂 Loading validation images...")
    load_images(val_dir)

    X_img = np.array(images)
    y_img = np.array(labels)

    print("✅ Images loaded:", X_img.shape)
    print("📊 Label distribution:", np.bincount(y_img))

    # ❗ IMPORTANT: DO NOT BALANCE DATA
    # class_weights in training will handle imbalance


    # ================= SCALE CTG =================
    scaler = StandardScaler()
    X_ctg = scaler.fit_transform(X_ctg)

    return X_img, X_ctg, y_img, y_ctg, scaler


# ================= SAVE =================
def save_data(X_img, X_ctg, y_img, y_ctg, scaler, base_path):

    save_path = os.path.join(base_path, "data", "processed_data.npz")

    np.savez_compressed(
        save_path,
        X_img=X_img,
        X_ctg=X_ctg,
        y_img=y_img,
        y_ctg=y_ctg
    )

    joblib.dump(scaler, os.path.join(base_path, "data", "ctg_scaler.pkl"))

    print("✅ Saved processed data")


# ================= RUN =================
if __name__ == "__main__":

    BASE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.join(BASE, "..")

    X_img, X_ctg, y_img, y_ctg, scaler = load_and_prepare_data(ROOT)
    save_data(X_img, X_ctg, y_img, y_ctg, scaler, ROOT)
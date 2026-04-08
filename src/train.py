import os
import numpy as np
import tensorflow as tf

from image_model import build_image_model
from ctg_model import build_ctg_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ================= PATH =================
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

DATA_PATH = os.path.join(ROOT, "data", "processed_data.npz")

data = np.load(DATA_PATH)

X_img, X_ctg = data["X_img"], data["X_ctg"]
y_img, y_ctg = data["y_img"], data["y_ctg"]

print("✅ Data loaded")


# ================= SPLIT =================
X_img_tr, X_img_val, y_img_tr, y_img_val = train_test_split(
    X_img, y_img, test_size=0.2, stratify=y_img, random_state=42
)

X_ctg_tr, X_ctg_val, y_ctg_tr, y_ctg_val = train_test_split(
    X_ctg, y_ctg, test_size=0.2, stratify=y_ctg, random_state=42
)


# ================= CLASS WEIGHTS =================
class_weights = {
    0: 1.0,   # Normal
    1: 2.5,   # Benign
    2: 3.0    # Malignant
}

print("📊 Using class weights:", class_weights)


# ================= FIXED FOCAL LOSS =================
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):

        # 🔥 FIX SHAPE ISSUE
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true = tf.one_hot(y_true, depth=3)

        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)

        return alpha * tf.pow(1 - pt, gamma) * cross_entropy

    return loss


# ================= AUGMENTATION =================
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)


# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, min_lr=1e-5)
]


# ================= IMAGE MODEL =================
print("\n🚀 Training MobileNetV2...")

img_model = build_image_model()

# 🔥 USE FIXED FOCAL LOSS
img_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=["accuracy"]
)

img_model.fit(
    datagen.flow(X_img_tr, y_img_tr, batch_size=32, shuffle=True),
    epochs=25,
    validation_data=(X_img_val, y_img_val),
    class_weight=class_weights,
    callbacks=callbacks
)


# ================= SAVE IMAGE MODEL =================
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

img_model.save(os.path.join(ROOT, "models", "image_model.keras"))

print("✅ Image model trained & saved")


# ================= CTG MODEL =================
print("\n📊 Training CTG model...")

ctg_model = build_ctg_model()

ctg_model.fit(
    X_ctg_tr, y_ctg_tr,
    epochs=30,
    batch_size=32,
    validation_data=(X_ctg_val, y_ctg_val),
    callbacks=callbacks
)

ctg_model.save(os.path.join(ROOT, "models", "ctg_model.keras"))

print("🚀 TRAINING COMPLETE")
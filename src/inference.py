import tensorflow as tf
import numpy as np
import base64, json, sys, os
from PIL import Image
from io import BytesIO
import joblib

from fusion_model import late_fusion

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_MODEL = tf.keras.models.load_model(
    os.path.join(BASE, "models", "image_model.keras"),
    compile=False
)

CTG_MODEL = tf.keras.models.load_model(
    os.path.join(BASE, "models", "ctg_model.keras"),
    compile=False
)

SCALER = joblib.load(os.path.join(BASE, "data", "ctg_scaler.pkl"))

LABELS = ["Normal", "Benign", "Malignant"]


# ================= IMAGE =================
def preprocess_image(image_b64):
    img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr[None]


# ================= CORE =================
def run(data):
    image = data.get("image")
    ctg = data.get("ctg")

    result = {
        "status": "success",
        "debug": {}
    }

    img_pred = None
    ctg_pred = None

    # ===== IMAGE =====
    if image:
        result["debug"]["image_received"] = True
        x = preprocess_image(image)
        img_pred = IMG_MODEL.predict(x)[0]

        result["image_prediction"] = LABELS[np.argmax(img_pred)]
        result["image_confidence"] = float(np.max(img_pred))

    # ===== CTG =====
    if ctg:
        result["debug"]["ctg_received"] = True
        x = SCALER.transform([ctg])
        ctg_pred = CTG_MODEL.predict(x)[0]

        result["ctg_prediction"] = LABELS[np.argmax(ctg_pred)]
        result["ctg_confidence"] = float(np.max(ctg_pred))

    # ===== SAFETY =====
    if img_pred is None and ctg_pred is None:
        return {
            "status": "error",
            "error": "No input provided"
        }

    # ===== FUSION =====
    final_pred = late_fusion(img_pred, ctg_pred)

    result["final_prediction"] = LABELS[np.argmax(final_pred)]
    result["final_confidence"] = float(np.max(final_pred))

    # ===== MODE =====
    if image and ctg:
        result["mode"] = "multimodal"
    elif image:
        result["mode"] = "image"
    else:
        result["mode"] = "ctg"

    return result


# 🔥 IMPORTANT FIX
if __name__ == "__main__":
    output = run(json.loads(sys.stdin.read()))
    print(json.dumps(output, indent=2))
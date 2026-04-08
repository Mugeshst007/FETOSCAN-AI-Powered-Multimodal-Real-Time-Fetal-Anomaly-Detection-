import tensorflow as tf
import numpy as np
import base64, json, sys, os, cv2
from PIL import Image
from io import BytesIO

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "image_model.keras")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ✅ FIXED for MobileNetV2
LAST_CONV_LAYER = "Conv_1"


def preprocess_image(image_b64):
    img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32)

    # IMPORTANT: same preprocessing as training
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    arr = preprocess_input(arr)

    return arr, arr[None]


def gradcam(img_tensor):

    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)

    # 🔥 FIX: avoid None gradients
    if grads is None:
        return np.zeros((7,7))

    grads = grads[0]
    conv_out = conv_out[0]

    weights = tf.reduce_mean(grads, axis=(0,1))
    heatmap = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))

    heatmap = np.maximum(heatmap, 0)

    # 🔥 FIX normalization
    max_val = np.max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap)

    heatmap /= max_val

    return heatmap


def overlay(heatmap, img):
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.uint8(img * 255)

    # 🔥 Sharper blending
    result = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    _, buf = cv2.imencode(".png", result)
    return base64.b64encode(buf).decode()


def run(data):
    image = data.get("image")

    if not image:
        return {"gradcam": ""}

    img_raw, img_tensor = preprocess_image(image)

    heatmap = gradcam(img_tensor)

    return {"gradcam": overlay(heatmap, img_raw)}


if __name__ == "__main__":
    print(json.dumps(run(json.loads(sys.stdin.read()))))
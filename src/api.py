import os
import base64
import json
import re
import subprocess

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# ================================
# PATH CONFIG
# ================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

ML_PYTHON = os.path.join(PROJECT_ROOT, "ml_env", "Scripts", "python.exe")
if not os.path.exists(ML_PYTHON):
    ML_PYTHON = os.path.join(PROJECT_ROOT, "ml_env", "bin", "python")

INFERENCE_SCRIPT = os.path.join(CURRENT_DIR, "inference.py")
GRADCAM_SCRIPT = os.path.join(CURRENT_DIR, "gradcam.py")
SHAP_SCRIPT = os.path.join(CURRENT_DIR, "explainability.py")

# ================================
# FASTAPI
# ================================
app = FastAPI()

# ================================
# SAFE JSON PARSER
# ================================
def safe_json_parse(text):
    try:
        return json.loads(text.strip())
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return {"status": "error", "error": "Invalid JSON output"}

# ================================
# OLLAMA SETUP
# ================================
USE_OLLAMA = True

try:
    import ollama
    ollama.list()
    print("✅ Ollama Connected")
except Exception:
    USE_OLLAMA = False
    print("⚠️ Ollama not available")

# ================================
# OLLAMA FUNCTION
# ================================
def get_advice(label, confidence):

    if not USE_OLLAMA:
        return {
            "diagnosis_summary": "AI unavailable",
            "recommended_actions": [],
            "care_plan": "",
            "lifestyle_tips": []
        }

    prompt = f"""
Return JSON only.

{{
"diagnosis_summary": "string",
"recommended_actions": ["string"],
"care_plan": "string",
"lifestyle_tips": ["string"]
}}

Diagnosis: {label}
Confidence: {confidence:.2f}
"""

    try:
        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        return safe_json_parse(response["message"]["content"])

    except Exception:
        return {
            "diagnosis_summary": "AI unavailable",
            "recommended_actions": [],
            "care_plan": "",
            "lifestyle_tips": []
        }

# ================================
# CLEAN ML RUNNER
# ================================
def run_ml(script, payload):

    try:
        result = subprocess.run(
            [ML_PYTHON, script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
            cwd=CURRENT_DIR
        )

        # 🔥 CLEAN STDERR (only useful logs)
        if result.stderr.strip():
            print("ML LOG:", result.stderr.strip())

        # 🔥 CLEAN STDOUT
        if "gradcam" in result.stdout:
            print("ML: GradCAM generated ✅")
        elif "shap" in result.stdout:
            print("ML: SHAP generated ✅")
        else:
            print("ML OUTPUT:\n", result.stdout)

        if result.returncode != 0:
            return {"status": "error", "error": result.stderr}

        return safe_json_parse(result.stdout)

    except Exception as e:
        return {"status": "error", "error": str(e)}

# ================================
# HOME
# ================================
@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(CURRENT_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# ================================
# CTG VALIDATION
# ================================
def validate_ctg(ctg):

    if ctg is None:
        return None

    if not isinstance(ctg, list) or len(ctg) != 21:
        raise ValueError("CTG must be exactly 21 values")

    return [float(x) for x in ctg]

# ================================
# MAIN API
# ================================
@app.post("/predict")
async def predict(file: UploadFile = File(None), ctg: str = Form(None)):

    try:
        image_b64 = None

        # IMAGE
        if file:
            image_bytes = await file.read()
            image_b64 = base64.b64encode(image_bytes).decode()

        # CTG
        ctg_data = None
        if ctg:
            try:
                parsed = json.loads(ctg)
                ctg_data = validate_ctg(parsed.get("ctg_data"))
            except Exception as e:
                return {"success": False, "error": f"Invalid CTG: {str(e)}"}

        payload = {
            "image": image_b64,
            "ctg": ctg_data
        }

        # ================= INFERENCE =================
        ml = run_ml(INFERENCE_SCRIPT, payload)

        if ml.get("status") != "success":
            return {"success": False, "error": ml.get("error")}

        # ================= GRADCAM =================
        gradcam_img = ""
        if image_b64:
            grad = run_ml(GRADCAM_SCRIPT, payload)
            if grad.get("gradcam"):
                gradcam_img = "data:image/png;base64," + grad["gradcam"]

        # ================= SHAP =================
        shap_img = ""
        if ctg_data:
            shap = run_ml(SHAP_SCRIPT, payload)
            if shap.get("shap"):
                shap_img = "data:image/png;base64," + shap["shap"]

        # ================= OLLAMA =================
        label = ml.get("final_prediction") or ml.get("image_prediction") or ml.get("ctg_prediction")
        confidence = ml.get("final_confidence") or ml.get("image_confidence") or ml.get("ctg_confidence")

        advice = get_advice(label, confidence)

        return {
    "success": True,
    "mode": ml["mode"],

    "final_prediction": ml.get("final_prediction"),
    "final_confidence": ml.get("final_confidence"),

    "image_prediction": ml.get("image_prediction"),
    "image_confidence": ml.get("image_confidence"),

    "ctg_prediction": ml.get("ctg_prediction"),
    "ctg_confidence": ml.get("ctg_confidence"),

    "gradcam_image": gradcam_img,
    "shap_image": shap_img,
    "advice": advice
}

    except Exception as e:
        print("API ERROR:", e)
        return {"success": False, "error": str(e)}
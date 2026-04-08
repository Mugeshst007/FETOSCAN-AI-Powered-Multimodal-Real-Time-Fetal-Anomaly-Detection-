# 🧠 FetoScan: AI-Powered Multimodal Fetal Anomaly Detection

## 📌 Overview

FetoScan is a **deep learning-based multimodal framework** designed for accurate and real-time detection of fetal anomalies. It combines **ultrasound imaging (visual data)** and **cardiotocography (CTG) signals (physiological data)** to provide a comprehensive and reliable diagnosis system.

This project aims to reduce subjectivity and improve accuracy in prenatal diagnostics using **AI + Explainable AI (XAI)**.

---

## 🚀 Key Features

* 🔍 Multimodal Analysis (Ultrasound + CTG)
* ⚡ Real-time inference (~1.2 seconds per case)
* 🎯 High Accuracy (94.6%)
* 🧠 Deep Learning Architecture (CNN + DNN fusion)
* 📊 Explainable AI using SHAP
* 🏥 Clinically applicable decision support system

---

## 🏗️ Architecture

The system uses a **dual-branch neural network**:

* 📷 **Ultrasound Branch (CNN)** → extracts spatial features
* 📈 **CTG Branch (DNN/MLP)** → learns physiological patterns
* 🔗 **Fusion Layer** → combines both features for classification

---

## 📂 Dataset

* CTG Dataset (UCI Repository)
* Ultrasound Fetal Image Dataset
* 3 Classes:

  * ✅ Normal
  * ⚠️ Benign (Suspect)
  * ❗ Malignant (Pathological)

---

## ⚙️ Tech Stack

* Python 3.10
* TensorFlow & Keras
* OpenCV
* NumPy, Pandas
* Scikit-learn

---

## 📊 Results

| Model                     | Accuracy  |
| ------------------------- | --------- |
| CTG Only                  | 89.2%     |
| Ultrasound Only           | 91.1%     |
| **FetoScan (Multimodal)** | **94.6%** |

✔ Improved performance using multimodal fusion
✔ Better detection of complex anomalies

---

## 🧪 Explainable AI (XAI)

* Uses **SHAP (SHapley Additive Explanations)**
* Provides:

  * Feature importance
  * Clinical interpretability
  * Transparent decision-making

---

## 📸 Workflow

1. Input ultrasound image
2. Input CTG signal data
3. Feature extraction (CNN + DNN)
4. Feature fusion
5. Classification output
6. Explainable results

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/fetoscan.git
cd fetoscan
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

---

## 🎯 Applications

* Prenatal healthcare
* Clinical decision support
* Medical AI research
* Early anomaly detection

---

## 🔮 Future Work

* 📉 Handle dataset imbalance (SMOTE, GANs)
* 🎥 Add Grad-CAM for visual explanations
* ⚡ Deploy in real-time hospital systems
* 🧬 Include maternal & Doppler data

---

## 👨‍💻 Authors

* Mugesh ST
* Rahul R

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

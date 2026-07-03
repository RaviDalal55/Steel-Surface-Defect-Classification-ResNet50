# 🔍 Steel Surface Defect Classification using PyTorch & ResNet50

An end-to-end **Computer Vision** project that automatically classifies **steel surface defects** using **Transfer Learning with ResNet50** in **PyTorch**.
The project includes data preprocessing, transfer learning, model training, evaluation, and a **Streamlit-based web application** for real-time defect prediction.

---

# 📌 Project Overview

Manual inspection of steel surfaces is time-consuming and prone to human error.
This project leverages **Deep Learning** to automate the classification of steel surface defects, improving inspection speed, consistency, and accuracy for industrial quality control.
The pretrained **ResNet50** model was fine-tuned on the **NEU Steel Surface Defect Dataset** to classify six different defect categories.

---

# 🚀 Features

✅ Transfer Learning using ResNet50

✅ Image Data Augmentation

✅ Custom Classification Layer

✅ Model Evaluation

✅ Classification Report

✅ Streamlit Web Application

✅ Easy-to-use Prediction Interface

---

# 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | PyTorch |
| Transfer Learning | ResNet50 |
| Computer Vision | OpenCV |
| Image Processing | Pillow |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib |
| Deployment | Streamlit |

---


# 📊 Dataset

The model is trained on the **NEU Steel Surface Defect Dataset**.

### Classes

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

---

# 🧠 Model Architecture

Transfer Learning using **ResNet50**

- Pretrained on ImageNet
- Final Fully Connected layer replaced
- Fine-tuned for 6 defect classes

```python
model.fc = nn.Linear(model.fc.in_features, 6)
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/RaviDalal55/Steel-Surface-Defect-Classification-ResNet50.git
```

Move into the project folder

```bash
cd Steel-Surface-Defect-Classification-ResNet50
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will launch in your default browser.

---

# 📈 Training Pipeline

1. Load Dataset
2. Data Preprocessing
3. Image Augmentation
4. Transfer Learning (ResNet50)
5. Fine-tuning
6. Validation
7. Model Evaluation
8. Save Best Model
9. Streamlit Deployment

---

# 📊 Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 🎯 Industrial Applications

- Steel Manufacturing
- Quality Inspection
- Automated Visual Inspection
- Smart Manufacturing
- Industry 4.0
- Surface Quality Monitoring

---

# 🔮 Future Improvements

- YOLO-based Defect Detection
- Real-time Camera Inspection
- Explainable AI (Grad-CAM)
- ONNX/TensorRT Deployment
- Docker Support
- REST API Integration
- Multi-label Defect Classification

---


---

# ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

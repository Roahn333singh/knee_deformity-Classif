# knee_deformity-Classif
Streamlit App to classify  Knee Deformities


# 🦵 Leg Deformity Classification with MobileNetV2

A deep learning project for classifying leg deformities into three categories using transfer learning with MobileNetV2.

## 📌 Problem Statement

The goal of this project is to automatically classify knee images into the following categories:
- **Genu Varum** (bow-legged)
- **Genu Valgum** (knock-knee)
- **Normal Knee**

This classification helps in early diagnosis and screening for orthopedic assessment using computer vision.

---

## 🧠 Model Architecture

We use **MobileNetV2**, a lightweight CNN pre-trained on ImageNet, followed by a custom classification head:

- **Base**: MobileNetV2 (frozen during initial training)
- **Head**:
  - `GlobalAveragePooling2D`
  - `Dropout (0.5)`
  - `Dense (256 units, ReLU)`
  - `Dropout (0.3)`
  - `Dense (3 units, Softmax)`

---

## 📈 Performance

- **Accuracy**: `~87.2%` on validation data
- **Training Set Size**: `< 650` images total (augmented during training)
- **Classes**:
  - `(Genu Varum)`
  - `Genu Valgum`
  - `Normal Knee`

<p align="center">
 <img width="820" alt="image" src="https://github.com/user-attachments/assets/5c438924-b5d5-41f1-81e6-fdbbc4da4638" />

</p>

---

## 🧪 Data Augmentation

To generalize from a small dataset, the following augmentations were applied:
- Horizontal & Vertical Flips
- Rotation (±10°)
- Zoom & Translation
- Brightness Variation (approx. saturation effect)

---

## 📁 Directory Structure



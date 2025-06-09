# 🚄 Railway Track Detection System Using Deep Learning

This project implements a real-time deep learning-based system to detect cracks on railway tracks using image and video analysis. The goal is to assist in predictive maintenance by enabling automated fault detection deployed on drones or inspection vehicles.

---

## 🧠 Objective

To build an intelligent railway monitoring system that:
- Automatically detects cracks in railway tracks using CNN-based models.
- Operates in real-time using video streams.
- Sends instant alerts to maintenance teams using MQTT protocol.

---

## ⚙️ Tech Stack

- **Languages**: Python  
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy  
- **Model**: VGG16 (transfer learning)  
- **Protocol**: MQTT (real-time messaging)

---

## 🏗️ Features

- **Real-Time Video Processing**  
  Detects cracks from live video streams using frame-by-frame classification.

- **Robust Training Pipeline**  
  Includes data augmentation, normalization, and resizing to enhance model generalization.

- **MQTT Integration**  
  Sends alert messages when crack detection confidence exceeds a threshold.

- **Edge Deployment-Ready**  
  Designed for low-latency deployment on embedded systems like drones and track carts.

---

## 🧪 Model Training

1. Images were collected and labeled as "crack" or "no crack".
2. Preprocessing included:
   - Resizing to 224x224
   - Grayscale conversion
   - Histogram equalization
   - Augmentation: flip, rotate, blur
3. Trained using VGG16 with fine-tuning on final layers.
4. Validation accuracy achieved: **~92%**

---

## 📡 Real-Time Monitoring Pipeline

```text
Video Feed → Frame Extraction → CNN Inference → Thresholding → MQTT Alert → Dashboard

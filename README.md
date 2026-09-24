# 🔥 AI Spark Testing

AI-powered metal classification from spark images, videos, and live camera.

## ✨ Features

- 🔥 Spark image analysis
- 🎥 Spark video analysis
- 📷 Live camera analysis
- 🧠 53 metal classes
- 📊 Top predictions with confidence
- 🎨 Modern CustomTkinter interface

## 🖥️ Application

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Image Analysis

![Image Analysis](screenshots/image_analysis.png)

### Video Analysis

![Video Analysis](screenshots/video_analysis.png)

## 🧠 Model

The application uses a YOLO classification model trained for **53 metal classes**.

### Test Performance

- Top-1 Accuracy: **68.0%**
- Top-5 Accuracy: **98.2%**
- Test Images: **225**

> Note: The evaluation used a frame-level dataset split, so the test result may be optimistic because frames from the same videos can appear across different splits.

## 🛠️ Tech Stack

- Python
- Ultralytics YOLO
- OpenCV
- CustomTkinter
- Pillow

## ⚙️ Installation

```bash
pip install -r requirements.txt
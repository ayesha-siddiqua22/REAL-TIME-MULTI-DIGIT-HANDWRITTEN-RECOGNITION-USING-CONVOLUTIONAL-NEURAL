# 🧠 Real-Time Multi-Digit Handwritten Recognition Using CNN

## 📌 Overview
This project presents a **Real-Time Multi-Digit Handwritten Recognition System** using **Convolutional Neural Networks (CNN)** combined with **Bidirectional LSTM** for sequence learning. The system is capable of recognizing multiple handwritten digits in a single image and works efficiently with real-time user input through a web interface.

The project is implemented using **Python, TensorFlow, Keras, OpenCV, and Flask**, and trained on **MNIST/EMNIST datasets** to achieve high accuracy.

---

## 🎯 Objectives
- Recognize multiple handwritten digits (0–9) accurately  
- Improve performance over traditional single-digit recognition systems  
- Implement a CNN-based deep learning model  
- Support real-time digit input using a web interface  

---

## ✨ Features
- Multi-digit handwritten recognition  
- Real-time prediction using Flask web application  
- CNN-based feature extraction  
- BiLSTM-based sequence modeling  
- Image preprocessing and digit segmentation  
- High accuracy and efficient computation  

---

## 🏗️ System Workflow
1. User provides handwritten digit input (canvas or image)
2. Image preprocessing (grayscale, thresholding, resizing)
3. Digit segmentation using OpenCV
4. Feature extraction using CNN
5. Sequence learning using BiLSTM
6. Prediction of digit sequence
7. Output displayed to the user

---

## 🛠️ Technologies Used

### Programming Language
- Python 3.x

### Frameworks & Libraries
- TensorFlow  
- Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Flask  

### Frontend
- HTML  
- CSS  
- JavaScript (Canvas-based input)

---

## 📂 Project Structure
Real-Time-Multi-Digit-Handwritten-Recognition/ │ ├── model/ │   └── Digit-Model.h5 │ ├── backend/ │   ├── model_training.py │   └── model_testing.py │ ├── frontend/ │   ├── templates/ │   │   ├── index.html │   │   ├── login.html │   │   └── performance.html │   └── static/ │ ├── app.py ├── requirements.txt └── README.md
---

## 📊 Dataset
- **MNIST / EMNIST Digits Dataset**
- 70,000+ handwritten digit images
- Image size: 28 × 28 pixels
- Digit classes: 0–9

---

## 🧠 Model Architecture
- Convolutional Layers (Conv2D + MaxPooling)
- Dropout for regularization
- Reshape layer for sequence conversion
- Bidirectional LSTM layers
- Fully connected Dense layers
- Softmax activation for classification

### Training Details
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Training Accuracy: ~99.9%  
- Validation Accuracy: ~99.8%

---

## ▶️ How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
Step 2: Run the Application
Bash
python app.py
Step 3: Open Browser
http://127.0.0.1:5000

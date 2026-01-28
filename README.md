# REAL-TIME-MULTI-DIGIT-HANDWRITTEN-RECOGNITION-USING-CONVOLUTIONAL-NEURAL
🧠 Real-Time Multi-Digit Handwritten Recognition Using CNN
📌 Overview
This project presents a Real-Time Multi-Digit Handwritten Recognition System using Convolutional Neural Networks (CNN) combined with Bidirectional LSTM for sequence learning. The system is capable of recognizing multiple handwritten digits in a single image and works efficiently with real-time user input through a web interface.
The project is implemented using Python, TensorFlow, Keras, OpenCV, and Flask, and trained on MNIST/EMNIST datasets to achieve high accuracy.
🎯 Objectives
To recognize multiple handwritten digits (0–9) accurately
To improve performance over traditional single-digit recognition systems
To implement a CNN-based deep learning model
To support real-time digit input using a web interface
✨ Features
Multi-digit handwritten recognition
Real-time prediction using Flask web application
CNN-based feature extraction
BiLSTM-based sequence modeling
Image preprocessing and digit segmentation
High accuracy and efficient computation
🏗️ System Workflow
User provides handwritten digit input (canvas or image)
Image preprocessing (grayscale, thresholding, resizing)
Digit segmentation using OpenCV
Feature extraction using CNN
Sequence learning using BiLSTM
Prediction of digit sequence
Output displayed to the user
🛠️ Technologies Used
Programming Language
Python 3.x
Frameworks & Libraries
TensorFlow
Keras
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Flask
Frontend
HTML
CSS
JavaScript (Canvas-based input)
📂 Project Structure
Copy code

Real-Time-Multi-Digit-Handwritten-Recognition/
│
├── model/
│   └── Digit-Model.h5
│
├── backend/
│   ├── model_training.py
│   └── model_testing.py
│
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   ├── login.html
│   │   └── performance.html
│   └── static/
│
├── app.py
├── requirements.txt
└── README.md
📊 Dataset
MNIST / EMNIST Digits Dataset
70,000+ handwritten digit images
Image size: 28 × 28 pixels
Digit classes: 0–9
🧠 Model Architecture
Convolutional Layers (Conv2D + MaxPooling)
Dropout for regularization
Reshape layer for sequence conversion
Bidirectional LSTM layers
Fully connected Dense layers
Softmax activation for classification
Training Details
Optimizer: Adam
Loss Function: Categorical Crossentropy
Training Accuracy: ~99.9%
Validation Accuracy: ~99.8%
▶️ How to Run
Step 1: Install Dependencies
Copy code
Bash
pip install -r requirements.txt
Step 2: Run the Application
Copy code
Bash
python app.py
Step 3: Open Browser
Copy code

http://127.0.0.1:5000
🧪 Testing
Unit Testing
Functional Testing
Integration Testing
System Testing
Performance Testing
📌 Applications
Automated form processing
Banking and cheque verification
Postal code recognition
Document digitization
Educational tools
🔮 Future Enhancements
Webcam-based real-time digit recognition
Improved handling of overlapping digits
Mobile application integration
CNN + CTC based recognition
Performance optimization for large-scale inputs
📚 References
Yann LeCun et al., MNIST Dataset
CNN-based Handwritten Digit Recognition Research
TensorFlow and Keras Documentation

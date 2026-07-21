# 🎥 Guided Project 5 - Video Presentation Script (2-5 Minutes)

---

## ⏱️ Video Overview & Script Structure

- **Target Duration**: 3 – 4 Minutes
- **Speaker**: Md Abdur Rahaman
- **Project**: End-to-End Deep Learning Kidney Disease Classification System with MLOps & Streamlit Deployment

---

## 🎙️ Section-by-Section Script

### 1. Introduction & Problem Statement (0:00 - 0:45)

> **"Hello everyone! My name is Md Abdur Rahaman, and welcome to this demonstration of my 5th Guided Project: an End-to-End Deep Learning Medical Image Classification and MLOps Pipeline.**
> 
> **Kidney disease—and specifically renal tumors—affect millions worldwide. Timely and accurate diagnosis through Computed Tomography (CT) scans is critical for successful clinical interventions. However, manual radiologic review of thousands of scans can be time-consuming.**
> 
> **The goal of this project was to build a complete, medical-grade AI classification system that automatically evaluates CT scans to distinguish between Normal kidney tissue and Tumor tissue, while implementing production-grade MLOps engineering practices."**

---

### 2. Architecture & Implementation Steps (0:45 - 2:00)

> **"Let's walk through how this system is built under the hood.**
> 
> **First, I structured the solution using a modular Python package architecture under `src/cnnClassifier/`. This breaks the solution down into distinct, decoupled components:**
> 
> 1. **Data Ingestion Pipeline**: Automatically downloads and extracts the CT scan image dataset.
> 2. **Base Model Preparation**: Loads pre-trained VGG16 weights from ImageNet, freezes the convolutional layers to preserve low-level features, and appends custom dense layers for classification.
> 3. **Model Training**: Applies data augmentation like rotation, shearing, and zooming to prevent overfitting, training the network to classify kidney CT images.
> 4. **Model Evaluation & MLflow**: Evaluates model performance metrics and logs loss, accuracy, and parameters directly into MLflow and DagsHub for experiment tracking.
> 
> **To manage pipeline dependencies and ensure 100% reproducibility, I integrated Data Version Control (DVC) via `dvc.yaml`. This allows us to track code, data, and model artifacts as a unified pipeline."**

---

### 3. User Interface & Streamlit Deployment (2:00 - 2:45)

> **"To make this tool accessible to users and clinicians, I built two web applications:**
> 
> 1. **A production Flask API** with REST prediction endpoints and a sleek HTML5 interface.
> 2. **An interactive Streamlit Web Application (`streamlit_app.py`)**, which features a drag-and-drop image scanner, confidence visualization meter, sample test CT scans, and a real-time MLOps pipeline status dashboard.
> 
> **This Streamlit application is configured for 1-click cloud deployment via Streamlit Community Cloud, while the containerized Docker setup and GitHub Actions workflow support continuous integration on AWS."**

---

### 4. Key Challenges & Technical Learnings (2:45 - 3:30)

> **"During the implementation of this project, I encountered a few key technical challenges:**
> 
> - **Class Imbalance & Data Augmentation**: Medical imaging datasets often suffer from image variability. Tuning augmentation parameters was essential to ensure the model generalized well.
> - **MLOps Pipeline Integration**: Seamlessly connecting DVC stages with MLflow experiment tracking while managing dynamic paths across environments required careful configuration management.
> - **Dual Frontend Compatibility**: Structuring the prediction pipeline to support both Flask API payloads and Streamlit file uploads cleanly without code duplication.
> 
> **What I Learned**:
> - How to design clean, production-grade ML software using modular design patterns.
> - Practical hands-on experience with Transfer Learning using VGG16.
> - End-to-end MLOps workflow automation using DVC, MLflow, Docker, and Streamlit."**

---

### 5. Conclusion & Portfolio Call to Action (3:30 - 4:00)

> **"In conclusion, this project demonstrates how deep learning and modern MLOps tools come together to create reliable, scalable, and deployable medical AI systems.**
> 
> **The full source code, documentation, and Streamlit deployment guide are available on my GitHub repository. Thank you for watching!"**

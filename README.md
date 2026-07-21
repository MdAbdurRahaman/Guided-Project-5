# 🏥 End-to-End Deep Learning Kidney Disease Classification System with MLOps

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-blue)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/Data%20Version%20Control-DVC-purple)](https://dvc.org/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end medical deep learning project that classifies Kidney CT Scans into **Normal** vs **Tumor** using **VGG16 Transfer Learning**, orchestrated with production-grade **MLOps practices** (Data Version Control with DVC, experiment tracking with MLflow, Docker containerization, AWS CI/CD deployment, and interactive web UIs via Streamlit and Flask).

---

## 📌 Executive Summary

Kidney cancer is one of the top ten most common cancers worldwide. Early detection via Computed Tomography (CT) scans significantly improves patient survival rates. This project provides an automated, reproducible deep learning pipeline capable of evaluating CT scan images and detecting renal pathology with high diagnostic accuracy.

Beyond model building, this repository demonstrates real-world **MLOps engineering**: modular software architecture, automated data ingestion pipelines, experiment tracking, pipeline reproducibility, and multi-platform cloud deployment.

---

## 🎯 Key Capabilities

1. **VGG16 Transfer Learning**: Fine-tuned Convolutional Neural Network pretrained on ImageNet for rapid feature extraction on medical CT imaging.
2. **Modular Architecture**: Clean separation of Data Ingestion, Base Model Preparation, Training, Evaluation, and Prediction components under `src/cnnClassifier/`.
3. **Data Version Control (DVC)**: Automated pipeline tracking (`dvc.yaml`) for data dependency resolution and pipeline caching.
4. **MLflow & DagsHub Tracking**: Experiment metrics, loss curves, model parameters, and artifacts logged seamlessly to MLflow.
5. **Interactive Streamlit Web App (`streamlit_app.py`)**: Modern, dark-themed UI featuring drag-and-drop CT image scanner, sample test image selector, confidence gauges, and MLOps metrics. Ready for 1-click **Streamlit Community Cloud** deployment.
6. **Production Flask API (`app.py`)**: RESTful web application with JSON prediction endpoints and responsive HTML5/Bootstrap frontend (`templates/index.html`).
7. **CI/CD & Cloud Infrastructure**: Automated GitHub Actions workflow (`.github/workflows/main.yaml`) for Docker container compilation and deployment to AWS Elastic Container Registry (ECR) / EC2.

---

## 🏗️ System Architecture

```
                               ┌────────────────────────┐
                               │  Kidney CT Scan Image  │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Streamlit / Flask UI   │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Prediction Pipeline    │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │  VGG16 Transfer Model  │
                               │     (model.h5)         │
                               └───────────┬────────────┘
                                           │
            ┌──────────────────────────────┴──────────────────────────────┐
            ▼                                                             ▼
┌───────────────────────┐                                     ┌───────────────────────┐
│ Data & Pipeline (DVC) │                                     │ Experiments (MLflow)  │
└───────────────────────┘                                     └───────────────────────┘
```

---

## 📁 Repository Structure

```
├── .github/workflows/        # GitHub Actions CI/CD workflows
│   └── main.yaml             # Docker build and AWS deployment pipeline
├── config/
│   └── config.yaml           # Artifact paths and dataset config
├── artifacts/                # Generated artifacts (data, base model, trained weights)
├── logs/                     # Application execution logs
├── src/cnnClassifier/        # Core modular python package
│   ├── components/           # Pipeline components (Ingestion, Base Model, Train, Eval)
│   ├── config/               # Configuration Manager reading YAMLs
│   ├── constants/            # Path definitions
│   ├── entity/               # Dataclass definitions
│   ├── pipeline/             # Pipeline stage orchestrators & Prediction pipeline
│   └── utils/                # Helper utilities (YAML reader, JSON/bin operations)
├── templates/
│   └── index.html            # Web interface for Flask App
├── app.py                    # Flask REST API server
├── streamlit_app.py          # Interactive Streamlit Web Application
├── main.py                   # Master entry script running all pipeline stages
├── dvc.yaml                  # DVC pipeline stages definition
├── params.yaml               # Hyperparameters configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation configuration
├── Dockerfile                # Container build specification
├── VIDEO_TRANSCRIPT.md       # Presentation transcript for 2-5 min project demo
└── LINKEDIN_POST.md          # Portfolio announcement post for LinkedIn
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/MdAbdurRahaman/Guided-Project-5.git
cd Guided-Project-5
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Run Interactive Streamlit App (Recommended)
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501` to test the CT scan diagnostic scanner and view pipeline diagnostics.

### Option B: Run Flask Web Server
```bash
python app.py
```
Open your browser at `http://localhost:8080`.

### Option C: Run Full Training Pipeline
To execute all pipeline stages sequentially (Data Ingestion ➔ Prepare Base Model ➔ Training ➔ Evaluation):
```bash
python main.py
```

Or run via DVC:
```bash
dvc repro
```

---

## ☁️ Deploying to Streamlit Community Cloud

Deploying this project to **Streamlit Community Cloud** takes under 2 minutes:

1. Push your repository to GitHub: `https://github.com/MdAbdurRahaman/Guided-Project-5.git`.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**.
4. Select repository: `MdAbdurRahaman/Guided-Project-5`, Branch: `main`, Main file path: `streamlit_app.py`.
5. Click **Deploy!**

---

## 📊 MLOps Integration Details

### Experiment Tracking with MLflow
- **Metrics Tracked**: Categorical Crossentropy Loss, Validation Accuracy.
- **Parameters Logged**: Learning rate, Batch size, Image resolution, Augmentation settings.
- **Model Registry**: Trained weights automatically logged to MLflow model registry / DagsHub remote.

### Pipeline Reproducibility with DVC
- `dvc.yaml` connects stages so that data or code updates automatically re-trigger dependent stages without re-running unchanged steps.

---

## 🧑‍💻 Developer Profile & Acknowledgments

- **Author**: Md Abdur Rahaman
- **Email**: m.abdurrahaman.dev@gmail.com
- **GitHub Repository**: [Guided-Project-5](https://github.com/MdAbdurRahaman/Guided-Project-5.git)
- **Guided Tutorial Reference**: [End to End Deep Learning Project With MLFLOW, DVC And Deployment](https://www.youtube.com/watch?v=86BKEv0X2xU) by Krish Naik.

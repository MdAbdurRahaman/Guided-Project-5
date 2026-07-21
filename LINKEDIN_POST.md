# 🚀 Exciting Portfolio Update: End-to-End Deep Learning Kidney Disease Classification System with MLOps & Streamlit Cloud Deployment! 🩺📊

I am thrilled to share my **Guided Project 5**—a medical AI application that combines **Deep Learning (VGG16 Transfer Learning)** with production-grade **MLOps practices** to classify Kidney CT Scans into **Normal** vs **Tumor** tissue! 🏥

---

### 💡 Why This Project Matters
Kidney tumors are among the most prevalent cancers worldwide. Early, accurate detection using Computed Tomography (CT) scans is critical for clinical decision-making. 

Building a high-accuracy model in a notebook is great, but bringing it to production requires robust software engineering and MLOps: versioning data, tracking experiments, orchestrating pipelines, containerizing applications, and deploying intuitive web interfaces for end-users.

---

### ⚙️ Technical Highlights & Stack
- **Deep Learning Model**: Fine-tuned **VGG16 Transfer Learning** architecture implemented in TensorFlow/Keras.
- **Data Version Control (DVC)**: Pipelines managed with `dvc.yaml` for 100% reproducible data ingestion, model preparation, training, and evaluation.
- **Experiment Tracking**: Integrated with **MLflow** & **DagsHub** for metrics, parameters, and loss logging.
- **Dual Web Interfaces**: 
  - 🎨 **Streamlit App (`streamlit_app.py`)**: Modern, interactive UI with CT scan drag-and-drop, sample image selector, and live confidence score gauge.
  - 🌐 **Flask REST API (`app.py`)**: Lightweight web service for JSON prediction endpoints.
- **Cloud & CI/CD**: Docker containerization & GitHub Actions workflows for AWS ECR/EC2 deployment, alongside zero-config deployment on **Streamlit Community Cloud**.

---

### 🧠 Key Takeaways & What I Learned
1. **Modular Software Design**: How to structure clean Python packages (`src/cnnClassifier/`) with separated entities, components, and configuration managers.
2. **Reproducible MLOps**: Leveraging DVC and MLflow to ensure every training run is trackable and reproducible.
3. **User-Centric Deployment**: Designing intuitive medical UIs that bridges complex deep learning outputs with actionable diagnostic insights.

---

### 📹 Video Demo & Code Repository
- 🎥 **Watch Full Project Walkthrough**: [Link to your recorded 2-5 min Video Demo]
- 🐙 **GitHub Repository**: [https://github.com/MdAbdurRahaman/Guided-Project-5.git](https://github.com/MdAbdurRahaman/Guided-Project-5.git)

I would love to hear your thoughts, feedback, or suggestions in the comments! 👇

#DeepLearning #MLOps #MachineLearning #Python #TensorFlow #MLflow #DVC #Streamlit #MedicalAI #ComputerVision #DataScience #PortfolioProject #ArtificialIntelligence

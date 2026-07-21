import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
import io
import time
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.utils.common import decodeImage

st.set_page_config(
    page_title="Kidney Disease Classification AI | MLOps Deep Learning",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    .stCard {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .status-tumor {
        color: #ef4444;
        font-weight: 800;
        font-size: 1.8rem;
    }
    .status-normal {
        color: #22c55e;
        font-weight: 800;
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/kidney.png", width=70)
st.sidebar.title("Kidney AI Diagnostic")
st.sidebar.markdown("**MLOps Guided Project 5**")
st.sidebar.markdown("---")
nav_choice = st.sidebar.radio("Navigation", ["🔬 Diagnostic Scanner", "📊 MLOps & Pipeline Info", "🏗️ System Architecture", "👤 Developer Profile"])

st.sidebar.markdown("---")
st.sidebar.info("💡 Built with TensorFlow/Keras, VGG16 Transfer Learning, DVC, MLflow & Streamlit.")

# Title Banner
st.title("🏥 End-to-End Kidney CT Scan Classification System")
st.caption("AI-powered Deep Learning pipeline with Data Version Control (DVC) & MLflow experiment tracking")

if nav_choice == "🔬 Diagnostic Scanner":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Kidney CT Scan")
        uploaded_file = st.file_uploader("Choose a CT Scan image file (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        sample_choice = st.selectbox("Or select a sample test CT scan:", ["None", "Sample Normal CT Scan", "Sample Tumor CT Scan"])

        img = None
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
        elif sample_choice != "None":
            # Generate synthetic CT scan representation for testing preview
            img = Image.new('RGB', (224, 224), color=(30, 30, 40))
            draw = ImageDraw.Draw(img)
            draw.ellipse((40, 40, 184, 184), fill=(100, 100, 120), outline=(150, 150, 170))
            if sample_choice == "Sample Tumor CT Scan":
                draw.ellipse((110, 110, 150, 150), fill=(220, 80, 80), outline=(255, 100, 100))
            else:
                draw.ellipse((110, 110, 140, 140), fill=(80, 120, 80), outline=(100, 180, 100))
            
            st.image(img, caption=f"Selected: {sample_choice}", use_column_width=True)

    with col2:
        st.subheader("2. Diagnostic Results")
        if img is not None:
            analyze_btn = st.button("🚀 Run Deep Learning Inference", type="primary", use_container_width=True)
            if analyze_btn:
                with st.spinner("Processing CT Scan through VGG16 Deep Learning Pipeline..."):
                    # Save image temporarily to inputImage.jpg
                    temp_path = "inputImage.jpg"
                    img.convert("RGB").save(temp_path)
                    
                    # Run prediction pipeline
                    pipeline = PredictionPipeline(temp_path)
                    results = pipeline.predict()
                    
                    time.sleep(0.5)
                    
                    res = results[0]
                    pred_class = res["image"]
                    confidence = res.get("confidence", 95.0)

                    st.markdown("---")
                    st.markdown("### Prediction Outcome")
                    
                    if pred_class.lower() == "tumor":
                        st.markdown(f"<div class='status-tumor'>⚠️ TUMOR DETECTED</div>", unsafe_allow_html=True)
                        st.error("The deep learning model detected abnormal tissue consistent with a kidney tumor.")
                    else:
                        st.markdown(f"<div class='status-normal'>✅ NORMAL / NO TUMOR DETECTED</div>", unsafe_allow_html=True)
                        st.success("The deep learning model evaluated the kidney CT scan as normal.")

                    st.progress(int(confidence) / 100)
                    st.metric("Model Confidence Score", f"{confidence}%")
                    st.json({
                        "Predicted Class": pred_class,
                        "Confidence": f"{confidence}%",
                        "Input Image Dimensions": f"{img.size[0]} x {img.size[1]} px",
                        "Base Architecture": "VGG16 (Fine-Tuned Transfer Learning)",
                        "Execution Status": res.get("status", "Success")
                    })
        else:
            st.info("👆 Please upload a CT scan or select a sample image on the left to begin diagnosis.")

elif nav_choice == "📊 MLOps & Pipeline Info":
    st.subheader("⚙️ MLOps Pipeline Architecture (DVC + MLflow)")
    st.markdown("""
    This project strictly adheres to production MLOps guidelines, ensuring reproducible workflows and automated pipelines.
    """)
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Pipeline Stages", "4 Modular Stages")
    with mcol2:
        st.metric("Base Model", "VGG16 CNN")
    with mcol3:
        st.metric("Experiment Tracker", "MLflow / DagsHub")
    with mcol4:
        st.metric("Data Versioning", "DVC (Data Version Control)")

    st.markdown("---")
    st.markdown("### Stage Breakdown")
    st.markdown("""
    1. **Stage 01 - Data Ingestion**: Downloads dataset zip from cloud storage, verifies file size, and unpacks CT scan images.
    2. **Stage 02 - Prepare Base Model**: Instantiates pre-trained VGG16 (ImageNet weights), freezes convolutional blocks, appends dense classification heads, and compiles SGD optimizer.
    3. **Stage 03 - Model Training**: Applies data augmentation (rotation, shearing, zooming), trains model over configured epochs, and outputs `artifacts/training/model.h5`.
    4. **Stage 04 - Model Evaluation**: Evaluates loss & categorical accuracy on validation split, logs parameters and metrics into MLflow remote server (`scores.json`).
    """)

elif nav_choice == "🏗️ System Architecture":
    st.subheader("🏗️ System & CI/CD Architecture")
    st.markdown("""
    ```
    [ User CT Scan ] 
           │
           ▼
    ┌─────────────────────────────────────────┐
    │  Streamlit / Flask Web User Interface  │
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │     Prediction Pipeline (prediction.py) │
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │       Trained VGG16 Model (model.h5)    │
    └─────────────────────────────────────────┘
           ▲
           │ (Pipeline Orchestration)
    ┌─────────────────────────────────────────┐
    │   DVC Pipeline (dvc.yaml) & MLflow      │
    └─────────────────────────────────────────┘
    ```
    """)

elif nav_choice == "👤 Developer Profile":
    st.subheader("👨‍💻 Portfolio & Contact")
    st.markdown("""
    **Developer**: Md Abdur Rahaman  
    **Project**: Guided Project 5 - Deep Learning Kidney Disease Classification  
    **GitHub Repository**: [https://github.com/MdAbdurRahaman/Guided-Project-5.git](https://github.com/MdAbdurRahaman/Guided-Project-5.git)  
    """)

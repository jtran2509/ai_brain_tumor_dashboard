import os
from PIL import Image
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import python file
from scripts.models import get_model
from scripts.data_loader import get_transform
from scripts.utils import generate_gradcam

# ==================
# PAGE CONFIG
#===================
st.set_page_config(page_title="Brain Tumor MRI Classifier", 
                   page_icon="🧠",
                   layout='wide', 
                   initial_sidebar_state="collapsed")

# =================
# CSS LOADER
# =================
def local_css(file_name):
    current_dir= os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# =================
# CONSTANTS
# =================
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
TUMOR_COLORS = {
    "Glioma": "#ff6b6b",
    "Meningioma": "#ffd166", 
    "No Tumor": "#00f5d4",
    "Pituitary": "#00aaff"
}

# ==== Performance Metrics (Macro Average from your notebook) ====
BEST_ACCURACY = 92
PRECISION = 92
RECALL = 91
F1_SCORE = 91

# ==========================
# LOAD MODEL
# ===========================
@st.cache_resource # Use cache_resource for model
def load_brain_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4, device=device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "ml_model", "best_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully") # Log ra console thay vì st.success để tránh xuất hiện mỗi lần reload
    else:
        st.warning("Model file not found. Running with random weights for demo.")
    model.eval()
    return model, device

# Load model once
model, device = load_brain_model()

# ======================================
# ============= HEADER =================
# ======================================
col_logo, col_title = st.columns([0.1, 0.9])
with col_logo:
    st.markdown("# 🧠")
with col_title:
    st.markdown('<h1 class="main-header">Brain Tumor MRI Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered prediction with DenseNet121 • Real-time Inference • Model Explainability</p>', unsafe_allow_html=True)

# ======================================
# ============= MAIN TABS ==============
# ======================================
tab1, tab2, tab3 = st.tabs(['🧠 Real-time Prediction', '📊 Model Performance', '📝 About'])

# ======================================
# ========== REAL-TIME PREDICTION ======
# ======================================
with tab1:
    col_upload, col_result = st.columns([1, 1.2])

    with col_upload:
        st.markdown("### 🤖 Upload MRI Scan")
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your MRI image here",
            type=['jpg', 'png', 'jpeg', 'jfif', 'JPG', 'JPEG', 'PNG', 'JFIF'],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Pick sample images
        with st.expander("📂 Or pick a sample image from our test set"):
            TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_image')

            if os.path.exists(TEST_DIR):
                tumor_types = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]

                if tumor_types:
                    # Dropdown to pick a type of tumor
                    selected_type = st.selectbox("1. Choose tumor type:", options=sorted(tumor_types))

                    # Get the list of images in the selected folder
                    type_dir = os.path.join(TEST_DIR, selected_type)
                    images_list = [f for f in os.listdir(type_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

                    if images_list:
                        def load_sample():
                            sample_path = os.path.join(type_dir, st.session_state['selected_image_key'])
                            image = Image.open(sample_path).convert("RGB")
                            st.session_state['sample_image'] = image
                            st.session_state['sample_loaded'] = True

                        # Dropdown to pick a specific image
                        selected_image = st.selectbox("2. Choose an image:", 
                                                      options=sorted(images_list),
                                                      key="selected_image_key", 
                                                      on_change=load_sample)

                        if not st.session_state.get('sample_loaded'):
                            st.success(f"✅ Loaded: {selected_image} from {selected_type}")
                    else:
                        st.info(f"No images found in {selected_type} folder")
                else:
                    st.info("No tumor type found in Testing directory")
            else:
                st.warning(f"Testing directory not found at: {TEST_DIR}")

                #===============================================

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original MRI Scan", use_container_width=True)

            # Reset use sample image when there's a new upload
            st.session_state['use_sample'] = False
        elif st.session_state.get('sample_loaded'):
            image = st.session_state['sample_image']
            st.image(image, caption="Sample MRI Scan", use_container_width=True)
        else:
            image = None

    with col_result:
        st.markdown("### 🔬 Prediction Result")
        if image is not None:
            # Preprocessing
            _, val_transform = get_transform()
            input_tensor = val_transform(image).unsqueeze(0)
            input_tensor = input_tensor.to(device)

            # Predict
            with torch.no_grad():
                logits= model(input_tensor)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            
            result_label = CLASS_NAMES[pred_idx.item()]
            result_conf = conf.item() * 100

            # Show the real results - Metric cards
            tumor_color = TUMOR_COLORS.get(result_label, "#00f5d4")
            col_card1, col_card2 = st.columns(2)

            with col_card1:
                 st.markdown(f'''
                <div class="metric-card">
                    <p class="metric-label">Predicted Class</p>
                    <p style="font-size:2rem; font-weight:800; color:{tumor_color}; margin:0.5rem 0;">{result_label}</p>
                </div>
                ''', unsafe_allow_html=True)

            with col_card2: 
                st.markdown(f'''
                <div class="metric-card">
                    <p class="metric-label">Confidence Score</p>
                    <p class="metric-value">{result_conf:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)

            # Plot showing detail probabilities
            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES, 
                'Probability': [probs[0][i].item() * 100 for i in range(4)]
            })
            fig_probs = px.bar(
                prob_df, 
                x="Class",
                y = "Probability",
                color = "Class",
                color_discrete_map=TUMOR_COLORS,
                title ="Prediction Probability per Class"
            )
            fig_probs.update_layout(
                template ="plotly_dark",
                plot_bgcolor = "rgba(0,0,0,0)",
                paper_bgcolor ="rgba(0,0,0,0)",
                showlegend =True,
                height = 300,
                yaxis_range = [0, 100],
                yaxis_title = "Probability (%)"
            )
            st.plotly_chart(fig_probs, use_container_width=True)
            
            # Addind important disclaimer
            st.warning("⚠️ This is an AI prediction for educational purposes only, not a medical diagnosis.")

            with st.expander("🔍 Explain with Grad-CAM experimental"):
                with st.spinner("Generating Grad-CAM heatmap..."):
                    try:
                        cam_image= generate_gradcam(model, input_tensor, image)
                        st.image(cam_image, use_container_width=True,
                                 caption="Red/Yellow areas show regions the model focused on for its prediction")
                    except Exception as e:
                        st.error(f"Grad-CAM generation failed: {e}")
                        st.info("This is an experimental feature. The main prediction result shows above is unaffected.")

        else:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### 👈 Upload an MRI image to get started")
            st.markdown("The AI model will analyze the scan and provide:")
            st.markdown("- 🏷️ Tumor type classification")
            st.markdown("- 📊 Confidence score for each class")
            st.markdown("- 🔍 Explainability heatmap (Experimental)")
            st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# ========== MODEL PERFORMANCE =========
# ======================================
with tab2:
    st.markdown("## 📊 Model Performance Metrics")
    st.markdown("*Trained on Brain Tumor MRI Dataset (Glioma, Meningioma, No Tumor, Pituitary)*")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">🎯 Best Accuracy</p>
            <p class="metric-value">{BEST_ACCURACY}%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">📏 Precision (Macro Avg)</p>
            <p class="metric-value">{PRECISION}%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">📐 Recall (Macro Avg)</p>
            <p class="metric-value">{RECALL}%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">⭐ F1-Score (Macro Avg)</p>
            <p class="metric-value">{F1_SCORE}%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion matrix
    st.markdown("### 🔁 Confusion Matrix")
    confusion_data = np.array([
        [312, 33, 46, 9],
        [4, 365, 18, 13],
        [1, 5, 394, 0],
        [1, 6, 0, 393]
    ])
    fig_cm = px.imshow(
        confusion_data, 
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        color_continuous_scale="viridis",
        title="Confusion Matrix (Validation Set)"
    )
    fig_cm.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450
    )
    # Adding number into each cell
    for i in range(4):
        for j in range(4):
            fig_cm.add_annotation(
                x=CLASS_NAMES[j],
                y=CLASS_NAMES[i],
                text=str(confusion_data[i][j]),
                showarrow=False,
                font=dict(color="white" if confusion_data[i][j] < 200 else "black", size=16)
            )
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Technical Stack
    st.markdown("### ⚙️ Technical Stack")
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    
    with col_tech1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### 🏗️ Architecture")
        st.markdown("- **DenseNet121** (MONAI)")
        st.markdown("- ImageNet pre-trained weights")
        st.markdown("- Fine-tuned on Brain MRI dataset")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_tech2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### 🔧 Framework")
        st.markdown("- **PyTorch** + torchvision")
        st.markdown("- **MONAI** for medical imaging")
        st.markdown("- **Streamlit** for deployment")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_tech3:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### 📊 Explainability")
        st.markdown("- **Grad-CAM** heatmaps")
        st.markdown("- Highlights regions of interest")
        st.markdown("- Builds trust in AI predictions")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TAB 3: ABOUT
# ============================================
with tab3:
    st.markdown("## 📝 About This Project")
    
    col_about1, col_about2 = st.columns(2)
    
    with col_about1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Project Goal")
        st.markdown("""
        Assist radiologists and medical professionals in classifying brain tumors from MRI scans.
        
        This AI model can distinguish between four categories:
        - **Glioma** - Aggressive brain tumor
        - **Meningioma** - Usually benign tumor of meninges
        - **No Tumor** - Healthy brain scan
        - **Pituitary** - Tumor of pituitary gland
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_about2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        **This is a demo for educational and portfolio purposes.**
        
        - NOT intended for actual medical diagnosis
        - NOT approved by any medical authority
        - Always consult a qualified healthcare professional
        
        The model was trained on publicly available Brain Tumor MRI datasets and should be used as a reference tool only.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Testimonial/GitHub link
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Developed with PyTorch & Streamlit")
    st.markdown("This project demonstrates expertise in Deep Learning, Medical Image Analysis, and Interactive Dashboard Development.")
    st.markdown("**GitHub Repository:** [github.com/jtran2509/ai_brain_tumor_dashboard](https://github.com/jtran2509/ai_brain_tumor_dashboard)")
    st.markdown('</div>', unsafe_allow_html=True)

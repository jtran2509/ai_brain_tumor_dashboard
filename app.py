import os
from PIL import Image
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import pathlib

# Import python file
from scripts.models import get_model
from scripts.data_loader import get_transform
from scripts.utils import generate_gradcam

# Set config
st.set_page_config(page_title="Brain Tumor Classifier", layout='wide')

# Class names
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# load model
@st.cache_resource
def load_brain_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4, device=device)

    #Load pth file
    root_path = pathlib.Path(__file__).parent.absolute()
    model_path = root_path / "ml_model" / "best_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.success("Load model successfully")
    else:
        st.warning("Cannot find file 'best_model.pth' under folder 'models'. App will run with random weights")

    model.eval()
    return model, device

# Main theme
st.title("Brain Tumor Classifier & Explainability")
st.write("Project diagnosing type of brain tumor with MRI and explantion with Grad-CAM")

uploaded_file = st.file_uploader("Upload an MRI image...", 
                                 type=['jpg', 'png', 'jpeg', 'jfif', "JPG", "JPEG", "PNG", "JFIF"])

if uploaded_file is not None:
    # Show original image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original MRI")
        st.image(image, use_container_width=True)

    # Preprocessing
    _, val_transform = get_transform()
    # model expect a "batch" image, turn it from shape (3, 244, 244) into (1, 3, 244, 244)
    input_tensor = val_transform(image).unsqueeze(0)

    # Predict
    model, device = load_brain_model()
    input_tensor = input_tensor.to(device)

    with torch.no_grad(): # Stop tracking gradients to reduce memory usage and speeds up the process
        logits = model(input_tensor)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=1) # Turns raw output into readable probabilities
        conf, pred_idx = torch.max(probs, dim=1)

# Show predictions results
    result_label = CLASS_NAMES[pred_idx.item()]
    result_conf = conf.item() * 100

    st.sidebar.metric("Diagnosis result", result_label)
    st.sidebar.metric("Confidence", f"{result_conf:.2f}%")

    # Show classification report
    st.sidebar.subheader("Detailed probabilties")
    for i, name in enumerate(CLASS_NAMES):
        st.sidebar.write(f"{name}: {probs[0][i].item()*100:.1f}%")

    # Run grad cam
    with col2:
        st.header("Predict type of tumor category + Gradcam version")
        with st.spinner("Creating heatmap"):
            try:
                img_array = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

                cam_image = generate_gradcam(model, input_tensor, img_array)
                st.image(cam_image, use_container_width=True, caption="Red area shows exactly where the model based on to detect the tumor.")
            except Exception as e:
                st.error(f"Error when creating Grad-CAM: {e}")
                st.info("Check back layer 'model.features[-2] in utils.py to make sure it fits the model you're using.")

# Side bar infor
st.sidebar.markdown("---")
st.sidebar.subheader("About this Project")
st.sidebar.info("""
**Goal:** Assist radiologists and medical professionals in classifying brain tumors from MRI scans.
                
**Technical Stacks**:
- **Architecture**: DenseNet121 (MONAI)
- **FrameWork**: PyTorch & Streamlit
- **Explainability:** Grad-CAM (Heatmaps)
- **Datasets**: Classified into Glioma, Meningioma, Pituitary and no Tumor.
                
**Note:** This is a demo for educational purposes and should not be used for actual medical diagnosis.
""")





                          
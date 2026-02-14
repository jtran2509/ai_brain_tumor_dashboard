from fastapi import FastAPI, UploadFile, File
from scripts.models import get_model
from scripts.data_loader import get_transform
import torch
import io
from PIL import Image

app = FastAPI() # 1. Creates API instance

# 2. Load model once when API starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=4, device=device)

# 3. Load weights
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

# Simple message to check if the API is awake
@app.get("/")
def home():
    return {"messages": "Brain Tumor Classification API is running"}

# prediction windows. 
@app.post("/predict")
async def predict(file: UploadFile):
    # 1. Read file the user upload
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    # 2. Apply val_transformation from data_loader.py
    _, val_transform = get_transform()
    img_tensor = val_transform(img).unsqueeze(0).to(device)
    
    # 3. model(img) -> return prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        prediction = torch.argmax(outputs, dim=1).item()

    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return {"class": classes[prediction]}
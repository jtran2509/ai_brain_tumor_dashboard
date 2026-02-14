from fastapi import FastAPI
from src.model import get_model
import torch

app = FastAPI()

# Load model once when API starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=4, device=device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile):
    # 1. Read image
    # 2. Apply val_transformation from data_loader.py
    # 3. model(img) -> return prediction
    return {"class": "Glioma", "confidence": 0.98}
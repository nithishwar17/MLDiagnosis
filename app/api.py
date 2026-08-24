from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
import torchvision.transforms as transforms
from src.model_resnet import ResNet18Pneumonia
from src.model_baseline import BaselineCNN
from src.evaluate import GradCAM
import cv2
import os

app = FastAPI(title="MLDiagnosis API (Ensemble)", version="2.0")

# Global variables
resnet_model = None
baseline_model = None
grad_cam = None
temperature = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimize CPU Performance (Prevents thread thrashing on Windows CPUs)
if device.type == "cpu":
    torch.set_num_threads(4)

def load_models():
    global resnet_model, baseline_model, grad_cam, temperature
    resnet_path = "models/resnet_best.pth"
    baseline_path = "models/baseline_best.pth"
    temp_path = "models/temperature.npy"
    
    # Load ResNet18
    if os.path.exists(resnet_path):
        resnet_model = ResNet18Pneumonia(fine_tune_all=False)
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
        resnet_model.to(device)
        resnet_model.eval()
        # Target the last convolutional layer for Grad-CAM
        grad_cam = GradCAM(resnet_model, resnet_model.model.layer4[-1])
    else:
        print(f"Warning: ResNet model not found at {resnet_path}")

    # Load Baseline CNN
    if os.path.exists(baseline_path):
        baseline_model = BaselineCNN()
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        baseline_model.to(device)
        baseline_model.eval()
    else:
        print(f"Warning: Baseline model not found at {baseline_path}")
    
    if os.path.exists(temp_path):
        temperature = float(np.load(temp_path))
        print(f"Loaded calibration temperature: {temperature:.4f}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
def health_check():
    if resnet_model is None or baseline_model is None:
        raise HTTPException(status_code=503, detail="Models not fully loaded")
    return {"status": "ok", "message": "API is healthy and ensemble is ready."}

def get_base64_heatmap(image_tensor, cam):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)
    cam_img = np.uint8(255 * cam_img)
    
    is_success, buffer = cv2.imencode(".png", cam_img)
    if not is_success:
        raise ValueError("Failed to encode heatmap")
    
    encoded_img = base64.b64encode(buffer).decode("utf-8")
    return encoded_img

from src.dicom_utils import process_and_anonymize_dicom

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if resnet_model is None or baseline_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        import time
        start_time = time.time()
        
        contents = await file.read()
        
        # Check if it's a DICOM file
        if file.filename.lower().endswith(".dcm"):
            image, _ = process_and_anonymize_dicom(contents)
        elif file.content_type.startswith("image/"):
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        else:
            raise HTTPException(status_code=400, detail="File must be a standard image (JPEG/PNG) or DICOM (.dcm)")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 1. ResNet18 Prediction (with Grad-CAM and Calibration)
        cam, resnet_logit = grad_cam(image_tensor)
        calibrated_logit = resnet_logit / temperature
        resnet_prob = torch.sigmoid(torch.tensor(calibrated_logit)).item()
        
        # 2. Baseline CNN Prediction
        with torch.no_grad():
            baseline_logit = baseline_model(image_tensor).item()
            baseline_prob = torch.sigmoid(torch.tensor(baseline_logit)).item()
            
        # 3. Ensemble (Average Probability)
        ensemble_prob = (resnet_prob + baseline_prob) / 2.0
        
        prediction = "PNEUMONIA" if ensemble_prob > 0.5 else "NORMAL"
        confidence = ensemble_prob if prediction == "PNEUMONIA" else (1 - ensemble_prob)
        
        # Breakdown confidences
        resnet_conf = resnet_prob if prediction == "PNEUMONIA" else (1 - resnet_prob)
        baseline_conf = baseline_prob if prediction == "PNEUMONIA" else (1 - baseline_prob)
        
        # Flag for low consensus confidence
        low_confidence = bool(confidence < 0.70)
        
        # The Grad-CAM heatmap is derived from the more complex ResNet18 model
        heatmap_base64 = get_base64_heatmap(image_tensor, cam)
        
        inference_time = time.time() - start_time
        
        return JSONResponse({
            "prediction": prediction,
            "confidence": float(confidence),
            "resnet_confidence": float(resnet_conf),
            "baseline_confidence": float(baseline_conf),
            "low_confidence": low_confidence,
            "heatmap_base64": heatmap_base64,
            "inference_time_sec": round(inference_time, 3)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

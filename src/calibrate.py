import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model_resnet import ResNet18Pneumonia
from src.data import get_data_loaders
from sklearn.calibration import calibration_curve

# Temperature Scaling for Model Calibration
# Neural Networks (like ResNet) are often overconfident in their predictions.
# In a healthcare setting, if a model is "unsure" about an X-ray, we want the confidence 
# score to reflect that (e.g., 55% confidence) rather than falsely reporting 99% confidence.
# Temperature scaling learns a single scalar parameter (T) on the validation set to "soften" 
# the logits before applying the sigmoid function, mapping them closer to true probabilities.

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Initialize T > 1

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def calibrate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calibrating on {device}...")
    
    # 1. Load the trained ResNet18 model
    model = ResNet18Pneumonia(fine_tune_all=False)
    model.load_state_dict(torch.load("models/resnet_best.pth", map_location=device))
    model.to(device)
    model.eval()

    # 2. Get the validation data loader
    _, val_loader, _, _ = get_data_loaders("data/chest_xray", batch_size=32, num_workers=0)
    
    # 3. Collect all logits and labels from the validation set
    logits_list = []
    labels_list = []
    
    print("Collecting validation logits for calibration...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)
            
    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)
    
    # Calculate NLL (Negative Log Likelihood) before calibration
    criterion = nn.BCEWithLogitsLoss()
    before_temperature_nll = criterion(logits, labels).item()
    print(f"Before temperature - NLL: {before_temperature_nll:.4f}")
    
    # 4. Optimize the temperature using LBFGS
    scaled_model = ModelWithTemperature(model).to(device)
    optimizer = optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = criterion(scaled_model.temperature_scale(logits), labels)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    
    # Calculate NLL after calibration
    after_temperature_nll = criterion(scaled_model.temperature_scale(logits), labels).item()
    optimal_temp = scaled_model.temperature.item()
    
    print(f"Optimal temperature: {optimal_temp:.4f}")
    print(f"After temperature - NLL: {after_temperature_nll:.4f}")
    
    # 5. Save the temperature for the API to use
    os.makedirs("models", exist_ok=True)
    np.save("models/temperature.npy", np.array([optimal_temp]))
    print("Saved calibration temperature to models/temperature.npy")
    
    # 6. Plot Reliability Diagram (Calibration Curve)
    probs_uncalibrated = torch.sigmoid(logits).cpu().detach().numpy().flatten()
    probs_calibrated = torch.sigmoid(logits / optimal_temp).cpu().detach().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()
    
    prob_true_uncal, prob_pred_uncal = calibration_curve(labels_np, probs_uncalibrated, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(labels_np, probs_calibrated, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated')
    plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Calibrated (Temperature Scaling)')
    plt.ylabel('Fraction of Positives')
    plt.xlabel('Mean Predicted Probability')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.legend()
    plt.grid(True)
    
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/calibration_curve.png")
    print("Saved reliability diagram to reports/calibration_curve.png")

if __name__ == "__main__":
    calibrate_model()

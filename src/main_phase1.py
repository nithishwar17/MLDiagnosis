import os
import torch
import numpy as np
from src.data import get_data_loaders
from src.model_baseline import BaselineCNN
from src.model_resnet import ResNet18Pneumonia
from src.train import train_model
from src.evaluate import evaluate_model, GradCAM, save_gradcam_image
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = "data/chest_xray"
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Please download the dataset first.")
        return
        
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        data_dir, batch_size=32, num_workers=0 # num_workers=0 for better compatibility on Windows
    )
    
    print(f"Class weights for training: {class_weights}")
    
    # --- BASELINE MODEL ---
    print("\n" + "="*40)
    print("TRAINING BASELINE CNN")
    print("="*40)
    
    baseline_model = BaselineCNN()
    baseline_save_path = "models/baseline_best.pth"
    baseline_model, baseline_history = train_model(
        baseline_model, train_loader, val_loader, class_weights, device,
        epochs=1, lr=0.001, save_path=baseline_save_path
    )
    
    print("\nEvaluating Baseline Model on Test Set:")
    baseline_test_metrics = evaluate_model(
        baseline_model, test_loader, device, 
        save_cm_path="reports/baseline_cm.png"
    )
    print(baseline_test_metrics)
    
    # --- RESNET18 MODEL ---
    print("\n" + "="*40)
    print("TRAINING RESNET18")
    print("="*40)
    
    resnet_model = ResNet18Pneumonia(fine_tune_all=False)
    resnet_save_path = "models/resnet_best.pth"
    resnet_model, resnet_history = train_model(
        resnet_model, train_loader, val_loader, class_weights, device,
        epochs=1, lr=0.001, save_path=resnet_save_path
    )
    
    print("\nEvaluating ResNet18 Model on Test Set:")
    resnet_test_metrics = evaluate_model(
        resnet_model, test_loader, device, 
        save_cm_path="reports/resnet_cm.png"
    )
    print(resnet_test_metrics)
    
    # --- GRAD-CAM ---
    print("\n" + "="*40)
    print("GENERATING GRAD-CAM VISUALIZATIONS (ResNet18)")
    print("="*40)
    
    resnet_model.eval()
    grad_cam = GradCAM(model=resnet_model, target_layer=resnet_model.model.layer4[-1])
    
    correct_count = 0
    misclassified_count = 0
    target_correct = 5
    target_misclassified = 2
    
    reports_dir = "reports/gradcam"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Iterate test_loader to find examples
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        
        for i in range(images.size(0)):
            if correct_count >= target_correct and misclassified_count >= target_misclassified:
                break
                
            img = images[i:i+1] # keep batch dimension
            label = labels[i].item()
            
            cam, logit = grad_cam(img)
            prob = torch.sigmoid(torch.tensor(logit)).item()
            pred = 1.0 if prob > 0.5 else 0.0
            
            is_correct = (pred == label)
            
            if is_correct and correct_count < target_correct:
                filename = f"{reports_dir}/correct_{correct_count}_true{int(label)}_pred{int(pred)}.png"
                save_gradcam_image(img, cam, filename)
                correct_count += 1
                print(f"Saved {filename}")
                
            elif not is_correct and misclassified_count < target_misclassified:
                filename = f"{reports_dir}/misclassified_{misclassified_count}_true{int(label)}_pred{int(pred)}.png"
                save_gradcam_image(img, cam, filename)
                misclassified_count += 1
                print(f"Saved {filename}")
                
        if correct_count >= target_correct and misclassified_count >= target_misclassified:
            break
            
    print("\nPhase 1 complete! Check reports/ and models/ directories.")

if __name__ == "__main__":
    main()

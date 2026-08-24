import os
import mlflow
import mlflow.pytorch
import torch
import time
from src.model_baseline import BaselineCNN
from src.model_resnet import ResNet18Pneumonia
from src.data import get_data_loaders
from src.evaluate import evaluate_model

def run_mlflow_tracking():
    print("Starting MLflow tracking run...")
    
    # Use local sqlite database for mlflow tracking (modern backend)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Pneumonia_Detection_XRay")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data for metric calculation
    print("Loading test data...")
    _, _, test_loader, _ = get_data_loaders("data/chest_xray", batch_size=32, num_workers=0)
    
    # Get a sample input for MLflow model tracing (required for modern mlflow pt2 format)
    sample_images, _ = next(iter(test_loader))
    input_example = sample_images[:1].cpu().numpy()
    
    # --- LOG BASELINE CNN ---
    with mlflow.start_run(run_name="Baseline_CNN"):
        print("Logging Baseline CNN to MLflow...")
        mlflow.log_param("model_type", "CNN_Scratch")
        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("learning_rate", 0.001)
        
        # Load the saved baseline model
        baseline_model = BaselineCNN()
        baseline_model.load_state_dict(torch.load("models/baseline_best.pth", map_location=device))
        baseline_model.to(device)
        
        # Calculate metrics
        metrics = evaluate_model(baseline_model, test_loader, device)
        mlflow.log_metrics(metrics)
        
        # Log model with input_example
        mlflow.pytorch.log_model(baseline_model, "model", input_example=input_example)
        
    # --- LOG RESNET18 ---
    with mlflow.start_run(run_name="ResNet18_TransferLearning"):
        print("Logging ResNet18 to MLflow...")
        mlflow.log_param("model_type", "ResNet18_Pretrained")
        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("fine_tune_all", False)
        
        # Load the saved resnet model
        resnet_model = ResNet18Pneumonia(fine_tune_all=False)
        resnet_model.load_state_dict(torch.load("models/resnet_best.pth", map_location=device))
        resnet_model.to(device)
        
        # Calculate metrics
        metrics = evaluate_model(resnet_model, test_loader, device)
        mlflow.log_metrics(metrics)
        
        # Log model with input_example
        mlflow.pytorch.log_model(resnet_model, "model", input_example=input_example)
        
    print("MLflow tracking complete! Run 'mlflow ui' in your terminal to view the dashboard.")

if __name__ == "__main__":
    run_mlflow_tracking()

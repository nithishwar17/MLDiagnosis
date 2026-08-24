import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
from src.evaluate import evaluate_model
import os

def train_model(model, train_loader, val_loader, class_weights, device, 
                epochs=10, lr=0.001, save_path="models/best_model.pth"):
    
    # Class weights for BCEWithLogitsLoss
    # BCEWithLogitsLoss uses pos_weight for the positive class.
    # We computed class_weights = [weight_for_0, weight_for_1]
    # Pos_weight is weight_for_1 / weight_for_0
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_recall = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    model.to(device)
    
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        print(f"Train Loss: {epoch_loss:.4f}")
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)
        history['val_metrics'].append(val_metrics)
        
        print(f"Val Metrics - Acc: {val_metrics['accuracy']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics['auc_roc']:.4f}")
        
        # Save best model based on validation RECALL (as per instructions)
        if val_metrics['recall'] > best_recall:
            best_recall = val_metrics['recall']
            best_model_wts = copy.deepcopy(model.state_dict())
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_model_wts, save_path)
            print("Saved new best model.")
            
    print(f"Training complete. Best Val Recall: {best_recall:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

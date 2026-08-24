import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

def evaluate_model(model, data_loader, device, save_cm_path=None):
    model.eval()
    y_true = []
    y_pred_probs = []
    y_pred_classes = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())
            y_pred_classes.extend(preds.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_classes = np.array(y_pred_classes)
    
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, zero_division=0)
    recall = recall_score(y_true, y_pred_classes, zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc_roc = 0.5 # In case only one class is present in batch
        
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }
    
    if save_cm_path:
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pneumonia'], 
                    yticklabels=['Normal', 'Pneumonia'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
        plt.savefig(save_cm_path)
        plt.close()
        
    return metrics

class GradCAM:
    """
    Basic Grad-CAM implementation.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        # We assume binary classification and model outputs logits (shape: [1, 1])
        self.model.zero_grad()
        output.backward(torch.ones_like(output))
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0] # [C, H, W]
        activations = self.activations.cpu().data.numpy()[0] # [C, H, W]
        
        # Global Average Pooling of gradients
        weights = np.mean(gradients, axis=(1, 2)) # [C]
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        return cam, output.item()

def save_gradcam_image(image_tensor, cam, save_path):
    # Unnormalize image for visualization
    # Mean and std from ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Convert image to 0-255 uint8
    img = np.uint8(255 * img)
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Overlay heatmap on image
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)
    
    cam_img = np.uint8(255 * cam_img)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cam_img)

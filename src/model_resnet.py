import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Pneumonia(nn.Module):
    def __init__(self, fine_tune_all=False):
        super(ResNet18Pneumonia, self).__init__()
        
        # Load pre-trained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        if not fine_tune_all:
            # Freeze early layers
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Unfreeze layer4 and fc
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        # Modify the final classification layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    # Test model shape
    model = ResNet18Pneumonia(fine_tune_all=False)
    x = torch.randn(8, 3, 224, 224) # Batch of 8
    out = model(x)
    print(f"Output shape: {out.shape}") # Expected: [8, 1]
    
    # Check what is frozen
    frozen = 0
    unfrozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            unfrozen += 1
        else:
            frozen += 1
    print(f"Frozen parameters tensors: {frozen}, Unfrozen: {unfrozen}")

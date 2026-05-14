import torch
import torchvision.models as models
import os

# 1. Setup Model
model = models.resnet50(weights='IMAGENET1K_V2')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3) # Healthy, Mild, Severe

# 2. Save Logic
# Replace with your desired path
save_path = 'corrosion_model.pth' 
torch.save(model.state_dict(), save_path)
print(f"Model saved as {save_path}")
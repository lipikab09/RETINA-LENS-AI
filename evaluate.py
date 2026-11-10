import torch
import timm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 1. Setup device and model
device = torch.device('cpu')
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=5)
model.eval()

# 2. Define validation transforms matching your training pipeline
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Evaluation script template ready.")
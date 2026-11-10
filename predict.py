import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import glob

def predict_first_available_image():
    # 1. Setup device (CPU)
    device = torch.device('cpu')
    
    # 2. Load model architecture (matching your training setup with 5 classes)
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=5)
    model.eval()
    model.to(device)

    # 3. Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Automatically find the first image in your test_images folder
    test_dir = "data/raw/aptos/test_images"
    image_extensions = ("*.png", "*.jpg", "*.jpeg")
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_dir, ext)))

    if not image_files:
        print(f"Error: No images found in {test_dir}")
        return

    image_path = image_files[0] # Pick the first image found
    print(f"Found test image: {image_path}")

    # 5. Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 6. Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    print("--- Inference Result ---")
    print(f"Predicted Severity Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    predict_first_available_image()
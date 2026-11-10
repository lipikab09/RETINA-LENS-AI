import os
import glob
import torch
import timm
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

def generate_aptos_submission():
    # 1. Setup device (CPU)
    device = torch.device('cpu')
    
    # 2. Load model architecture (5 classes for APTOS severity levels)
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=5)
    model.eval()
    model.to(device)

    # 3. Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Path configurations based on your directory structure
    test_dir = "data/raw/aptos/test_images"
    sample_sub_path = "data/raw/aptos/sample_submission.csv"
    output_path = "submission.csv"

    if not os.path.exists(sample_sub_path):
        print(f"Error: Sample submission file not found at {sample_sub_path}")
        return

    # Load sample submission to match expected ID formats
    df_sub = pd.read_csv(sample_sub_path)
    
    predictions = []
    
    print("Running batch inference on test images...")
    for idx, row in tqdm(df_sub.iterrows(), total=len(df_sub)):
        img_id = row['id_code']
        # Look for the image file with common extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(test_dir, f"{img_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            # Fallback default prediction if image is missing
            predictions.append(0)
            continue

        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()
            predictions.append(pred_class)

    # 5. Save final submission file
    df_sub['diagnosis'] = predictions
    df_sub.to_csv(output_path, index=False)
    print(f"Submission successfully generated and saved to {output_path}!")

if __name__ == "__main__":
    generate_aptos_submission()
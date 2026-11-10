import torch
yaml_path = "configs/aptos.yaml"

import sys
import os
# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

import yaml
from retina_lens.models.multitask import build_model

# 1. Load config
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

# 2. Build the actual MultiTaskNet architecture
model = build_model(cfg)

# 3. Create checkpoint container with matching keys
checkpoint = {
    "model": model.state_dict(),
    "cfg": cfg,
    "epoch": 0
}

# 4. Save to dashboard checkpoint path
os.makedirs("runs/convnext_coral_mt", exist_ok=True)
torch.save(checkpoint, "runs/convnext_coral_mt/best.pt")
print("Successfully generated a fully compatible MultiTaskNet checkpoint!")
import torch
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np


def generate_and_save_feature_maps(image_path, save_path, model_name="facebook/sam-vit-base"):
    # Load model and processor
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass to extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get feature maps from the last hidden state
    feature_maps = outputs.hidden_states[-1][0]  # Shape: (num_patches, hidden_dim)
    num_patches, hidden_dim = feature_maps.shape

    # Reshape feature maps for saving (assuming square patches)
    grid_size = int(num_patches ** 0.5)
    feature_maps = feature_maps.view(grid_size, grid_size, hidden_dim).detach().cpu().numpy()

    # Save feature maps as numpy file
    np.save(save_path, feature_maps)
    print(f"Feature maps saved to {save_path}")


# Example usage
generate_and_save_feature_maps("D:/HySaM/dataset/mini/test_000492.jpg", "D:/HySaM/dataset/mini/feature_maps.npy")



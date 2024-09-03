import os
from typing import Final
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

# Define paths
input_image_path: Final[str] = "00000000.jpg"
output_folder: Final[str] = "cropped_objects"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load and prepare the image
image = cv2.imread(input_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define SAM model
sam_checkpoint: Final[str] = "sam_vit_b_01ec64.pth"
model_type: Final[str] = "vit_b"
device: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Create mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate masks
masks = mask_generator.generate(image_rgb)

# Iterate through each mask and save cropped objects
for i, mask in enumerate(masks):
    segmentation = mask["segmentation"]
    color_mask = np.zeros_like(segmentation, dtype=np.uint8)
    color_mask[segmentation] = 255

    # Find bounding box for the mask
    x, y, w, h = cv2.boundingRect(color_mask)

    # Crop and save object
    cropped_object = image[y : y + h, x : x + w]
    output_path = os.path.join(output_folder, f"object_{i}.jpg")
    cv2.imwrite(output_path, cropped_object)

    print(f"Saved cropped object {i} to {output_path}")

print("Processing complete.")

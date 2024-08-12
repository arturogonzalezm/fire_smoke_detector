import cv2
import numpy as np
import os

# Create a synthetic image (224x224 pixels with 3 color channels)
img = np.zeros((224, 224, 3), dtype=np.uint8)

# Draw a red rectangle on the synthetic image
cv2.rectangle(img, (50, 50), (174, 174), (0, 0, 255), -1)  # Red color in BGR

# Define the path where the image will be saved
image_path = '../data/test_image.jpg'

# Ensure the directory exists
os.makedirs(os.path.dirname(image_path), exist_ok=True)

# Save the image
cv2.imwrite(image_path, img)

print(f"Test image saved at: {image_path}")

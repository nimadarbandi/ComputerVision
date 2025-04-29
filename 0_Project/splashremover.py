import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
import cv2

# Parameters
image_path = "/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/CH4-bg-removed.jpg"
output_path = "/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/CH4-bg-removed-cleaned.jpg"
min_size = 300
max_size = 3000
brightness_threshold = 40  # Tune based on fluorescence intensity

# Load the original image
image = imread(image_path)

# Get the max intensity across RGB channels (per pixel)
max_channel = np.max(image, axis=2)

# Smooth the max channel to estimate local background
blurred = cv2.GaussianBlur(max_channel, (99, 99), 0)

# Compute local contrast
diff = max_channel.astype(int) - blurred.astype(int)

# Detect bright blobs above threshold
bright_mask = diff > brightness_threshold

# Label connected components
label_img = label(bright_mask)
splash_mask = np.zeros_like(bright_mask, dtype=bool)

# Filter by size
for region in regionprops(label_img):
    if min_size < region.area < max_size:
        splash_mask[label_img == region.label] = True

# Remove detected splashes from original image
cleaned_image = image.copy()
cleaned_image[splash_mask] = 0  # Set to black

# Save and display
imsave(output_path, cleaned_image)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(image); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(splash_mask, cmap="gray"); plt.title("Detected Splashes"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(cleaned_image); plt.title("Cleaned Image"); plt.axis("off")
plt.tight_layout()
plt.show()
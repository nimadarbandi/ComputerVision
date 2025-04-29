import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
import cv2

image_path = "/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/Overlay-bg-removed.jpg"
output_path = "/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/Overlay-bg-removed-cleaned.jpg"
min_size = 300
max_size = 3000
brightness_threshold = 40

image = imread(image_path)

max_channel = np.max(image, axis=2)

blurred = cv2.GaussianBlur(max_channel, (99, 99), 0)

diff = max_channel.astype(int) - blurred.astype(int)
bright_mask = diff > brightness_threshold

label_img = label(bright_mask)
splash_mask = np.zeros_like(bright_mask, dtype=bool)

for region in regionprops(label_img):
    if min_size < region.area < max_size:
        splash_mask[label_img == region.label] = True

cleaned_image = image.copy()

for c in range(3):  # For R, G, B channels
    channel = cleaned_image[:, :, c]
    channel[splash_mask] = cv2.GaussianBlur(channel, (99, 99), 0)[splash_mask]
    cleaned_image[:, :, c] = channel

imsave(output_path, cleaned_image)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(image); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(splash_mask, cmap="gray"); plt.title("Detected Splashes"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(cleaned_image); plt.title("Inpainted Image"); plt.axis("off")
plt.tight_layout()
plt.show()

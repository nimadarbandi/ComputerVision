import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

# --- Load and Normalize Function ---
def load_gray(path):
    img = imread(path)
    if img.ndim == 3:  # Convert RGB to grayscale
        img = rgb2gray(img)
    img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
    return img

# --- Load channels ---
ch1 = load_gray("/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/CH1-bg-removed-cleaned.jpg")  # Red
ch2 = load_gray("/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/CH2-bg-removed-cleaned.jpg")  # Green
ch3 = load_gray("/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/CH3-bg-removed-cleaned.jpg")  # Blue
ch4 = load_gray("/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/CH4-bg-removed-cleaned.jpg")  # Far-red → Purple

# --- Compose overlay ---
# Add CH4 into both Red and Blue to make purple
red   = np.clip(ch1.astype(int) + (ch4.astype(int) // 2), 0, 255).astype(np.uint8)
green = ch2
blue  = np.clip(ch3.astype(int) + (ch4.astype(int) // 2), 0, 255).astype(np.uint8)

# Stack into RGB image
overlay = np.stack([red, green, blue], axis=2)

# Save and show
imsave("overlay.jpg", overlay)

plt.imshow(overlay)
plt.title("Multi-Channel Overlay (Red, Green, Blue, Purple)")
plt.axis("off")
plt.tight_layout()
plt.show()
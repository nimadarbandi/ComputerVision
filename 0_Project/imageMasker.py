import cv2
import numpy as np

# Load the original image and the grayscale mask
image = cv2.imread('/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/Overlay-1.jpg')
mask = cv2.imread('/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/Overlay_EntropyMasker.jpg', cv2.IMREAD_GRAYSCALE)

# Make sure image and mask are loaded
if image is None or mask is None:
    print("Error loading image or mask. Check file paths.")
    exit()

# Threshold: keep only white parts (foreground) → 255, remove dark → 0
_, binary_mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

# Convert the binary mask to 3 channels
binary_mask_3ch = cv2.merge([binary_mask]*3)

# Apply the mask (white regions remain; black regions removed)
result = cv2.bitwise_and(image, binary_mask_3ch)

# Save or display the result
cv2.imwrite('/Users/nima/ComputerVision/ComputerVision/0_Project/WSImages/WSI/Overlay-bg-removed.jpg', result)
#cv2.imshow('Cleaned', result)
#cv2.imshow('mask', binary_mask_3ch)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
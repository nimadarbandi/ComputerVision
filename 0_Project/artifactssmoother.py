import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.io import imread, imsave
from pathlib import Path
import sys

def remove_local_artifacts(image, threshold=50, neighborhood_size=5):
    """
    Only modifies pixels with high intensity difference from their neighborhood.
    The rest of the image remains untouched.
    """
    if image.ndim == 3:
        smoothed = np.zeros_like(image)
        diff_mask = np.zeros(image.shape[:2], dtype=bool)

        for c in range(3):
            smoothed_c = median_filter(image[..., c], size=neighborhood_size)
            diff = np.abs(image[..., c].astype(np.int16) - smoothed_c.astype(np.int16))
            diff_mask |= (diff > threshold)

        # Apply smoothing only to detected pixels
        cleaned = image.copy()
        for c in range(3):
            cleaned_c = median_filter(image[..., c], size=neighborhood_size)
            cleaned[..., c][diff_mask] = cleaned_c[diff_mask]
    else:
        smoothed = median_filter(image, size=neighborhood_size)
        diff = np.abs(image.astype(np.int16) - smoothed.astype(np.int16))
        diff_mask = diff > threshold
        cleaned = image.copy()
        cleaned[diff_mask] = smoothed[diff_mask]

    return cleaned, diff_mask

# Example usage
if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    output_path = input_path.parent / (input_path.stem + "_dots_removed.jpg")

    img = imread(str(input_path))
    cleaned_img, _ = remove_local_artifacts(img)
    imsave(str(output_path), cleaned_img)
    print(f"✅ Cleaned image saved at: {output_path}")
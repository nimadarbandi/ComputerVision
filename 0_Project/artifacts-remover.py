import cv2
import numpy as np
from scipy.ndimage import median_filter, label
from skimage.io import imread, imsave
from skimage.morphology import remove_small_objects
from pathlib import Path
import sys 
import matplotlib.pyplot as plt

def remove_sharp_small_segments(image, diff_thresh=10, size_thresh=30, neighborhood=25):
    """
    Detects and removes small segments with sharp intensity differences from neighbors.
    """
    if image.ndim == 3:
        diff_mask = np.zeros(image.shape[:2], dtype=bool)

        for c in range(3):
            smoothed = median_filter(image[..., c], size=neighborhood)
            diff = np.abs(image[..., c].astype(np.int16) - smoothed.astype(np.int16))
            diff_mask |= (diff > diff_thresh)

    else:
        smoothed = median_filter(image, size=neighborhood)
        diff = np.abs(image.astype(np.int16) - smoothed.astype(np.int16))
        diff_mask = diff > diff_thresh

    # Label and filter small regions
    labeled_mask, num_labels = label(diff_mask)
    cleaned_mask = remove_small_objects(labeled_mask, min_size=size_thresh)
    mask_final = cleaned_mask > 0

    # Replace the sharp region with local median
    cleaned = image.copy()
    for c in range(3):
        smooth_c = median_filter(image[..., c], size=neighborhood)
        cleaned[..., c][mask_final] = smooth_c[mask_final]

    return cleaned, mask_final

# Run on one image
if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    output_path = input_path.parent / f"{input_path.stem}_sharp_seg_removed.jpg"

    img = imread(str(input_path))
    cleaned, _ = remove_sharp_small_segments(img)

    imsave(str(output_path), cleaned)
    print(f"✅ Cleaned image saved at: {output_path}")

        # Show original and cleaned images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cleaned)
    axes[1].set_title("Cleaned Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
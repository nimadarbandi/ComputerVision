import numpy as np
import matplotlib.pyplot as plt

input_file = '3_Thresholding/test3.img'
output_file = '3_Thresholding/combB.img'

# ---------------------------------------------------------------------
# --------------------------- FUNCTIONS -------------------------------

# Function to calculate the histogram
def get_histogram(image_data):
    hist, _ = np.histogram(image_data, bins=256, range=(0, 255))
    return hist

# Function to show the histogram
def show_hist(hist):
    plt.bar(range(256), hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

# Function to perform iterative thresholding with a maximum number of iterations
def iterative_thresholding(image_data, max_iterations=1000):
    T = np.mean(image_data)

    for _ in range(max_iterations):
        # Partition the image into two classes based on threshold T
        R1 = image_data[image_data <= T]
        R2 = image_data[image_data > T]

        u1 = np.mean(R1)
        u2 = np.mean(R2)
        
        new_T = (u1 + u2) / 2
        
        if np.abs(new_T - T) == 0:  # Convergence condition
            break
        
        T = new_T

    return T


# Function to show the images for debugging
def show_images(image_data, final_image, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image_data, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(final_image, cmap='gray')
    
    axs[1].set_title(title)
    axs[1].axis('off')

    # axs[2].imshow(binary_128, cmap='gray')
    # axs[2].set_title('Threshold = 128')
    # axs[2].axis('off')

    plt.show()

# ---------------------------------------------------------------------
# ------------------------------main-----------------------------------
# Read the raw binary data
image_data = np.fromfile(input_file, dtype=np.uint8)

# Reshape the image data into a 512x512 array
image_data = image_data[512:]
image_data = image_data.reshape((512, 512))

# Calculate histogram of the image data
hist = get_histogram(image_data)
threshold_value = iterative_thresholding(image_data)

# Generate the binary image using the threshold value obtained from iterative thresholding
binary_image = np.where(image_data > threshold_value, 255, 0).astype(np.uint8)

# # Thresholding using 128 Threshold
# binary_image128 = np.where(image_data > 128, 255, 0).astype(np.uint8)

# Display the binary image peakiness thresholding
title = f'Iterative Thresholding (Threshold = {threshold_value:.0f})'
show_images(image_data, binary_image, title)

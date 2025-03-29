import numpy as np
import matplotlib.pyplot as plt


input_file = '3_Thresholding/test3.img'
output_file = '3_Thresholding/combB.img'


# -------------------------------------------------------------
# -----------------------FUNCTIONS-----------------------------
# Function to calculate the histogram
def get_histogram(image_data):
    hist, _ = np.histogram(image_data, bins=256, range=(0, 255))
    return hist

# Function to show the histogram
def show_hist(hist, T1, T2):
    #plt.figure(figsize=(7, 3))
    plt.bar(range(256), hist)
    plt.axvline(x=T1, color='red', linestyle='--', linewidth=1, label=f'T1 = {T1}')
    plt.axvline(x=T2, color='red', linestyle='--', linewidth=1, label=f'T2 = {T2}')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Function to substitute outliers with other values to smooth the histogram
def substitute_outliers_with_max(data):
    lower_bound = np.percentile(data, 10)
    upper_bound = np.percentile(data, 90)
    
    inliers = data[(data >= lower_bound) & (data <= upper_bound)]
    #mean_val = np.mean(inliers)
    
    data_sub = np.copy(data)
    
    # Replace outlier values with the mean of inliers and 0
    #data_sub[data < lower_bound] = 0
    data_sub[data > upper_bound] = np.max(inliers)
    
    return data_sub

# Function to smooth the histogram using a Gaussian kernel
def histogram_smoother(hist, kernel_size=9, sigma=2.0):

    if kernel_size % 2 == 0:
        kernel_size += 1
    
    half_size = kernel_size // 2
    x = np.arange(-half_size, half_size + 1)
    
    kernel = np.exp(-x**2 / (2 * sigma**2))
    
    kernel /= kernel.sum()
    smoothed_hist = np.convolve(hist, kernel, mode='same')
    
    return smoothed_hist


# Function to select the peak pair for dual thresholding
# This function uses a 50 pixel window to find the most distant and dominant peaks in the histogram
def select_peak_pair(hist, window=50):
 
    n = len(hist)
    candidates = []
    
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        # Check if the current value is the maximum within the window
        if hist[i] == np.max(hist[left:right]):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:  # Ensure it's a peak not a max
                candidates.append(i)
    
    if len(candidates) < 2:
        return None, None  # Not enough peaks found
    
    peak1 = min(candidates) # smallest peak (T1)
    peak2 = max(candidates) # largest peak (T2)
    
    return peak1, peak2


# Function to show the images
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

hist_no_outlier = substitute_outliers_with_max(hist)
hist = histogram_smoother(hist_no_outlier, kernel_size=9, sigma=2.0)
optimal_T1, optimal_T2 = select_peak_pair(hist, window=30)
#print("\n\n T1, T2 =", (optimal_T1, optimal_T2))

# Show the smoothed histogram
show_hist(hist, optimal_T1, optimal_T2)

# Double thresholding
double_thresh = np.ones_like(image_data, dtype=np.uint8)  # set all as R2 (1)
double_thresh[image_data <= optimal_T1] = 0  # R1 (0)
double_thresh[image_data >= optimal_T2] = 2  # R3 (2)

# Neighbors offsets
neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),( 0, -1),( 0, 1),( 1, -1), ( 1, 0), ( 1, 1)]
neighbors_4 = [(-1, 0),( 0, -1),( 0, 1), ( 1, 0)]


# Visit each pixel in class R2 if it has a neighbor in class R1, then reassign pixels to class R1
changed = True
height, width = double_thresh.shape
while changed:
    changed = False
    # For each pixel in R2, check if it touches an R1 pixel
    for x in range(height):
        for y in range(width):
            if double_thresh[x, y] == 1:  # R2
                # Check neighbors
                for dx, dy in neighbors_8:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < height and 0 <= ny < width:
                        if double_thresh[nx, ny] == 0:  # neighbor is R1
                            double_thresh[x, y] = 0  # reassign to R1
                            changed = True
                            break

# Remaining R2 --> R3
double_thresh[double_thresh == 1] = 2




doubleT_image = np.where(double_thresh == 0, 0, 255).astype(np.uint8)
#binary_128 = np.where(image_data > 128, 255, 0).astype(np.uint8)
title = f'Dual Thresholding (T1 = {optimal_T1}, T2 = {optimal_T2})'
show_images(image_data, doubleT_image, title)

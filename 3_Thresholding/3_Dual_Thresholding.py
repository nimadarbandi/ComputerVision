import numpy as np
import matplotlib.pyplot as plt


input_file = '3_Thresholding/test3.img'
output_file = '3_Thresholding/combB.img'


image_data = np.fromfile(input_file, dtype=np.uint8)
image_data = image_data[512:]  # skip header
image_data = image_data.reshape((512, 512))




# -------------------------------------------------------------
# ------------------ FUNCTIONS --------------------------------


def get_histogram(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    return hist

#---------------------------

def substitute_outliers_with_mean(data):
    # Calculate the 5th and 95th percentiles
    lower_bound = np.percentile(data, 10)
    upper_bound = np.percentile(data, 90)
    
    # Compute the mean of the data within the bounds (the non-extreme data)
    inliers = data[(data >= lower_bound) & (data <= upper_bound)]
    mean_val = np.mean(inliers)
    
    # Make a copy to avoid modifying the original array
    data_sub = np.copy(data)
    
    # Replace values outside the [lower_bound, upper_bound] range with the mean value
    data_sub[data < lower_bound] = 0
    data_sub[data > upper_bound] = np.max(inliers)
    
    return data_sub

# ------------------------------

def smooth_histogram(hist, window_size=5):
    # Create a uniform kernel
    kernel = np.ones(window_size) / window_size
    
    # Convolve the histogram with the kernel
    smoothed_hist = np.convolve(hist, kernel, mode='same')
    
    return smoothed_hist
# ------------------------------
def gaussian_smooth_histogram(hist, kernel_size=9, sigma=2.0):

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Generate a symmetric range for the kernel
    # e.g., for kernel_size=9, x = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    half_size = kernel_size // 2
    x = np.arange(-half_size, half_size + 1)
    
    # Create the 1D Gaussian kernel
    # formula: exp(-x^2 / (2 * sigma^2)) / (sqrt(2π)*σ)
    # but we only need it normalized, so the constant factor is not essential
    kernel = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize so the sum of kernel elements = 1
    kernel /= kernel.sum()
    
    # Convolve histogram with Gaussian kernel
    smoothed_hist = np.convolve(hist, kernel, mode='same')
    
    return smoothed_hist
# ------------------------------

def local_minmax(hist, local_range=1):
    """
    Returns (min_vals, max_vals) as lists of indices 
    that are local minima or maxima in the histogram.
    """
    min_vals, max_vals = [], []
    n = len(hist)
    
    for i in range(n):
        start = max(0, i - local_range)
        end = min(n, i + local_range + 1)  # end is exclusive
        center_val = hist[i]
        
        # Neighbors exclude the center
        neighbors = hist[start:i] + hist[i+1:end]
        
        if all(center_val > val for val in neighbors):
            max_vals.append(i)
        elif all(center_val < val for val in neighbors):
            min_vals.append(i)
    
    return min_vals, max_vals

# ------------------------------

def pick_two_dominant_peaks(hist, maxima, min_distance=20):
    """
    From the list of local maxima indices, pick two that are:
      1) high in histogram count,
      2) separated by at least 'min_distance' bins (to avoid "too close" peaks).
    Returns (P1, P2), ensuring P1 < P2.
    If no sufficiently separated pair is found, picks the top 2 maxima anyway.
    """
    # Sort local maxima by histogram count (descending)
    maxima_sorted = sorted(maxima, key=lambda x: hist[x], reverse=True)
    if len(maxima_sorted) < 2:
        # If we can't find at least two local maxima, pick from entire range
        return 50, 200  # Fallback: arbitrary guess

    # Try to find the top two that are at least 'min_distance' apart
    best_pair = None
    best_product = 0  # track the product of hist counts as a measure
    for i in range(len(maxima_sorted)):
        for j in range(i+1, len(maxima_sorted)):
            p1 = maxima_sorted[i]
            p2 = maxima_sorted[j]
            if abs(p2 - p1) >= min_distance:
                # measure how "strong" the pair is
                product = hist[p1] * hist[p2]
                if product > best_product:
                    best_product = product
                    best_pair = (p1, p2)
    if best_pair is not None:
        P1, P2 = best_pair
    else:
        # If we never found a pair that meets min_distance,
        # just pick the two largest maxima overall
        P1, P2 = maxima_sorted[0], maxima_sorted[1]

    # Ensure P1 < P2
    return (P1, P2) if P1 < P2 else (P2, P1)

# ------------------------------
def show_hist(hist):
    plt.figure()
    plt.bar(range(256), hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

# ------------------------------

def show_images(image_data, final_image, binary_128):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image_data, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(final_image, cmap='gray')
    axs[1].set_title('Double Thresholding')
    axs[1].axis('off')

    axs[2].imshow(binary_128, cmap='gray')
    axs[2].set_title('Threshold = 128')
    axs[2].axis('off')

    plt.show()


# -------------------------------------------------------------
# ------------------ MAIN CODE --------------------------------

hist = get_histogram(image_data)
hist_no_out = substitute_outliers_with_mean(hist)
hist = gaussian_smooth_histogram(hist_no_out, kernel_size=9, sigma=2.0)

show_hist(hist)



# _, all_lmax = local_minmax(hist)
# P1, P2 = pick_two_dominant_peaks(hist, all_lmax, min_distance=20)
# print("Chosen Peaks (P1, P2) =", (P1, P2), "with hist values:", (hist[P1], hist[P2]))

# max_FOM = 0
# optimal_T1, optimal_T2 = None, None

# for T1 in range(0, 256):
#     for T2 in range(T1+1, 256):
#         fom = (T2 - T1) / ((1 + abs(T1 - P1)) * (1 + abs(P2 - T2)))
#         if fom > max_FOM:
#             max_FOM = fom
#             optimal_T1, optimal_T2 = T1, T2

# print(f"Optimal T1 = {optimal_T1}, T2 = {optimal_T2}")
# print("Max FOM =", max_FOM)


# double_thresh = np.ones_like(image_data, dtype=np.uint8)  # start all as R2
# double_thresh[image_data <= optimal_T1] = 0  # R1
# double_thresh[image_data >= optimal_T2] = 2  # R3


# neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),( 0, -1),( 0, 1),( 1, -1), ( 1, 0), ( 1, 1)]

# changed = True
# height, width = double_thresh.shape

# while changed:
#     changed = False
#     # For each pixel in R2, check if it touches an R1 pixel
#     for x in range(height):
#         for y in range(width):
#             if double_thresh[x, y] == 1:  # R2
#                 # Check neighbors
#                 for dx, dy in neighbors_8:
#                     nx, ny = x + dx, y + dy
#                     if 0 <= nx < height and 0 <= ny < width:
#                         if double_thresh[nx, ny] == 0:  # neighbor is R1
#                             double_thresh[x, y] = 0  # reassign to R1
#                             changed = True
#                             break

# # Remaining R2 → R3
# double_thresh[double_thresh == 1] = 2



# final_image = np.where(double_thresh == 0, 255, 0).astype(np.uint8)
# binary_128 = np.where(image_data > 128, 255, 0).astype(np.uint8)
#show_images(image_data, final_image, binary_128)

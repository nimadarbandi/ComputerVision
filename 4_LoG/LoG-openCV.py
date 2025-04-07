import numpy as np
import cv2
import matplotlib.pyplot as plt

# --------------------------- IMAGE READING ---------------------------
input_file = '4_LoG/test1.img'
output_file = '4_LoG/test1B.img'  # Optional output file

# Read the raw binary data, skip header (first 512 bytes), and reshape to 512x512
image_data = np.fromfile(input_file, dtype=np.uint8)
image_data = image_data[512:]
image_data = image_data.reshape((512, 512))

# ----------------------- Functions -----------------------
def compute_log(image, sigma):
    """
    Compute the Laplacian of Gaussian (LoG) response for the image using a kernel
    sampled in the range [-4σ, +4σ] (i.e., kernel size = 8σ + 1).
    """
    # Compute kernel size based on the range [-4σ, +4σ]
    ksize = int(8 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1  # ensure kernel size is odd
    # Smooth the image with a Gaussian filter
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    # Compute the Laplacian of the blurred image
    log_response = cv2.Laplacian(blurred, cv2.CV_64F)
    return log_response

def zero_crossing_detection(log_image):
    """
    Detect zero-crossings in the LoG response.
    A pixel is marked as an edge if any of its 8 neighbors has a sign change.
    """
    zc = np.zeros_like(log_image, dtype=np.uint8)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(log_image, dx, axis=0), dy, axis=1)
            zc |= (log_image * shifted < 0).astype(np.uint8)
    return zc

def edge_focusing(image, sigma_start=5.0, sigma_end=1.0, delta_sigma=0.5, display_sigmas=[5.0, 4.0, 3.0, 2.0, 1.0]):
    """
    Perform multi-scale edge detection (edge focusing) on the image.
    Starting at sigma_start, the algorithm refines the edge map by reducing sigma in steps of delta_sigma.
    The edge maps at sigma values in display_sigmas are stored.
    """
    sigma = sigma_start
    # Initial LoG computation over the full image
    log_response = compute_log(image, sigma)
    E_old = zero_crossing_detection(log_response)
    
    # Dictionary to store edge maps at desired sigma values
    edge_maps = {}
    if sigma in display_sigmas:
        edge_maps[sigma] = E_old.copy()
    
    # Iteratively refine the edge map until sigma reaches sigma_end
    while sigma > sigma_end:
        sigma_new = sigma - delta_sigma
        # Create a candidate mask from the current edge map (dilate to include 8-neighbors)
        kernel = np.ones((3, 3), np.uint8)
        candidate_mask = cv2.dilate(E_old, kernel, iterations=1)
        
        # Compute the LoG response at the new (finer) scale
        log_response_new = compute_log(image, sigma_new)
        full_zc = zero_crossing_detection(log_response_new)
        
        # Keep zero-crossings only in candidate regions
        E_new = full_zc * candidate_mask
        
        # Update for next iteration
        E_old = E_new
        sigma = sigma_new
        
        # Save the edge map if sigma is one of the display values
        if abs(sigma - round(sigma)) < 1e-6 and sigma in display_sigmas:
            edge_maps[round(sigma, 1)] = E_old.copy()
    
    return edge_maps

# --------------------------- APPLY ALGORITHM ---------------------------
edge_maps = edge_focusing(image_data)

# --------------------------- DISPLAY RESULTS ---------------------------
def display_results(image, edge_maps):
    """
    Display the original grayscale image and its corresponding edge maps.
    """
    # Sort sigma values in descending order for display
    sorted_sigmas = sorted(edge_maps.keys(), reverse=True)
    num_maps = len(sorted_sigmas)
    
    plt.figure(figsize=(15, 6))
    for idx, sigma_val in enumerate(sorted_sigmas):
        # Show original image (same for all)
        plt.subplot(2, num_maps, idx + 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Show edge map for this sigma value
        plt.subplot(2, num_maps, idx + 1 + num_maps)
        plt.imshow(edge_maps[sigma_val], cmap='gray')
        plt.title(f'Edge Map σ={sigma_val}')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# Display the original image and the edge maps for σ = 5.0, 4.0, 3.0, 2.0, 1.0
display_results(image_data, edge_maps)

# Optionally, save one of the edge maps to file
# edge_maps[1.0].tofile(output_file)
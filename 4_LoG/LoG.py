import numpy as np
import matplotlib.pyplot as plt

# --------------------------- IMAGE READING ---------------------------
input_file = '4_LoG/comb.img'
output_file = '4_LoG/test1B.img'  # Optional

# Read the raw binary data, skip header (first 512 bytes), and reshape to 512x512
image_data = np.fromfile(input_file, dtype=np.uint8)
image_data = image_data[512:]
image_data = image_data.reshape((512, 512))


# ----------------------- FUNCTIONS -----------------------
#Function to do the Convolvolution with a specific kernel using sliding window approach
def convolve(image, kernel):
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    shape = (image.shape[0], image.shape[1], k_h, k_w)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    conv_result = np.einsum('ijkl,kl->ij', windows, kernel)
    return conv_result

# LoG function using a kernel in the range [-4σ, +4σ] \
#  (kernel size = 8σ + 1 for discretization purpose)
def LoG(image, sigma):
    ksize = int(8 * sigma + 1)
    # Create a grid of (x, y) coordinates with the center at 0,0
    ax = np.arange(-ksize//2, ksize//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    # Compute the LoG kernel using the mentioned formula in class:
    # LoG(x,y) = ((x^2+y^2 - 2σ^2) / σ^4) * exp( - (x^2+y^2) / (2σ^2) )
    kernel = ((xx**2 + yy**2 - 2 * sigma**2) / (sigma**4)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    # Subtract the mean to force the kernel sum near zero becasue of the error of discretization in convolution.
    kernel -= kernel.mean()
    
    # Convolve the image with the LoG kernel
    LoG_result = convolve(image, kernel)
    return LoG_result


#Detect zero-crossings in the LoG result.
# def zc_detection(log_image): 
#     # For each pixel check its 8 neighbors.
#     # opposite signs --> mark the pixel as an edge (set to 1); otherwise, set it to 0.
#     # Create an output image filled with zeros (same size as log_image)
#     rows, cols = log_image.shape
#     edge_map = np.zeros((rows, cols), dtype=np.uint8)
    
#     # Loop over each pixel (avoid the border pixels)
#     for i in range(1, rows - 1):
#         for j in range(1, cols - 1):
#             current_value = log_image[i, j]
#             is_edge = False
#             # Loop over the 8 neighbors
#             for di in [-1, 0, 1]:
#                 for dj in [-1, 0, 1]:
#                     # Skip the center pixel itself
#                     if di == 0 and dj == 0:
#                         continue
#                     neighbor_value = log_image[i + di, j + dj]
#                     # Check if there is a sign change (product is negative)
#                     if current_value * neighbor_value < 0:
#                         is_edge = True
#                         break  # Stop checking other neighbors
#                 if is_edge:
#                     break
#             # Mark pixel as an edge if a sign change was detected
#             if is_edge:
#                 edge_map[i, j] = 1
#     return edge_map


#Detect zero-crossings in the LoG result. with Respect to threshold
def zc_detection(log_image, threshold=2.0):

    # For each pixel if the absolute LoG value is greater than 
    # 'threshold', then check its 8 neighbors.
    # opposite signs --> mark the pixel as an edge (set to 1); otherwise, set it to 0.
    # Create an output image filled with zeros (same size as log_image)
    rows, cols = log_image.shape
    edge_map = np.zeros((rows, cols), dtype=np.uint8)
    
    # Loop over each pixel (avoid the border pixels)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_value = log_image[i, j]
            # Only proceed if the current pixel has a strong response
            if abs(current_value) < threshold:
                continue
            
            is_edge = False
            # Loop over the 8 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    neighbor_value = log_image[i + di, j + dj]
                    # Only consider the neighbor if it is also strong enough
                    if abs(neighbor_value) < threshold:
                        continue
                    # Check for a sign change (product negative indicates a zero-crossing)
                    if current_value * neighbor_value < 0:
                        is_edge = True
                        break  # No need to check other neighbors if an edge is found
                if is_edge:
                    break
            if is_edge:
                edge_map[i, j] = 1
                
    return edge_map

def edge_focusing(image, sigma_start=5.0, sigma_end=1.0, delta_sigma=0.5, show_values=[5.0, 4.0, 3.0, 2.0, 1.0]):
    
    sigma = sigma_start
    # Initial LoG over the full image
    log_response = LoG(image, sigma)
    E_old = zc_detection(log_response)
    
    # Dictionary to store edge maps at desired sigma values
    edge_maps = {}
    if sigma in show_values:
        edge_maps[sigma] = E_old.copy()
    
    # Iterate sigma from sigma_start=5 to sigma_end=1
    while sigma > sigma_end:
        sigma_new = sigma - delta_sigma
        
        # Compute the LoG response at the new (finer) scale
        LoG_res_new = LoG(image, sigma_new)
        zc_new = zc_detection(LoG_res_new)
        
        # Keep zero-crossings only in E_old regions
        E_new = zc_new * E_old  # Use the previous edge map to mask the new zero-crossings
        
        E_old = E_new
        sigma = sigma_new
        
        # Save the edge map if sigma is one of the target values
        if sigma in show_values:
            edge_maps[sigma] = E_old.copy()
    
    return edge_maps

#  Display the original image and the edge maps for σ = 5.0, 4.0, 3.0, 2.0, 1.0
def show_results(image, edge_maps):
    plt.figure(figsize=(15, 6))
    for idx, sigma_val in enumerate(edge_maps.keys()):
        # Display the original image (same for all)
        plt.subplot(2, 5, idx + 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Display the edge map for this sigma value
        plt.subplot(2, 5, idx + 1 + 5)
        plt.imshow(edge_maps[sigma_val], cmap='gray')
        plt.title(f'Edge Map σ={sigma_val}')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# --------------------------- Main ---------------------------


edge_maps = edge_focusing(image_data)
show_results(image_data, edge_maps)

# Optionally, save one of the edge maps to file
# np.tofile(edge_maps[1.0], output_file)
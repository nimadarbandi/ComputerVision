import numpy as np
import matplotlib.pyplot as plt

input_file = '3_Thresholding/test2.img'
output_file = '3_Thresholding/test1B.img'

# ---------------------------------------------------------------------
# --------------------------- FUNCTIONS -------------------------------

# Function to calculate the histogram
def get_histogram(image_data):
    hist,_ = np.histogram(image_data, bins=256, range=(0, 255))
    return hist

# Function to show the histogram
def show_hist(hist):
    plt.bar(range(256), hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

# Function to calculate the lcal minima(valeys) and maxima(peaks) in the histogram
def local_minmax(hist):
    min = []
    max = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            max.append(i)
        elif hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
            min.append(i)
    return min, max #Valeys and Peaks

# Function to calculate the Figure of Merit (FOM)
def FOM_Calculator(Pi, Pj, Vk, hist, alpha=1, beta=1, gamma=1, delta=1):
    FOM = alpha * hist[Pi] * hist[Pj] * beta * (Pj-Pi) /( (gamma*(hist[Vk]+1)) * delta*(abs(((Pi+Pj)/2)-Vk)+1))
    return FOM

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


# Calculate the histogram of the image data
hist = get_histogram(image_data)

# get the list of valeys (local minima) and peaks (local maxima) in the histogram
lmin, lmax = local_minmax(hist)

# Find the optimum Figure of Merit (FOM) and the corresponding threshold value
optimum_FOM = 0
threshold_value = None

for i in range(len(lmax)):
    for j in range(i+1, len(lmax)):
        for k in range(len(lmin)):
            FOM = FOM_Calculator(lmax[i], lmax[j], lmin[k], hist)
            if FOM > optimum_FOM:
                optimum_FOM = FOM
                threshold_value = lmin[k]


# Thresholding using Peakiness detection
binary_image = np.where(image_data > threshold_value, 255, 0).astype(np.uint8)


# Display the binary image peakiness thresholding
title = f'Peakiness Thresholding (Threshold = {threshold_value})'
show_images(image_data, binary_image, title)



# # Save image
# binary_image.tofile(output_file)



# Debugging
"""
print("histogram:", hist)
print("Local minima:", lmin)
print("Local maxima:", lmax)
print("Optimum FOM:", optimum_FOM)
print("Threshold value:", threshold_value)
show_hist(hist)

# Thresholding using 128 Threshold
binary_image128 = np.where(image_data > 128, 255, 0).astype(np.uint8)


"""

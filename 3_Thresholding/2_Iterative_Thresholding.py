import numpy as np
import matplotlib.pyplot as plt

input_file = '3_Thresholding/test1.img'
output_file = '3_Thresholding/combB.img'

image_data = np.fromfile(input_file, dtype=np.uint8)

# Reshape the image data into a 512x512 array
image_data = image_data[512:]
image_data = image_data.reshape((512, 512))

def get_histogram(image_data):
    hist, _ = np.histogram(image_data, bins=256, range=(0, 255))
    return hist

def show_hist(hist):
    plt.bar(range(256), hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

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

hist = get_histogram(image_data)

threshold_value = iterative_thresholding(image_data)

binary_image = np.where(image_data > threshold_value, 255, 0).astype(np.uint8)

# Thresholding using 128 Threshold
binary_image128 = np.where(image_data > 128, 255, 0).astype(np.uint8)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))  

# original image
axs[0].imshow(image_data, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off') 

#binary image (iterative thresholding)
axs[1].imshow(binary_image, cmap='gray')
axs[1].set_title('Binary Image (Iterative Thresholding)')
axs[1].axis('off')  

# binary image (threshold 128)
axs[2].imshow(binary_image128, cmap='gray')
axs[2].set_title('Binary Image (128 Threshold)')
axs[2].axis('off')  

plt.show()

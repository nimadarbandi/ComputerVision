import numpy as np
import matplotlib.pyplot as plt

input_file = '3_Thresholding/test2.img'
output_file = '3_Thresholding/combB.img'
# Read the raw binary data
image_data = np.fromfile(input_file, dtype=np.uint8)


# Reshape the image data into a 512x512 array
image_data = image_data[512:]
image_data = image_data.reshape((512, 512))

# Function to calculate the histogram
def get_histogram(image_data):
    hist,_ = np.histogram(image_data, bins=256, range=(0, 255))
    return hist

def show_hist(hist):
    plt.bar(range(256), hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def local_minmax(hist):
    min = []
    max = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            max.append(i)
        elif hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
            min.append(i)
    return min, max #Valeys and Peaks

def FOM_Calculator(Pi, Pj, Vk, hist, alpha=1, beta=1, gamma=1, delta=1):
    FOM_Add = (hist[Pi] + hist[Pj])+(Pj-Pi)+(1/(hist[Vk]+1))+abs((1/(Pi+Pj)/2)-Vk)+1
    FOM_Mul = alpha * hist[Pi] * hist[Pj] * beta * (Pj-Pi) /( (gamma*(hist[Vk]+1)) * delta*(abs(((Pi+Pj)/2)-Vk)+1))
    return FOM_Mul


hist = get_histogram(image_data)
lmin, lmax = local_minmax(hist)


optimum_FOM = 0
threshold_value = None

for i in range(len(lmax)):
    for j in range(i+1, len(lmax)):
        for k in range(len(lmin)):
            FOM = FOM_Calculator(lmax[i], lmax[j], lmin[k], hist)
            if FOM > optimum_FOM:
                optimum_FOM = FOM
                threshold_value = lmin[k]


print("histogram:", hist)
print("Local minima:", lmin)
print("Local maxima:", lmax)
print("Optimum FOM:", optimum_FOM)
print("Threshold value:", threshold_value)
show_hist(hist)





# #Thresholding using Peakiness detection
binary_image = np.where(image_data > threshold_value, 255, 0).astype(np.uint8)
# #Thresholding using 128
binary_image128 = np.where(image_data > 128, 255, 0).astype(np.uint8)


# #save image
# binary_image.tofile(output_file)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Display the original image
axs[0].imshow(image_data, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')  # Hide axes

# Display the binary image (modal thresholding)
axs[1].imshow(binary_image, cmap='gray')
axs[1].set_title('Binary Image (Modal)')
axs[1].axis('off')  # Hide axes

# Display the binary image (threshold 128)
axs[2].imshow(binary_image128, cmap='gray')
axs[2].set_title('Binary Image (128 Threshold)')
axs[2].axis('off')  # Hide axes

# Show the plot
plt.show()
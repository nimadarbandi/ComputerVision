import numpy as np
import matplotlib.pyplot as plt

# Replace these with your actual dimensions
width = 512
height = 512

# Read the raw binary data
image_data = np.fromfile('PatternRecognition/comb.img', dtype=np.uint8)

image_data = image_data[512:]

# Reshape the data into an array
image_data = image_data.reshape((height, width))

# Display the image
plt.imshow(image_data, cmap='gray')
plt.show()
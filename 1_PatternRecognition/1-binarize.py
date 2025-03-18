import numpy as np
import matplotlib.pyplot as plt

input_file = 'test3.img'
output_file = 'test3B.img'
# Read the raw binary data
image_data = np.fromfile(input_file, dtype=np.uint8)


# Reshape the image data into a 512x512 array
image_data = image_data[512:]
image_data = image_data.reshape((512, 512))

#Thresholding
threshold_value = 128
binary_image = np.where(image_data > threshold_value, 255, 0).astype(np.uint8)

#save image
binary_image.tofile(output_file)


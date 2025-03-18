import numpy as np
import matplotlib.pyplot as plt

filename = 'test1B.img'
filepath = '/Users/nima/ComputerVision/ComputerVision/PatternRecognition/'


# Read the raw binary data
image_data = np.fromfile(filepath + filename, dtype=np.uint8)
#header stipping
#image_data = image_data[512:]


# Reshape the data into an array
image_data = image_data.reshape((512, 512))


# Display the image
plt.imshow(image_data, cmap='gray')
plt.show()

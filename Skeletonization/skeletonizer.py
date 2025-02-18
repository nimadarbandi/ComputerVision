#get the medial axis for each component of binary image B
#display the medial axis image M of B as a binary image (axis pixles = 1, rest = 0)
#keep distance transform values in each pixel in M.

import cv2
import numpy as np

input_file = '/Users/nima/ComputerVision/ComputerVision/Skeletonization/combB.img' #the binary image B
foreground = 0
output_file = 'combM.img' #the output file
distance_file = 'combD.img' #the output file for distance transform
distance_type = 'euclidean' #the distance metric to use

####################################Functions######################################################################
#####################################################################################################

# Compute the iterative distance propagation
# 


def iterative_distance_propagation(binary_img, distance_type="manhattan"):
    """
    Implements the Iterative Distance Propagation (IDP) algorithm.
    
    Args:
        binary_img (numpy.ndarray): Binary input image (0 for foreground, 255 for background).
        distance_type (str): Distance metric ('euclidean', 'manhattan', 'chessboard').

    Returns:
        numpy.ndarray: Distance map of the same size as input binary image.
    """
    rows, cols = binary_img.shape
    # Corrected initialization: Foreground = inf, Background = 0
    dist_map = np.where(binary_img == 0, np.inf, 0)

    iteration = 0
    while True:
        iteration += 1
        changed = False
        new_map = dist_map.copy()

        # Forward pass (top-left to bottom-right)
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if binary_img[i, j] == 0:  # Process foreground only
                    neighbors = get_neighbors(new_map, i, j, distance_type)
                    min_value = min(neighbors) + 1
                    if min_value < dist_map[i, j]:
                        new_map[i, j] = min_value
                        changed = True

        dist_map = new_map.copy()

        # Backward pass (bottom-right to top-left)
        for i in range(rows-2, 0, -1):
            for j in range(cols-2, 0, -1):
                if binary_img[i, j] == 0:
                    neighbors = get_neighbors(new_map, i, j, distance_type)
                    min_value = min(neighbors) + 1
                    if min_value < dist_map[i, j]:
                        new_map[i, j] = min_value
                        changed = True

        dist_map = new_map.copy()

        # Convergence check
        if not changed:
            break

    print(f"IDP converged after {iteration} iterations.")
    return dist_map










def get_four_neighbors(dist_img, i, j):
    return [dist_img[i-1, j], dist_img[i+1, j], dist_img[i, j-1], dist_img[i, j+1]]

# Get neighbor distances based on distance metric
def get_neighbors(dist_map, i, j, distance_type):
    if distance_type == "manhattan":
        return [dist_map[i-1, j], dist_map[i+1, j], dist_map[i, j-1], dist_map[i, j+1]]
    elif distance_type == "chessboard":
        return [dist_map[i-1, j], dist_map[i+1, j], dist_map[i, j-1], dist_map[i, j+1],
                dist_map[i-1, j-1], dist_map[i-1, j+1], dist_map[i+1, j-1], dist_map[i+1, j+1]]
    elif distance_type == "euclidean":
        return [dist_map[i-1, j], dist_map[i+1, j], dist_map[i, j-1], dist_map[i, j+1],
                dist_map[i-1, j-1]*1.41, dist_map[i-1, j+1]*1.41, dist_map[i+1, j-1]*1.41, dist_map[i+1, j+1]*1.41]



# Extract the skeleton from the distance map
def extract_skeleton(distance_image):
    rows, cols = distance_image.shape
    skeleton = np.zeros_like(distance_image, dtype=np.uint8)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if distance_image[i, j] > 0:
                # Local maximum in distance map
                if distance_image[i, j] >= max(get_neighbors(distance_image, i, j, "euclidean")):
                    skeleton[i, j] = 255
                    #print("Skeleton pixel at:", i, j)
    return skeleton



binary_image = np.fromfile(input_file, dtype=np.uint8)
binary_image = binary_image.reshape((512, 512))  # adjust dimensions as needed

distance_map = iterative_distance_propagation(binary_image, distance_type="chessboard")
skeleton = extract_skeleton(distance_map)
skeleton.tofile(output_file)
distance_map.tofile(distance_file)
for i in range(0, 512):
    for j in range(0, 512):
        print(distance_map[i][j], end="")
# Display the skeleton and distance map
cv2.imshow('Skeleton', skeleton)
#cv2.imshow('Distance Map', distance_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
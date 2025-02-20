#get the medial axis for each component of binary image B
#display the medial axis image M of B as a binary image (axis pixles = 1, rest = 0)
#keep distance transform values in each pixel in M.

import cv2
import numpy as np
input_file = "combB.img"
input_file = '/Users/nima/ComputerVision/ComputerVision/Skeletonization/'+input_file #the binary image B
foreground = 255 #select between 0 and 255 based on the input image
skeleton_file = 'combM.img' #the medial axis image M
distance_file = 'combD.img' 
recunstruncted_file = 'rec-combB.img' 

#Distance Type used is Chessboard


####################################Functions######################################################################
###################################################################################################################

def iterative_distance_propagation(binary_img):
 
    rows, cols = binary_img.shape
   
    # Corrected input image: Foreground = inf, Background = 0
    distance_image = np.where(binary_img == 0, 0, np.inf)

    iteration = 0
    while True:
        iteration += 1
        changed = False
        temp_img = distance_image.copy()

        for i in range(0, rows):
            for j in range(0, cols):
                if temp_img[i, j] > 0 :  # Process foreground pixels
                    neighbors = get_neighbors(temp_img, i, j, "chessboard")
                    min_value = min(neighbors) + 1
                    if min_value < temp_img[i, j]:
                        temp_img[i, j] = min_value
                        changed = True

        distance_image = temp_img.copy()

        # Convergence check
        if not changed:
            break

    print(f"IDP converged after {iteration} iterations.")
    return distance_image




def get_neighbors(dist_map, i, j, distance_type):
    neighbors = []
    rows, cols = dist_map.shape

    if distance_type == "manhattan":
        if i > 0:  
            neighbors.append(dist_map[i-1, j])  # Up
        if i < rows - 1:  
            neighbors.append(dist_map[i+1, j])  # Down
        if j > 0:  
            neighbors.append(dist_map[i, j-1])  # Left
        if j < cols - 1:  
            neighbors.append(dist_map[i, j+1])  # Right

    elif distance_type == "chessboard":
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip the center pixel
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append(dist_map[ni, nj])  # Add valid neighbors

    return neighbors


# Extract the skeleton from the distance map
def extract_skeleton(distance_image):
    rows, cols = distance_image.shape
    skeleton = np.zeros_like(distance_image, dtype=np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            if distance_image[i, j] > 0:
                # Local maximum in distance map
                if distance_image[i, j] >= max(get_neighbors(distance_image, i, j, "chessboard")):
                    skeleton[i, j] = distance_image[i, j]
    return skeleton

def reconstruct_image(skeleton):
    rows, cols = skeleton.shape
    image_M = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, cols):
            if skeleton[i, j] > 0:
               radius = int(skeleton[i, j])
               for m in range(i-radius,i+radius):
                     for n in range(j-radius,j+radius):
                        if 0 <= m < rows and 0 <= n < cols:
                                image_M[m, n] = 255
    return image_M

############################################################################################################################################################
# Main
############################################################################################################################################################

binary_image = np.fromfile(input_file, dtype=np.uint8)
binary_image = binary_image.reshape((512, 512))  # adjust dimensions as needed
if foreground == 0: # inverse the images with white background
    binary_image = 255 - binary_image

distance_map = iterative_distance_propagation(binary_image)


skeleton_M = extract_skeleton(distance_map)
skeleton_image = np.where(skeleton_M > 0, 255, 0).astype(np.uint8)
recons_image = reconstruct_image(skeleton_M)

cv2.imshow('Binary Image', binary_image)
cv2.imshow('Skeleton', skeleton_image)
cv2.imshow('Reconstructed Image', recons_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
skeleton_M.tofile(skeleton_file)
recons_image.tofile(recunstruncted_file)
distance_map.tofile(distance_file)

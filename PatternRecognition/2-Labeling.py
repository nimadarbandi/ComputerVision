import numpy as np
import matplotlib.pyplot as plt
import pprint
from math import sqrt

# Inputs
input_file = 'combB.img' #the binary image B
min_size = 500 #maximum pixel count for a component to be considered valid

# Outputs
area = {} #the component size for each component
centroid = {} # the location of the centroid of each component
bounding_box = {} #the coordinates of the bounding box for each component
theta = {} #the orientation of axis of ellongation of each component
eccentricity = {} #the eccentricity of each component
perimeters = {} #the perimeter of each component
compactness = {} #the compactness of each component
output_file = 'combO.jpg' #the color coded displlay of the components of the image B

############################################################Functions############################################################
########################################################################################################################################################################################

def iterative_CCL(binary_image):
    height, width = 512, 512
    labels = np.zeros_like(binary_image, dtype=int)
    #print(labels.shape)

    new_label = 1
    # equivalance table
    equivalence = {}

    
    for i in range(height):
        for j in range(width):
            #print(binary_image[i, j],end=",")
            if binary_image[i, j] == 0:  # If pixel is part of a component
                
                neighbor_labels = [0, 0] #[upper, left] pixels
                if i > 0 and labels[i-1, j] > 0: #upper pixel has label
                    neighbor_labels[0] = labels[i-1, j]
                if j > 0 and labels[i, j-1] > 0: #left pixel has label
                    neighbor_labels[1] = labels[i, j-1]

                if neighbor_labels[0] > 0 or neighbor_labels[1] > 0:
                    if neighbor_labels[0] == 0 : #only left pixel has label
                        labels[i, j] = neighbor_labels[1]
                    elif neighbor_labels[1] == 0 : #only upper pixel has label
                        labels[i, j] = neighbor_labels[0]
                    elif neighbor_labels[0] == neighbor_labels[1] : #both upper and left pixels have the same label
                        labels[i, j] = neighbor_labels[0]
                    elif neighbor_labels[0] != neighbor_labels[1] : #both upper and left pixels have different labels
                        labels[i, j] = neighbor_labels[0]
                        # Record equivalence: all neighbor labels become equivalent.
                        if neighbor_labels[0] in equivalence:
                            equivalence[neighbor_labels[0]].add(int(neighbor_labels[1]))
                        else:
                            equivalence[neighbor_labels[0]] = {int(neighbor_labels[1])}
                        if neighbor_labels[1] in equivalence:
                            equivalence[neighbor_labels[1]].add(int(neighbor_labels[0]))
                        else:
                            equivalence[neighbor_labels[1]] = {int(neighbor_labels[0])}
                else:
                    labels[i, j] = new_label
                    equivalence[new_label] = {new_label}
                    new_label += 1
    #pprint.pprint(equivalence)
    #print(equivalence.items())
    
    # merge overlapping sets in equivalance table until stable.
    changed = True
    while changed:
        changed = False
        for key in list(equivalence.keys()):
            merged_set = set(equivalence[key])
            for lbl in list(equivalence[key]):
                merged_set = merged_set.union(equivalence.get(lbl, set()))
            if merged_set != equivalence[key]:
                equivalence[key] = merged_set
                changed = True

    # Build a final mapping table based on the equivalance table (for each label, assign the minimum label from its equivalence set.)
    final_map = {}
    for key, eq_set in equivalence.items():
        min_label = min(eq_set)
        for l in eq_set:
            final_map[l] = min_label

    #pprint.pprint(final_map)

    # Replace each label in the image using final_map.
    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0:
                labels[i, j] = final_map[labels[i, j]]

    return labels

###########################################################

def count_components(labels, min_size):
    valid_components = 0
    area = {}
    centroid = {}
    bounding_box = {}
    theta = {}
    eccentricity = {}
    perimeters = {}
    compactness = {}

    # Count pixels for each label
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    for i in range(len(unique_labels)):
        if counts[i] >= min_size:
            valid_components += 1
            #area
            area[unique_labels[i]] = counts[i]
            #component coordinates
            row, col = np.where(labels == unique_labels[i])
            #centroid
            centroid_row = np.mean(row)
            centroid_col = np.mean(col)
            centroid[unique_labels[i]] = (centroid_row, centroid_col)
            #bounding box
            min_row = np.min(row)
            min_col = np.min(col)
            max_row = np.max(row)
            max_col = np.max(col)
            bounding_box[unique_labels[i]] = (min_row, min_col, max_row, max_col)
            #orientation
            a = np.sum((row - centroid_row) ** 2)
            b = np.sum((col - centroid_col) ** 2)
            c = np.sum((row - centroid_row) * (col - centroid_col))
            theta [unique_labels[i]] = 0.5 * np.arctan2(2 * c, a - b)
            #eccentricity
            delta = 0.5 * np.sqrt((a - b) ** 2 + 4 * c ** 2)
            E1 = (a + b + delta) / 2 #largest eigenvalue
            E2 = (a + b - delta) / 2 #smallest eigenvalue
            if E1 > 0 :
                eccentricity[unique_labels[i]] = np.sqrt(1 - (E2 / E1))
            else:
                eccentricity[unique_labels[i]] = 0
            #perimeter
            component_label = (labels == unique_labels[i]).astype(np.uint8)
            boundary = get_boundary(component_label)
            perim = 0
            for j in range(1, len(boundary)):
                r1, c1 = boundary[j-1]
                r2, c2 = boundary[j]
                dr = abs(r2 - r1)
                dc = abs(c2 - c1)
                if dr == 1 and dc == 1:
                    perim += sqrt(2)
                else:
                    perim += 1
            perimeters[unique_labels[i]] = perim
            #compactness
            compactness[unique_labels[i]] = (perimeters[unique_labels[i]] ** 2) / area[unique_labels[i]]

    return valid_components, area, centroid, bounding_box, thetha, eccentricity, perimeters, compactness

###############################################

def get_boundary(component_label):

    # getting component pixels
    rows, cols = np.where(component_label == 1)
    if len(rows) == 0:
        return []  # no component pixels
    s = (rows[0], cols[0])  # starting pixel for the boundary
    boundary_list = [s]
    
    # previous pixel = the 4-neighbor to the west of s.
    p = (s[0], s[1] - 1)
    
    # Define 8-neighbor in clockwise order starting from West.
    eight_neighbor = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
                        (0, 1), (1, 1), (1, 0), (1, -1)]
    

    try:
        start_pixel = eight_neighbor.index((0, -1))
    except ValueError:
        start_pixel = 0
    
    current = s
    while True:
        found = False
        next_pixel = None
        # Examine the 8 neighbors of 'current' in clockwise order starting at start_pixel.
        for i in range(8):
            idx = (start_pixel + i) % 8
            offset = eight_neighbor[idx]
            candidate = (current[0] + offset[0], current[1] + offset[1])
            # Ensure candidate is in image pixels.
            if (candidate[0] < 0 or candidate[0] >= component_label.shape[0] or 
                candidate[1] < 0 or candidate[1] >= component_label.shape[1]):
                continue
            if component_label[candidate] == 1:
                # The first neighbor that is part of the component.
                next_pixel = candidate
                found = True
                # Set the new start
                start_pixel = (idx - 1) % 8
                break
        if not found:
            # If no neighbor is found
            break
        # Add the found boundary pixel.
        boundary_list.append(next_pixel)

        if next_pixel == s:
            break
        current = next_pixel
    
    return boundary_list

###############################################################
def Paint_image(labels, output_file, min_size):
    # Create an RGB image
    rgb_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    # Get unique labels
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)

    # random colors for each label
    for j in range(len(unique_labels)):
        if counts[j] >= min_size:
            color = np.random.randint(0, 256, size=3)
            rgb_image[labels == unique_labels[j]] = color

    # Save the RGB image
    plt.imsave(output_file, rgb_image)

##############################################################Main############################################################
##############################################################################################################################

# Read the binary image:
binary_image = np.fromfile(input_file, dtype=np.uint8)
#header stipping
#binary_image = binary_image[512:]
binary_image = binary_image.reshape((512, 512))  # adjust dimensions as needed

#conncected component labeling
labels = iterative_CCL(binary_image)


total_components, area, centroid, bounding_box, theta, eccentricity, perimeters, compactness = count_components(labels, min_size)
print("Number of components:", total_components)
pprint.pprint(area)
pprint.pprint(centroid)
pprint.pprint(bounding_box)
pprint.pprint(theta)
print("eccentricity")
pprint.pprint(eccentricity)
print("perimeter")
pprint.pprint(perimeters)
print("compactness")
pprint.pprint(compactness)

Paint_image(labels, output_file, min_size)
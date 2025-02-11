import numpy as np
import matplotlib.pyplot as plt
import pprint
from math import sqrt

input_file = 'combB.img'
output_file = 'test3C.img'

min_size = 50  # For example, 50 pixels

############################################################Functions############################################################

def iterative_CCL(binary_image):
    height, width = 512, 512
    labels = np.zeros_like(binary_image, dtype=int)
    #print(labels.shape)

    new_label = 1
    # A dictionary to hold equivalence information: label -> set of equivalent labels
    equivalence = {}

    # First pass: assign temporary labels and record equivalences.
    for i in range(height):
        for j in range(width):
            #print(binary_image[i, j],end=",")
            if binary_image[i, j] == 0:  # If pixel is part of a component
                # Get labels of already processed neighbors (above and left)
                neighbor_labels = [0, 0]
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
    
    # Iteratively merge overlapping sets until stable.
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

    # Build a final mapping: for each label, assign the minimum label from its equivalence set.
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
    thetha = {}
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
            thetha [unique_labels[i]] = 0.5 * np.arctan2(2 * c, a - b)
            #eccentricity
            delta = 0.5 * np.sqrt((a - b) ** 2 + 4 * c ** 2)
            L1 = (a + b + delta) / 2 #largest eigenvalue
            L2 = (a + b - delta) / 2 #smallest eigenvalue
            if L1 > 0 :
                eccentricity[unique_labels[i]] = np.sqrt(1 - (L2 / L1))
            else:
                eccentricity[unique_labels[i]] = 0
            #perimeter
            component_mask = (labels == unique_labels[i]).astype(np.uint8)
            boundary = trace_boundary(component_mask)
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

def trace_boundary(component_mask):

    # Find the starting pixel (the first '1' found in a row-major (raster) scan)
    rows, cols = np.where(component_mask == 1)
    if len(rows) == 0:
        return []  # no component pixels
    s = (rows[0], cols[0])  # starting pixel for the boundary
    boundary_list = [s]
    
    # Set the "previous" pixel to be the 4-neighbor to the west of s.
    # (If s is at (r, c), then set p = (r, c-1).)
    p = (s[0], s[1] - 1)
    
    # Define 8-neighbor offsets in clockwise order starting from West.
    # Order: West, Northwest, North, Northeast, East, Southeast, South, Southwest.
    eight_neighbor_offsets = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
                        (0, 1), (1, 1), (1, 0), (1, -1)]
    
    # Determine the starting index in neighbor_offsets corresponding to p relative to s.
    # p relative to s is: (s_row, s_col - 1) => offset = (0, -1)
    try:
        start_index = eight_neighbor_offsets.index((0, -1))
    except ValueError:
        start_index = 0
    
    current = s
    while True:
        found = False
        next_pixel = None
        # Examine the 8 neighbors of 'current' in clockwise order starting at start_index.
        for i in range(8):
            idx = (start_index + i) % 8
            offset = eight_neighbor_offsets[idx]
            candidate = (current[0] + offset[0], current[1] + offset[1])
            # Ensure candidate is within image bounds.
            if (candidate[0] < 0 or candidate[0] >= component_mask.shape[0] or 
                candidate[1] < 0 or candidate[1] >= component_mask.shape[1]):
                continue
            if component_mask[candidate] == 1:
                # The first neighbor that is part of the component.
                next_pixel = candidate
                found = True
                # Set the new start index to be the neighbor index immediately preceding the found pixel.
                start_index = (idx - 1) % 8
                break
        if not found:
            # If no neighbor is found (which is unusual), terminate.
            break
        # Add the found boundary pixel.
        boundary_list.append(next_pixel)
        # Termination condition: when the next pixel equals the starting pixel s.
        # (Some variants check that the new candidate is s and that the pixel before s is the initial p.)
        if next_pixel == s:
            break
        # Update current and continue.
        current = next_pixel
    
    return boundary_list

##############################################################Main############################################################

# Read the binary image:
binary_image = np.fromfile(input_file, dtype=np.uint8)
#header stipping
#binary_image = binary_image[512:]
binary_image = binary_image.reshape((512, 512))  # adjust dimensions as needed

# Run the iterative connected component labeling:
labels = iterative_CCL(binary_image)

area = {}
centroid = {}
bounding_box = {}
thetha = {}
eccentricity = {}
perimeters = {}
compactness = {}
# Count the valid components:
total_components, area, centroid, bounding_box, thetha, eccentricity, perimeters, compactness = count_components(labels, min_size)
print("Number of components:", total_components)
pprint.pprint(area)
pprint.pprint(centroid)
pprint.pprint(bounding_box)
pprint.pprint(thetha)
print("eccentricity")
pprint.pprint(eccentricity)
print("perimeter")
pprint.pprint(perimeters)
print("compactness")
pprint.pprint(compactness)

import numpy as np
import matplotlib.pyplot as plt
import pprint

input_file = 'combB.img'
output_file = 'test3C.img'

min_size = 4  # For example, 50 pixels

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
    # Count pixels for each label
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    for count in counts:
        if count >= min_size:
            valid_components += 1
    return valid_components

##############################################################Main############################################################

# Read the binary image:
binary_image = np.fromfile(input_file, dtype=np.uint8)
#header stipping
#binary_image = binary_image[512:]
binary_image = binary_image.reshape((512, 512))  # adjust dimensions as needed

# Run the iterative connected component labeling:
labels = iterative_CCL(binary_image)


# Count the valid components:
total_components = count_components(labels, min_size)
print("Number of components:", total_components)
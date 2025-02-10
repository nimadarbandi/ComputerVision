import numpy as np
import matplotlib.pyplot as plt

input_file = 'test3B.img'
output_file = 'test3C.img'

min_size = 50  # For example, 50 pixels

############################################################Functions############################################################

def iterative_CCL(binary_image):
    height, width = 512, 512
    labels = np.zeros_like(binary_image, dtype=int)
    next_label = 1
    # A dictionary to hold equivalence information: label -> set of equivalent labels
    equivalence = {}

    # First pass: assign temporary labels and record equivalences.
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] > 0:  # If pixel is part of a component
                # Get labels of already processed neighbors (above and left)
                neighbor_labels = []
                if i > 0 and labels[i-1, j] > 0:
                    neighbor_labels.append(labels[i-1, j])
                if j > 0 and labels[i, j-1] > 0:
                    neighbor_labels.append(labels[i, j-1])
                    
                if not neighbor_labels:
                    # No neighbor has a label: assign new label.
                    labels[i, j] = next_label
                    equivalence[next_label] = {next_label}
                    next_label += 1
                else:
                    # Use the smallest label among neighbors
                    min_label = min(neighbor_labels)
                    labels[i, j] = min_label
                    # Record equivalence: all neighbor labels become equivalent.
                    for lbl in neighbor_labels:
                        # Merge the equivalence sets.
                        if lbl in equivalence:
                            equivalence[min_label] = equivalence[min_label].union(equivalence[lbl])
                        else:
                            equivalence[min_label].add(lbl)
                        # Make sure every involved label maps to the same set.
                        for l in equivalence[min_label]:
                            equivalence[l] = equivalence[min_label]

    # Second pass: resolve label equivalences.
    # Create a mapping from each label to the smallest equivalent label.
    label_map = {}
    for lbl, eq_set in equivalence.items():
        smallest = min(eq_set)
        for l in eq_set:
            label_map[l] = smallest

    # Replace each label in the image using the mapping.
    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0:
                labels[i, j] = label_map[labels[i, j]]

    return labels

###########################################################

def count_components(labels, min_size):
    # Count pixels for each label (ignore background, label 0)
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    # Count components that meet or exceed the minimum size.
    valid_components = sum(1 for count in counts if count >= min_size)
    return valid_components

##############################################################Main############################################################

# Read the binary image:
binary_image = np.fromfile(input_file, dtype=np.uint8)
binary_image = binary_image.reshape((512, 512))  # adjust dimensions as needed

# Run the iterative connected component labeling:
labels = iterative_CCL(binary_image)


# Count the valid components:
total_components = count_components(labels, min_size)
print("Number of components:", total_components)
**Steps to Compute Skeletonization using IDP**
1. Load a binary image: Convert it to a black-and-white format.
2. Initialize a distance map: Set object pixels (foreground) to 0 and background to ∞.
3. Iteratively propagate the minimum distance:
    1. Use Manhattan (4-neighbor), Chessboard (8-neighbor), or Euclidean distance.
    2. Update each pixel based on the minimum value from its neighbors.
4. Extract the skeleton: Identify ridge points where distance values are locally maximal.

For the binary images I used 0 and 255 in last assignment. because of the simplicity in preview. is it fine to do the same or this should be in 0 and 1?

can you explain this "Note that M by itself is not a 
binary image (i.e., it retains the distance transform value at each skeleton pixel). "


Which distance metric is prefered for the skeletonization?

How can we recosntruct the image with skeleton image solely? don't we need distance image aswell?


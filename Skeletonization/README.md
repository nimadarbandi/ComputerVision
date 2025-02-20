**Steps to Compute Skeletonization using IDP**
1. Load a binary image: Convert it to a black-and-white format.
2. Initialize a distance map: Set object pixels (foreground) to 0 and background to ∞.
3. Iteratively propagate the minimum distance:
    1. Use Manhattan (4-neighbor), Chessboard (8-neighbor), or Euclidean distance.
    2. Update each pixel based on the minimum value from its neighbors.
4. Extract the skeleton: Identify ridge points where distance values are locally maximal.

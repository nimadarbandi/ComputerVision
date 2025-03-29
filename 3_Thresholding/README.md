For the three grayscale test images test1.img, test2.img and test3.img (in .img format) provided in eLC, show the result of the following thresholding algorithms:

1. Thresholding using peakiness detection. Explain clearly the evaluation criterion used to select the pair of grayscale peaks and the intervening valley.

After generating the histogram from the pixel values of the image, and counting the number of pixels with 256 different grayscale values, I compared the adjacent gray levels of the histogram to find local minimum and local maximums. The local minima must be lower than the right and left values of the histogram and the local maxima must be higher than the neighbouring graylevels in the histogram. I added the local maxima and minima in two lists and then used the FOM you mentioned in the class to find the best threshold between candidate values.
I tried two different FOMs you mentioned in class and this one showed more reasonable thresholds: 
![alt text](<Pasted Graphic 3.png>)


2. Iterative thresholding. 


3. Dual thresholding with region growing. Explain how you would choose the two threshold values automatically from the histogram. Explain clearly the logic behind the heuristic(s) for selection of the two threshold values.

a. remove the header from the binary image
b. Create the histogram of the image
c. remove the outliers and smooth the histogram
d. Select two conservative thresholds from two sides of the histogram
e. Assign R1 and R3 labels to the pixels two sides of the histogram
f. Scan the pixels in R2 section of the histogram and assign R1 and R3 labels to them


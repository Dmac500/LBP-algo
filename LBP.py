import numpy as np
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
 
# calculate if the neighboring pixels is a higher value of a lower value
def calc(currentPix, arr):
    
    newarr = []
    for i in arr: 
        if i >= currentPix:
            newarr.append(1)
        else:
            newarr.append(0)

    return np.array(newarr)
# Takes in an array that represents binary for a number (0-255)
def BitCalc(array):
    count = 7
    runningTotal = 0
    for i in array:
        if i == 1:
            runningTotal += 2**count 
        count -= 1
    return runningTotal
    
# take the image convert it to grey scale and read in the pixels into a 2D array
img = Image.open('input_image.jpg').convert('L')
img.save('greyscale1.jpg')
im = imread("greyscale1.jpg")

histogramCount = []
lbpimage= np.zeros_like(im)

# loop through the 2D array
for row in range(0,len(im)-3):
    for col in range(0,len(im[0])-3):
        
        # get a 3X3 part of image
        # find the center point
        # flatten it out to be a 1d array 
        # convert to binary 
        # calculate the value from the binary 
        img1 = np.array(im[row:row+3,col:col+3])
        centerPoint = img1[1][1]
        image1 = (img1 >= centerPoint)*1
        image_vector = np.ndarray.flatten(img1)
        image_vector = np.delete(image_vector,4)
        binaryRep = calc(centerPoint ,image_vector ) 
        num = BitCalc(binaryRep)
    
        # add value to histogram
        # add pixel to the blank LBPImage
        histogramCount.append(num)
        lbpimage[row+1,col+1] = num
       
        
    
    


arrForHis = np.array(histogramCount)
# creat photo out of array
new_image = Image.fromarray(lbpimage,'L')

plt.hist(arrForHis, bins=256, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
 
 
new_image.save('LBPIMAGE.jpg')


print("done")

plt.show()

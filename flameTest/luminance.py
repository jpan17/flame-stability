# Pixel line dims: 470,471,230,270
# Box dims: 480, 450, 270, 220

import cv2

# for each pixel in an image, calculate luminance and store it in a 1D array
def lumArray (image, height, width): 
    # array to return
    luminances = []
    
    # extract luminances
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    frameL, frameA, frameB = cv2.split(frame)
    
    # iterate through all pixels and add to array
    for row in range(220, 270):
        for col in range(450, 480):
            luminances.append(frameL[row][col])
    
    return luminances
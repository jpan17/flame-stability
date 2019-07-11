import cv2

# for each pixel in an image, calculate RGB and store it in a 1D array 
def rgbArray (image, height, width, test):
    # array to return
    rgbs = []
    
    # extract RGB values
    frameB, frameG, frameR = cv2.split(image)
    
    # code for all RGB
    if test == 'RGB':
        for row in range(0, height):
            for col in range(0, width):
                rgbs.append(frameR[row][col])
                rgbs.append(frameG[row][col])
                rgbs.append(frameB[row][col])
        
    # iterate through all pixels and add to array
    # R
    if test == 'R':
        for rowR in range(0, height):
            for colR in range(0, width):
                rgbs.append(frameR[rowR][colR])
            
    # G
    if test == 'G':
        for rowG in range(0, height):
            for colG in range(0, width):
                rgbs.append(frameG[rowG][colG])
            
    # B
    if test == 'B':
        for rowB in range(0, height):
            for colB in range(0, width):
                rgbs.append(frameB[rowB][colB])

    return rgbs

import cv2
import numpy as num 
import sys
# =========================================================================== #

# read in the video
user = str(sys.argv[1])
fire = cv2.VideoCapture(user)

# check if video opened successfully
if (fire.isOpened() == False):
    print("Error opening video stream or file")
    
# let's count the number of frames in the video
frameCount = 1
ret, frame = fire.read()

# record frame dimensions
height, width, channels = frame.shape
tallness = height
wideness = width

# display the video until 'q' is pressed or until it terminates
while(fire.isOpened()):
    ret, frame = fire.read()
    frameCount += 1
    
    if ret == True:
        cv2.imshow('Fire', frame)
        
        # terminate the video before it finishes
        if cv2.waitKey(10) == ord('q'):
            break
        
        # record the frame number but don't stop the video
        if cv2.waitKey(10) == ord('w'):
            print("Stop at: " + str(frameCount))
        
    else:
        break

print("Total Number of Frames: " + str(frameCount))
print("Height of video: " + str(tallness))
print("Width of video: " + str(wideness))

# close everything
fire.release()
cv2.destroyAllWindows()

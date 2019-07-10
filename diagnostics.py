import cv2
import numpy as num 
# =========================================================================== #

# read in the video
fire = cv2.VideoCapture('fire.mp4')

# check if video opened successfully
if (fire.isOpened() == False):
    print("Error opening video stream or file")
    
# let's count the number of frames in the video
frameCount = 1
ret, frame = fire.read()

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
        if cv2.waitKey(25) == ord('q'):
            break
        
        # record the frame number
        if cv2.waitKey(25) == ord('w'):
            print("Stop at: " + str(frameCount))
        
    else:
        break

print("Total Number of Frames: " + str(frameCount))
print("Height of video: " + str(tallness))
print("Width of video: " + str(wideness))
fire.release()
cv2.destroyAllWindows()

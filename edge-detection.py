import cv2
import csv
import numpy as np
import pandas
from PIL import Image
from PIL import ImageEnhance
# =========================================================================== #

def edgeDetection():
    
    df = pandas.read_csv('EtOH_flamemap.csv')

    
    for i in range(0, len(df['File name'])):
        
        # read in video
        fire = cv2.VideoCapture('./fireFiles/' + df['File name'][i])
        
        # print error message if you can't read it in
        if (fire.isOpened() == False):
            print("Error opening video file or stream")

        # display the video until 'q' is pressed or until it terminates
        while (fire.isOpened()):
            ret, frame = fire.read()
            
            if ret == True:
                grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                enhancer = ImageEnhance.Sharpness(frame)
                
                cv2.imshow('default', frame)
                # laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                # sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5)
                # sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
                # edges = cv2.Canny(frame, 25, 50)
                
                # cv2.imshow('laplacian', laplacian)
                # cv2.imshow('sobelx', sobelx)
                # cv2.imshow('sobely', sobely)
                # cv2.imshow('edges', edges)

                # terminates the video before it finishes
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
   
    fire.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    edgeDetection()

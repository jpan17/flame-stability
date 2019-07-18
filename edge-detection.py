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
        fileName = df['File name'][i]
        fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
        ret, frame = fire.read()
        height, width, channels = frame.shape
        frameWidth = width
        frameHeight = height
        out = cv2.VideoWriter('edges-' + fileName, cv2.VideoWriter_fourcc(*'XVID'),
                               30, (frameWidth, frameHeight))
        
        # print error message if you can't read it in
        if (fire.isOpened() == False):
            print("Error opening video file or stream")

        # display the video until 'q' is pressed or until it terminates
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            if ret == True:
                
                
                cv2.imshow('default', frame)
        
                clahe = cv2.createCLAHE(clipLimit = 15, tileGridSize = (6,6))
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l2 = clahe.apply(l)
                
                lab = cv2.merge((l2,a,b))
                newFrame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                cv2.imshow('Contrast', newFrame)
                grayscale = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('grayscale', grayscale)
                
                gray_filtered = cv2.bilateralFilter(grayscale, 5, 35, 35)
                
                edges_filtered = cv2.Canny(gray_filtered, 40, 50)
                cv2.imshow('Bilateral', edges_filtered)
                
                hey = cv2.cvtColor(edges_filtered, cv2.COLOR_GRAY2BGR)
                
                # print(edges_filtered)
                out.write(hey)
                
                # laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                # sobelx = cv2.Sobel(newFrame, cv2.CV_64F, 1, 0, ksize = 3)
                # sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
                # edges = cv2.Canny(newFrame, 50, 100)
                
                # cv2.imshow('laplacian', laplacian)
                # cv2.imshow('sobelx', sobelx)
                # cv2.imshow('sobely', sobely)
                # cv2.imshow('edges', edges)

                # terminates the video before it finishes
           
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                print("broken")
                break
    
    print('yike')
    out.release()
    fire.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    edgeDetection()

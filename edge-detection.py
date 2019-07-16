import cv2
import numpy as np
# =========================================================================== #

df = pandas.read_csv('EtOH_flamemap.csv')

def edgeDetection():
    for i in range(0, len(df['File name'])):
        
        fire = cv2.VideoCapture('./fireFiles/' + df['File name'][i])
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
    
        ret, frame = fire.read()
        
        while (fire.isOpened()):
                    
            laplacian = cv2.Laplacean(frame, cv2.CV_64F)
            sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5)
            sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
            edges = cv2.canny(frame, 100, 200)
            
            
            cv2.imshow('default', frame)
            cv2.imshow('laplacian', laplacian)
            cv2.imshow('sobelx', sobelx)
            cv2.imshow('sobely', sobely)
            cv2.imshow('edges', edges)
            
            if cv2.waitKey(5) & 0xFF:
            break
        
    cv2.destroyAllWindows()
    fire.release()
    

if __name__ == "__main__":
    edgeDetection()

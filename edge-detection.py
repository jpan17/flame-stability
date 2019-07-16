import cv2
import numpy as np
import pandas
# =========================================================================== #

def edgeDetection():
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    for i in range(0, len(df['File name'])):
        
        fire = cv2.VideoCapture('.\\fireFiles\\' + df['File name'][i])
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
        
        while (fire.isOpened()):
            
            ret, frame = fire.read()
    
            if ret == True:
                        
                laplacian = cv2.Laplacian(frame, cv2.CV_64F)
                sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5)
                sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
                edges = cv2.Canny(frame, 25, 25)
                
                
                cv2.imshow('default', frame)
                cv2.imshow('laplacian', laplacian)
                cv2.imshow('sobelx', sobelx)
                cv2.imshow('sobely', sobely)
                cv2.imshow('edges', edges)
            else:
                break
            
            if cv2.waitKey(25) == ord('q'):
                print('hey')
                break
        
    
    fire.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    edgeDetection()

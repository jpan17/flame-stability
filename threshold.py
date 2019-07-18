import cv2
import csv
import numpy as np
import pandas
# =========================================================================== #

def threshold():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0 
    videoCount = 0
    
    for i in range(0, len(df['File name'])):
        
        videoCount += 1
        
        fileName = df['File name'][i]
        fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            
            if ret == True:
                
                # use 60 for just the inside flame
                grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                retval, threshold = cv2.threshold(grayscaled, 20, 255,
                                                  cv2.THRESH_BINARY)
                cv2.imshow('threshold', threshold)
                cv2.imshow('default', frame)
                
                frameCount += 1
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
    fire.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    threshold()
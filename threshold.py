import cv2
import csv
import numpy as np
import pandas
# =========================================================================== #

def threshold():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0 
    
    for i in range(0, len(df['File name'])):
        print('hielo')
        fileName = df['File name'][i]
        fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            
            frameCount += 1
            
            if frameCount == 1: 
                cv2.imwrite("./testImage.png", frame)
            
            if ret == True:
                print('hi')
                cv2.imshow('default', frame)
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
        fire.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    threshold()
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
        
        ret, frame = fire.read()
        height, width, channels = frame.shape
        frameWidth = width
        frameHeight = height
        out = cv2.VideoWriter('threshold100-' + fileName, cv2.VideoWriter_fourcc(*'XVID'),
                              30, (frameWidth, frameHeight))
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            
            if ret == True:
                
                # use 60 for just the inside flame, 20 for outside
                grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                retval, threshold = cv2.threshold(grayscaled, 100, 255,
                                                  cv2.THRESH_BINARY)
                
                threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
                threshold[np.where((threshold == [255,255,255]).all(axis=2))] = [255, 255, 255]
                cv2.imshow('threshold', threshold)
                cv2.imshow('default', frame)
                
                out.write(threshold)
                
                frameCount += 1
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
    
    fire.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    threshold()
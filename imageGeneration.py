import cv2
import csv
import numpy as np
import pandas
# =========================================================================== #

def main():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0
    numFrames = 0
    stability = ''
    
    for i in range (0, len(df['File name'])):
        
        fileName = df['File name'][i]        
        isStable = df['box'][i]
        
        if isStable > 1.25:
            stability = "stable"
        elif isStable > .5:
            stability = "uncertain"
        else:
            stability = "unstable"
            
        fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
        ret, frame = fire.read()
        height, width, channels = frame.shape
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            frameCount += 1
            
            if ret == True:
                
                cv2.imshow('Fire', frame)
                
                if frameCount % 30 == 0:
                    numFrames += 1
                    if numFrames == 312:
                        name = "image_" + str(numFrames) + "_" + stability + ".png"
                        print(name)
                        cv2.imwrite('./imageData/' + name, frame)
                    
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
            
        
    fire.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
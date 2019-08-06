import cv2
import csv
import pandas
import flameTest.luminance as luminance
import flameTest.twoComponentPCA as twoComponentPCA
from pandas import Series
# =========================================================================== #

def autocorrelation():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    for i in range(0, len(df['File name'])):
        
        fire = cv2.VideoCaptures('./fireFiles/' + df['File name'][i])
        print(df['File name'][i])
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        ret, frame = fire.read()
        height, width, channels = frame.shape
        vidHeight = height
        vidWidth = width
        
        while (fire.isOpened() and numFrames < 250):
            
            ret, frame = fire.read()
            
            if ret == True:
                
                cvv2.imshow('Fire', frame)
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
            
    fire.release()
    cv2.destoryAllWindows()
    
if __name__ == "__main__":
    autocorrelation()
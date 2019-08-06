import cv2
import csv
import pandas
import flameTest.luminance as luminance
import flameTest.twoComponentPCA as twoComponentPCA
import statistics
import matplotlib.pyplot as plt
# =========================================================================== #

def autocorrelation():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    temp = []
    series = []
    frames = []
    numFrames = 0
    
    for i in range(0, len(df['File name'])):
        
        fire = cv2.VideoCapture('./fireFiles/' + df['File name'][i])
        print(df['File name'][i])
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        ret, frame = fire.read()
        height, width, channels = frame.shape
        vidHeight = height
        vidWidth = width
        
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            
            if ret == True:
                
                cv2.imshow('Fire', frame)
                numFrames += 1 
                temp = luminance.lumArray(frame, vidHeight, vidWidth)
                series.append(statistics.mean(temp))
                frames.append(numFrames)
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
    
    plt.plot(frames, series)
    plt.show()
    fire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    autocorrelation()
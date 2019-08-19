import cv2
import csv
import pandas
import flameTest.luminance as luminance
import flameTest.twoComponentPCA as twoComponentPCA
import statistics
import matplotlib.pyplot as plt
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# =========================================================================== #

def autocorrelation():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    temp = []
    series = []
    frames = []
    numFrames = 0
    
    # for i in range(0, len(df['File name'])):
        
    fileName = "flame-spray-02.avi"
    fire = cv2.VideoCapture('./fireFiles/' + fileName)
    # print(df['File name'][i])
    isStable = 2
    
    if isStable > 1.25:
        stability = "stable"
    elif isStable > .5:
        stability = "uncertain"
    else:
        stability = "unstable"
    
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
            # plot_acf(series, lags = 250) 
            # plt.title('Autocorrelation ' + stability)
            # plt.savefig('acf-' + fileName +'.png')
            # plt.close()
            # # plot_pacf(series, lags = 50)
            # # plt.savefig('pacf' + fileName + '.png')
            # # plt.show() 
            # # plt.show()
            # # plt.close()
            # series = []
            break
    
    plt.plot(frames, series)
    plt.title('Mean Luminance of Bounding Box Pixels vs Time (frames)', fontsize = 24)
    plt.xlabel('Time (frames)', fontsize = 24)
    plt.ylabel('Mean Luminance', fontsize = 24)
    
    # plot_acf(series, lags = 250) 
    # plt.title('Autocorrelation flame-spray-02.avi (unstable)', fontsize = 24)
    # plt.xlabel('Lag (frames)', fontsize = 24)
    # plt.ylabel('Correlation Coefficient')
    plt.show()
    # plt.close()
            
    series = []
    
    fire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    autocorrelation()
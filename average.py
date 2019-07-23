import cv2
import csv
import numpy as np 
import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# =========================================================================== #

def averageLine(image):
    
    count = 0
    total = 0
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(frame)
    
    for row in range(230, 270):
        for col in range(470, 471):
            total += l[row][col]
            count += 1
            
    average = total / count
    return average

def main():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0
    frames = []
    videoCount = 0
    fileName = ''
    
    totalAverage = 0
    averages = []
    
    # for i in range(0, 1):
        
    #     videoCount += 1
        
    fileName = "flame-spray-54.avi"
    fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
    ret, frame = fire.read()
    height, width, channels = frame.shape
    frameWidth = width
    frameHeight = height
        
    while (fire.isOpened()):
           
        ret, frame = fire.read()
          
        if ret == True:
            
            frameCount += 1
                
            cv2.imshow('default', frame)
            averages.append(averageLine(frame))
            frames.append(frameCount)
                
            if cv2.waitKey(25) == ord('q'):
                break
                
        else:
            break
        
    totalAverage = sum(averages) / len(averages)
    averages -= totalAverage
    totalAverage *= 0.10
    
    plt.axhline(totalAverage, c = 'black')
    plt.axhline(-totalAverage, c = 'black')
    
    legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
                              label = 'Thresholds (within 10% of the average)',
                              markerfacecolor = 'black', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                              label = 'Luminance difference from mean',
                              markerfacecolor = 'red', markersize = 10)]
    
    plt.legend(handles = legend_elements)
    plt.plot(frames, averages, c = 'red')
    plt.xlabel('Time (frames)')
    plt.ylabel('Luminance difference from average')
    plt.title('Luminance fluctuation vs time of ' + fileName)
    plt.show()
    
    print(totalAverage)
    fire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
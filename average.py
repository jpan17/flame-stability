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
    
    for row in range(220, 270):
        for col in range(430, 480):
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
        
    fileName = "flame-spray-02.avi"
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
            # print(type(averageLine(frame)))
            averages.append(averageLine(frame))
            frames.append(frameCount)
                
            if cv2.waitKey(25) == ord('q'):
                break
                
        else:
            break

    totalAverage = sum(averages) / len(averages)
    averages -= totalAverage
    first = totalAverage * 0.15
    second = totalAverage * 0.25
    
    plt.axhline(first, c = 'black')
    plt.axhline(-first, c = 'black')
    plt.axhline(-second, c = 'black')
    plt.axhline(second, c = 'black')
    
    legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
                              label = 'Thresholds (within 15%, 25% of the average)',
                              markerfacecolor = 'black', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                              label = 'Luminance difference from mean',
                              markerfacecolor = 'red', markersize = 10)]
    
    plt.legend(handles = legend_elements, fontsize = 18)
    plt.plot(frames, averages, c = 'red')
    plt.xlabel('Time (frames)', fontsize = 24)
    plt.ylabel('Luminance difference from average', fontsize = 24)
    plt.title('Luminance fluctuation vs time of ' + fileName, fontsize = 24)
    plt.show()
    
    # print(totalAverage)
    fire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
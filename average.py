import cv2
import csv
import numpy as np 
import pandas
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
    videoCount = 0
    
    totalAverage = 0
    averages = []
    
    for i in range(0, len(df['File name'])):
        
        videoCount += 1
        
        fileName = df['File name'][i]
        print(fileName)
        fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
        ret, frame = fire.read()
        height, width, channels = frame.shape
        frameWidth = width
        frameHeight = height
        
        if ret == True:
            frameCount += 1
            if videoCount == 1:
                cv2.imshow('default', frame)
                averages.append(averageLine(frame))
            
            if cv2.waitKey(25) == ord('q'):
                break
            
        else:
            break
        
    totalAverage = sum(averages) / len(averages)
    print(totalAverage)
    fire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
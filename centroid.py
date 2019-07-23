import cv2
import csv
import numpy as np
import pandas
# =========================================================================== #

def centroid():

    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0
    videoCount = 0
    
    for i in range(0, len(df['File name'])):
        
        videoCount += 1
        
        fileName = df['File name'][i]
        fire = cv2.VideoCapture('./threshold60/threshold60-' + fileName)
        
        ret, frame = fire.read()
        height, width, channels = frame.shape
        frameWidth = width
        frameHeight = height
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            
            if ret == True:
                
                frameCount += 1
                
                img = frame
                cropped_img = img[:, 0:500]
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                moment = cv2.moments(gray_img)
                X = int(moment["m10"]/moment["m00"])
                Y = int(moment["m01"]/moment["m00"])
  
                cv2.circle(img, (X, Y), 5, (255, 255, 255), -1)
                cv2.putText(img, "centroid", (X + 15, Y + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)      
                    
                cv2.imshow("Center of the Image", img)
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
            
    fire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    centroid()
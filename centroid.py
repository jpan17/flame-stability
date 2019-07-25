import cv2
import csv
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# =========================================================================== #



def centroid():

    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0
    videoCount = 0
    
    wXCentroids = []
    rXCentroids = []
    bXCentroids = []
    wYCentroids = []
    rYCentroids = []
    bYCentroids = []
    
    frames = []
    
    # for i in range(0, len(df['File name'])):
        
    videoCount += 1
    fileName = "flame-spray-1.avi"
    # fileName = df['File name'][i]
    redFire = cv2.VideoCapture('./threshold60/threshold60-' + fileName)
    blueFire = cv2.VideoCapture('./threshold20/threshold20-' + fileName)
    whiteFire = cv2.VideoCapture('./threshold100/threshold100-' + fileName)
        
    ret, frame = blueFire.read()
    height, width, channels = frame.shape
    frameWidth = width
    frameHeight = height
        
        # out = cv2.VideoWriter('centroid-' + df['File name'][i], cv2.VideoWriter_fourcc(*'XVID'),
        #                       30, (frameWidth, frameHeight))
        
    if (whiteFire.isOpened() == False):
        print("Error opening video file or stream")
            
    while (whiteFire.isOpened() and blueFire.isOpened() and redFire.isOpened()):
            
        bRet, bFrame = blueFire.read()
        rRet, rFrame = redFire.read()
        wRet, wFrame = whiteFire.read()
            
        if bRet == True:
                
            frameCount += 1
                
            blended = cv2.addWeighted(bFrame, 0.5, rFrame, 0.5, 0)
            moreBlended = cv2.addWeighted(blended, 0.7, wFrame, 0.3, 0)
                
            # blue white red
            bImg = bFrame
            bImg = bImg[:, 0:480]
            b_gray_img = cv2.cvtColor(bImg, cv2.COLOR_BGR2GRAY)
            moment = cv2.moments(b_gray_img)
            if moment['m00'] != 0: 
                bX = int(moment["m10"]/moment["m00"])
                bY = int(moment["m01"]/moment["m00"])
            else:
                bX, bY = 0, 0
                    
            rImg = rFrame
            rImg = rImg[:, 0:480]
            r_gray_img = cv2.cvtColor(rImg, cv2.COLOR_BGR2GRAY)
            moment = cv2.moments(r_gray_img)
            if moment['m00'] != 0: 
                rX = int(moment["m10"]/moment["m00"])
                rY = int(moment["m01"]/moment["m00"])
            else:
                rX, rY = 0, 0
                    
            wImg = wFrame
            wImg = wImg[:, 0:480]
            w_gray_img = cv2.cvtColor(wImg, cv2.COLOR_BGR2GRAY)
            moment = cv2.moments(w_gray_img)
            if moment['m00'] != 0: 
                wX = int(moment["m10"]/moment["m00"])
                wY = int(moment["m01"]/moment["m00"])
            else:
                wX, wY = 0, 0
                    
            cv2.circle(moreBlended, (wX, wY), 5, (255, 255, 255), -1)
                # cv2.putText(moreBlended, "core centroid", (wX + 15, wY + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            cv2.circle(moreBlended, (rX, rY), 5, (255, 255, 255), -1)
                # cv2.putText(moreBlended, "inner centroid", (rX + 15, rY + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            cv2.circle(moreBlended, (bX, bY), 5, (255, 255, 255), -1)
                # cv2.putText(moreBlended, "outer centroid", (bX + 15, bY + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            wXCentroids.append(wX)
            wYCentroids.append(wY)
            rXCentroids.append(rX)
            rYCentroids.append(rY)
            bXCentroids.append(bX)
            bYCentroids.append(bY)
                
            frames.append(frameCount)
                
            cv2.imshow("centroids", moreBlended)
                
            # out.write(moreBlended)
                
                
            if cv2.waitKey(25) == ord('q'):
                break
                
        else:
            break
            

    wXAverage = sum(wXCentroids) / len(wXCentroids)
    wYAverage = sum(wYCentroids) / len(wYCentroids)
    rXAverage = sum(rXCentroids) / len(rXCentroids)
    rYAverage = sum(rYCentroids) / len(rYCentroids)
    bXAverage = sum(bXCentroids) / len(bXCentroids)
    bYAverage = sum(bYCentroids) / len(bYCentroids)
    
    wXCentroids -= wXAverage
    wYCentroids -= wYAverage
    rXCentroids -= rXAverage
    rYCentroids -= rYAverage
    bXCentroids -= bXAverage
    bYCentroids -= bYAverage
    
    plt.plot(frames, math.sqrt(np.square(wXCentroids) + np.square(wYCentroids)), c = "gold")
    plt.xlabel('Time (frames)')
    plt.ylabel('Centroid Position Absolute Difference (from average)')
    plt.title('Centroid Position Fluctuation from Average vs Time of ')
    plt.show()
    
    redFire.release()
    blueFire.release()
    whiteFire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    centroid()
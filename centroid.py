import cv2
import csv
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from sklearn.preprocessing import StandardScaler
import flameTest.twoComponentPCA as twoComponentPCA
import statistics
# =========================================================================== #

def standardize(array):
    standardized = StandardScaler().fit_transform(array)
    return standardized

def centroid():

    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0
    videoCount = 0
    
    videoCentroids = []
    features = []
    
    wXCentroids = []
    rXCentroids = []
    bXCentroids = []
    wYCentroids = []
    rYCentroids = []
    bYCentroids = []
    stability = []
    videos = []
    
    allWX = []
    allWY = []
    allRX = []
    allRY = []
    allBX = []
    allBY = []
    
    frames = []
    
    for i in range(0, len(df['File name'])):
            
        videoCount += 1
        fileName = df['File name'][i]
        print(fileName)
        stability.append(df['Stability10'][i])
        # fileName = df['File name'][i]
        redFire = cv2.VideoCapture('./threshold60/threshold60-' + fileName)
        blueFire = cv2.VideoCapture('./threshold20/threshold20-' + fileName)
        whiteFire = cv2.VideoCapture('./threshold100/threshold100-' + fileName)
            
        ret, frame = blueFire.read()
        height, width, channels = frame.shape
        frameWidth = width
        frameHeight = height
        numFrames = 0
            
        if i > 0:
            # print(len(videoCentroids))
            # features.append(videoCentroids)
            features.append(mean(allWX))
            features.append(mean(allWY))
            features.append(mean(allRX))
            features.append(mean(allRY))
            features.append(mean(allBX))
            features.append(mean(allBY))
            features.append(stdev(allWX))
            features.append(stdev(allWY))
            features.append(stdev(allRX))
            features.append(stdev(allRY))
            features.append(stdev(allBX))
            features.append(stdev(allBY))
            # videoCentroids = []
            
            videos.append(1)
            # out = cv2.VideoWriter('centroid-' + df['File name'][i], cv2.VideoWriter_fourcc(*'XVID'),
            #                       30, (frameWidth, frameHeight))
            
        if (whiteFire.isOpened() == False):
            print("Error opening video file or stream")
                
        while (whiteFire.isOpened() and blueFire.isOpened() and redFire.isOpened()) and numFrames < 250:
                
            bRet, bFrame = blueFire.read()
            rRet, rFrame = redFire.read()
            wRet, wFrame = whiteFire.read()
                
            if bRet == True:
                    
                frameCount += 1
                
                numFrames += 1
                    
                blended = cv2.addWeighted(bFrame, 0.5, rFrame, 0.5, 0)
                moreBlended = cv2.addWeighted(blended, 0.7, wFrame, 0.3, 0)
                    
                # blue white red
                bImg = bFrame
                bImg = bImg[:, 0:480]
                b_gray_img = cv2.cvtColor(bImg, cv2.COLOR_BGR2GRAY)
                moment = cv2.moments(b_gray_img)
                if moment['m00'] != 0: 
                    bX = moment["m10"]/moment["m00"]
                    bY = moment["m01"]/moment["m00"]
                else:
                    bX, bY = 0, 0
                        
                rImg = rFrame
                rImg = rImg[:, 0:480]
                r_gray_img = cv2.cvtColor(rImg, cv2.COLOR_BGR2GRAY)
                moment = cv2.moments(r_gray_img)
                if moment['m00'] != 0: 
                    rX = moment["m10"]/moment["m00"]
                    rY = moment["m01"]/moment["m00"]
                else:
                    rX, rY = 0, 0
                        
                wImg = wFrame
                wImg = wImg[:, 0:480]
                w_gray_img = cv2.cvtColor(wImg, cv2.COLOR_BGR2GRAY)
                moment = cv2.moments(w_gray_img)
                if moment['m00'] != 0: 
                    wX = moment["m10"]/moment["m00"]
                    wY = moment["m01"]/moment["m00"]
                else:
                    wX, wY = 0, 0
                        
                cv2.circle(moreBlended, (int(wX), int(wY)), 5, (255, 255, 255), -1)
                    # cv2.putText(moreBlended, "core centroid", (wX + 15, wY + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                cv2.circle(moreBlended, (int(rX), int(rY)), 5, (255, 255, 255), -1)
                    # cv2.putText(moreBlended, "inner centroid", (rX + 15, rY + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                cv2.circle(moreBlended, (int(bX), int(bY)), 5, (255, 255, 255), -1)
                    # cv2.putText(moreBlended, "outer centroid", (bX + 15, bY + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                wXCentroids.append(wX)
                wYCentroids.append(wY)
                rXCentroids.append(rX)
                rYCentroids.append(rY)
                bXCentroids.append(bX)
                bYCentroids.append(bY)
                
                videoCentroids.append(wX)
                videoCentroids.append(wY)
                videoCentroids.append(rX)
                videoCentroids.append(rY)
                videoCentroids.append(bX)
                videoCentroids.append(bY)
                
                allWX.append(wX)
                allWY.append(wY)
                allRX.append(rX)
                allRY.append(rY)
                allBX.append(bX)
                allBY.append(bY)
                                        
                frames.append(frameCount)
                    
                cv2.imshow("centroids", moreBlended)
                    
                # out.write(moreBlended)
                    
                    
                if cv2.waitKey(25) == ord('q'):
                    break
                    
            else:
                break
    
    # wXAverage = sum(wXCentroids) / len(wXCentroids)
    # wYAverage = sum(wYCentroids) / len(wYCentroids)
    # rXAverage = sum(rXCentroids) / len(rXCentroids)
    # rYAverage = sum(rYCentroids) / len(rYCentroids)
    # bXAverage = sum(bXCentroids) / len(bXCentroids)
    # bYAverage = sum(bYCentroids) / len(bYCentroids)
    
    # wXCentroids[:] = [x - wXAverage for x in wXCentroids]
    # wYCentroids[:] = [x - wYAverage for x in wYCentroids]
    # rXCentroids[:] = [x - rXAverage for x in rXCentroids]
    # rYCentroids[:] = [x - rYAverage for x in rYCentroids]
    # bXCentroids[:] = [x - bXAverage for x in bXCentroids]
    # bYCentroids[:] = [x - bYAverage for x in bYCentroids]
    
    # [i**2 for i in wXCentroids]
    # [i**2 for i in wYCentroids]
    # wCentroids = [sum(i) for i in zip(wXCentroids, wYCentroids)]
    
    # [i**2 for i in rXCentroids]
    # [i**2 for i in rYCentroids]
    # rCentroids = [sum(i) for i in zip(rXCentroids, rYCentroids)]
    
    # [i**2 for i in bXCentroids]
    # [i**2 for i in bYCentroids]
    # bCentroids = [sum(i) for i in zip(bXCentroids, bYCentroids)]
    
    # legend_elements = [Line2D([0],[0], marker = 'o', color = 'gold', 
    #                           label = 'Core',
    #                           markerfacecolor = 'gold', markersize = 10),
    #                    Line2D([0],[0], marker = 'o', color = 'crimson',
    #                           label = 'Inner',
    #                           markerfacecolor = 'crimson', markersize = 10),
    #                    Line2D([0],[0], marker = 'o', color = 'blue',
    #                           label = 'Outer',
    #                           markerfacecolor = 'blue', markersize = 10)]
    
    # plt.legend(handles = legend_elements)
    
    # plt.plot(frames, [i**0.5 for i in wCentroids], c = "gold")
    # plt.plot(frames, [i**0.5 for i in rCentroids], c = "crimson")
    # plt.plot(frames, [i**0.5 for i in bCentroids], c = "blue")
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Centroid Position Absolute Difference (from average)')
    # plt.title('Centroid Position Fluctuation from Average vs Time of ' + fileName)
    # plt.show()
    
    print(len(features))
    print(len(features[0]))
    features = standardize(features)
    twoComponentPCA.applyPCA(features, frameCount, '', videos, stability)
    
    redFire.release()
    blueFire.release()
    whiteFire.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    centroid()
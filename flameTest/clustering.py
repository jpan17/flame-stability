import cv2
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation 
from scipy.spatial import ConvexHull
import seaborn as sns
from array import *
import sys
import statistics
# =========================================================================== #

cluster1x = []
cluster1y = []
cluster2x = []
cluster2y = []
cluster3x = []
cluster3y = []
cluster4x = []
cluster4y = []
cluster5x = []
cluster5y = []
# =========================================================================== #

def encircle(x, y, ax = None, **kw):
    if not ax: ax = plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)
    
# =========================================================================== #

def luminance(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    frameL, frameA, frameB = cv2.split(frame)
    
    imageHeight = len(frameL[1])
    imageWidth = len(frameL[0])
    actualWidth = int(imageWidth)
    actualHeight = int(imageHeight * 7 / 15)
    
    rowStart = 0
    rowEnd = actualHeight
    colStart = 0
    colEnd = actualWidth
    pixelNum = actualHeight * actualWidth
    
    total = 0

    for row in range(rowStart, rowEnd):
        for col in range(colStart, colEnd):
            total += abs(frameL[row][col])
            
     # sum up luminance differences and divide by number of pixels
    # together = sum(difference)
    total /= pixelNum
    
    # print(difference)
    return total

# =========================================================================== #
     
# compares the difference in luminance between two frames
def diffLum(previousFrame, currentFrame):
    previous = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2LAB)
    current = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2LAB)
    previousL, previousA, previousB = cv2.split(previous)
    currentL, currentA, currentB = cv2.split(current)
    
    difference = 0
    imageHeight = len(previousL)
    imageWidth = len(previousL[0])
    actualWidth = int(imageWidth)
    actualHeight = int(imageHeight * 7 / 15)
    rowStart = 0
    rowEnd = actualHeight
    colStart = 0
    colEnd = actualWidth
    pixelNum = actualHeight * actualWidth

    for row in range(rowStart, rowEnd):
        for col in range(colStart, colEnd):
            difference += abs(int(currentL[row][col]) - int(previousL[row][col]))
            
    # sum up luminance differences and divide by number of pixels
    # together = sum(difference)
    difference /= pixelNum
    
    # print(difference)
    return difference
# =========================================================================== #

# applies k-means clustering with time on the x-axis and average luminance 
# difference on the y-axis
def cluster_time_lum(array, rate):
    luminanceDifference = np.array(array)
    
    # meanshift = MeanShift()
    # meanshift.fit(luminanceDifference)

    affinity = AffinityPropagation(damping = 0.5)
    affinity.fit(luminanceDifference)

    # kmeans = KMeans(n_clusters = 3)

    # kmeans.fit(luminanceDifference)
    # print(kmeans.cluster_centers_)
    
    for i in range(0, len(luminanceDifference)):
        if i >= 0 and i <= 100 / rate:
            plt.scatter(luminanceDifference[i,0], luminanceDifference[i,1], c = "yellow")       
        elif i >= 100 / rate and i <= 260 / rate: 
            plt.scatter(luminanceDifference[i,0], luminanceDifference[i,1], c = "orange")
        else:
            plt.scatter(luminanceDifference[i,0], luminanceDifference[i,1], c = "crimson")
    
    for i in range(0, len(luminanceDifference)):
        if affinity.labels_[i] == 0: 
            cluster1x.append(luminanceDifference[i,0])
            cluster1y.append(luminanceDifference[i,1])
        
        elif affinity.labels_[i] == 1:
            cluster2x.append(luminanceDifference[i,0])
            cluster2y.append(luminanceDifference[i,1])
        
        elif affinity.labels_[i] == 2:
            cluster3x.append(luminanceDifference[i,0])
            cluster3y.append(luminanceDifference[i,1])
        
        elif affinity.labels_[i] == 3:
            cluster4x.append(luminanceDifference[i,0])
            cluster4y.append(luminanceDifference[i,1])
        else:
            cluster5x.append(luminanceDifference[i,0])
            cluster5y.append(luminanceDifference[i,1])
        
    plt.scatter(affinity.cluster_centers_[:,0], affinity.cluster_centers_[:,1],
                color = 'black')
    
    if cluster1x and cluster1y and len(cluster1x) > 2:
        encircle(cluster1x, cluster1y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster2x and cluster2y and len(cluster2x) > 2:
        encircle(cluster2x, cluster2y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster3x and cluster3y and len(cluster3x) > 2:
        encircle(cluster3x, cluster3y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster4x and cluster4y and len(cluster3x) > 2:
        encircle(cluster4x, cluster4y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster5x and cluster5y and len(cluster5x) > 2:
        encircle(cluster5x, cluster5y, ec = "orange", fc = "gold", alpha = 0.2)

    
    plt.xlabel("Frame Count")
    plt.ylabel("Average Luminance Difference Over n Frames")
    plt.title("Frame Count vs. Luminance Difference, kmeans = 3, n = " + 
              str(rate))
    
    # plt.set_cmap("Wistia")
    # cbar = plt.colorbar(cmap = "Wistia", label = "Frame Progression", ticks = [0, 1])
    # cbar.set_ticklabels(['Begin', 'End'])
    plt.show()
    
    return

# =========================================================================== #

# applies k-means clustering with one set of frames on the x-axis and another
# set (the frame right after it) on the y-axis
def cluster_frames(array, rate):
    
    framesDifference = np.array(array)

    kmeans = KMeans(n_clusters = 3)

    kmeans.fit(framesDifference)
    print(kmeans.cluster_centers_)
    
    for i in range(0, len(framesDifference)):
        if i >= 0 and i <= 100 / rate:
            plt.scatter(framesDifference[i,0], framesDifference[i,1], c = "yellow")       
        elif i >= 100 / rate and i <= 260 / rate: 
            plt.scatter(framesDifference[i,0], framesDifference[i,1], c = "orange")
        else:
            plt.scatter(framesDifference[i,0], framesDifference[i,1], c = "crimson") 
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                color = 'black')
    
    for i in range(0, len(framesDifference)):
        if kmeans.labels_[i] == 0: 
            cluster1x.append(framesDifference[i,0])
            cluster1y.append(framesDifference[i,1])
        
        elif kmeans.labels_[i] == 1:
            cluster2x.append(framesDifference[i,0])
            cluster2y.append(framesDifference[i,1])
        
        elif kmeans.labels_[i] == 2:
            cluster3x.append(framesDifference[i,0])
            cluster3y.append(framesDifference[i,1])
        
        else:
            cluster4x.append(framesDifference[i,0])
            cluster4y.append(framesDifference[i,1])
            
    if cluster1x and cluster1y and len(cluster1x) >= 3:
        encircle(cluster1x, cluster1y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster2x and cluster2y and len(cluster2x) >= 3:
        encircle(cluster2x, cluster2y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster3x and cluster3y and len(cluster3x) >= 3:
        encircle(cluster3x, cluster3y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster4x and cluster4y and len(cluster4x) >= 3:
        encircle(cluster4x, cluster4y, ec = "orange", fc = "gold", alpha = 0.2)
        
    plt.xlabel("Frame 1")
    plt.ylabel("Frame 2")
    plt.title("Frame 1 Luminance Difference vs. Frame 2 Luminance Difference, kmeans = 3, n = " + 
              str(rate))
    
    # plt.set_cmap("Wistia")
    # cbar = plt.colorbar(cmap = "Wistia", label = "Frame Progression", ticks = [0, 1])
    # cbar.set_ticklabels(['Begin', 'End'])
    plt.show()
    
    return

# =========================================================================== #

# applies k-means clustering with one set of frames on the x-axis and another
# set (the frame right after it) on the y-axis
def cluster_mean(array, rate):
    
    framesMean = np.array(array)

    kmeans = KMeans(n_clusters = 4)

    kmeans.fit(framesMean)
    print(kmeans.cluster_centers_)

    for i in range(0, len(framesMean)):
        if i >= 0 and i <= 100 / rate:
            plt.scatter(framesMean[i,0], framesMean[i,1], c = "yellow")       
        elif i >= 100 / rate and i <= 260 / rate: 
            plt.scatter(framesMean[i,0], framesMean[i,1], c = "orange")
        else:
            plt.scatter(framesMean[i,0], framesMean[i,1], c = "crimson")
            
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                color = 'black')
    
    for i in range(0, len(framesMean)):
        if kmeans.labels_[i] == 0: 
            cluster1x.append(framesMean[i,0])
            cluster1y.append(framesMean[i,1])
        
        elif kmeans.labels_[i] == 1:
            cluster2x.append(framesMean[i,0])
            cluster2y.append(framesMean[i,1])
        
        elif kmeans.labels_[i] == 2:
            cluster3x.append(framesMean[i,0])
            cluster3y.append(framesMean[i,1])
        
        else:
            cluster4x.append(framesMean[i,0])
            cluster4y.append(framesMean[i,1])
            
    if cluster1x and cluster1y and len(cluster1x) >= 3:
        encircle(cluster1x, cluster1y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster2x and cluster2y and len(cluster2x) >= 3:
        encircle(cluster2x, cluster2y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster3x and cluster3y and len(cluster3x) >= 3:
        encircle(cluster3x, cluster3y, ec = "orange", fc = "gold", alpha = 0.2)
    if cluster4x and cluster4y and len(cluster4x) >= 3:
        encircle(cluster4x, cluster4y, ec = "orange", fc = "gold", alpha = 0.2)
        
    plt.xlabel("Mean (over n frames)")
    plt.ylabel("Standard Deviation (over n frames)")
    plt.title("Frames Mean vs. Frames Standard Deviation, kmeans = 3, n = " + 
              str(rate))
    
    # plt.set_cmap("Wistia")
    # cbar = plt.colorbar(cmap = "Wistia", label = "Frame Progression", ticks = [0, 1])
    # cbar.set_ticklabels(['Begin', 'End'])
    plt.show()
    
    return

# =========================================================================== #

# read in the video and establish dimensions
fire = cv2.VideoCapture('fire.mp4')

# check if video opened successfully
if (fire.isOpened() == False):
    print("Error opening video stream or file")
    
choice = input("What would you like to do with this video clip? 0 = kmeans" + 
               " with time, 1 = kmeans with mean/std, 2 = kmeans with two frames" +
                "\n")
    
# counts the number of frames in the video 
ret, frame = fire.read()
cv2.imshow('Frame',frame)
frameCount = 1
previousFrame = frame.copy()
previousLum = luminance(previousFrame)
cv2.imshow('Frame', previousFrame)

frames = []
difflist = []
array = []
diff = 0
rate = int(input("How many frames are you averaging over?\n"))
frameBegin = 0
frameEnd = 400
meanLums = []
frameLuminances = []

# display the video until 'q' is pressed or until it terminates
while (fire.isOpened()):
    ret, frame = fire.read()
    
    frameCount += 1
    if ret == True:
        
        currentFrame = frame.copy()
        cv2.imshow('Frame', currentFrame)
        
        if frameCount <= frameEnd and frameCount >= frameBegin:
            
            if choice == "0":
                diff += diffLum(previousFrame, currentFrame)
                # every 10 frames, compare current frame with the previous frame
                if frameCount % rate == 0:      
                    difflist.append(diff / rate)
                    frames.append(frameCount)
                    
                    array.append([frameCount, diff])
                    
                    diff = 0
                    
            elif choice == "1":
                    frameLuminances.append(luminance(previousFrame))
                    if frameCount % rate == 0:
                        meanLums.append([np.mean(frameLuminances), np.std(frameLuminances)])
                        frameLuminances = []
            elif choice == "2":
                if frameCount == 2:
                        diffPrevious = diffLum(previousFrame, currentFrame)
                if frameCount % rate == 0:
                    diff = diffLum(previousFrame, currentFrame)
                    array.append([diff, diffPrevious])
                    diffPrevious = diff
            else:
                break
                    
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
        previousFrame = currentFrame
    
    else:
        break

if choice == "0":
    cluster_time_lum(array, rate)
if choice == "1":
    cluster_mean(meanLums, rate)
if choice == "2":
    cluster_frames(array, rate)

fire.release()

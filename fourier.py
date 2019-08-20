import cv2
import csv
import numpy as np 
import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.fftpack import fft, ifft
# =========================================================================== #

def averageBox(image):
    
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

def fourier():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0
    videoCount = 0
    
    features = []
    videos = []
    
    averages = []
   
    frames = []
    stabilities = []
    
    fouriers = []
    fourierMeans = []
    fourierMax = []
    
    # for i in range(0, len(df['File name'])):   
        
<<<<<<< HEAD
    videoCount += 1
    
    # fileName = df['File name'][i]
    fileName = 'flame-spray-40.avi'
    # stabilities.append(df['box'][i])
    stabilities.append(0)
    print(fileName)
    
    redFire = cv2.VideoCapture('./threshold60/threshold60-' + fileName)
    blueFire = cv2.VideoCapture('./threshold20/threshold20-' + fileName)
    whiteFire = cv2.VideoCapture('./threshold100/threshold100-' + fileName)
    
    ret, frame = blueFire.read()
    height, width, channels = frame.shape
    frameWidth = width
    frameHeight = height
    
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
=======
        videoCount += 1
        
        fileName = df['File name'][i]
        # fileName = 'flame-spray-33.avi'
        stabilities.append(df['box'][i])
        # stabilities.append(0)
        print(fileName)
        
        fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
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
                cv2.imshow('Default', frame)
                
                averages.append(averageBox(frame))
                frames.append(frameCount)
>>>>>>> expert-opinions
                    
            cv2.circle(moreBlended, (int(rX), int(rY)), 5, (255, 255, 255), -1)
                    
<<<<<<< HEAD
            cv2.circle(moreBlended, (int(bX), int(bY)), 5, (255, 255, 255), -1)
                
            wXCentroids.append(wX)
            wYCentroids.append(wY)
            rXCentroids.append(rX)
            rYCentroids.append(rY)
            bXCentroids.append(bX)
            bYCentroids.append(bY)
                
            frames.append(frameCount)
            
            cv2.imshow("Centroids", moreBlended)
=======
            else:  
                
                average = sum(averages) / len(averages) 
                averages[:] = [x - average for x in averages]    
                
                N = len(averages)
                fourier = fft(averages) / N # 1 / N is a normalization factor
                fourier = fourier[range(int(N / 2))]
>>>>>>> expert-opinions
                
            if cv2.waitKey(25) == ord('q'):
                break
                
<<<<<<< HEAD
        else:
            wXAverage = sum(wXCentroids) / len(wXCentroids)
            wYAverage = sum(wYCentroids) / len(wYCentroids)
            rXAverage = sum(rXCentroids) / len(rXCentroids)
            rYAverage = sum(rYCentroids) / len(rYCentroids)
            bXAverage = sum(bXCentroids) / len(bXCentroids)
            bYAverage = sum(bYCentroids) / len(bYCentroids)
            
            wXCentroids[:] = [x - wXAverage for x in wXCentroids]
            wYCentroids[:] = [x - wYAverage for x in wYCentroids]
            rXCentroids[:] = [x - rXAverage for x in rXCentroids]
            rYCentroids[:] = [x - rYAverage for x in rYCentroids]
            bXCentroids[:] = [x - bXAverage for x in bXCentroids]
            bYCentroids[:] = [x - bYAverage for x in bYCentroids]
            
            [i**2 for i in wXCentroids]
            [i**2 for i in wYCentroids]
            wCentroids = [sum(i) for i in zip(wXCentroids, wYCentroids)]
            
            [i**2 for i in rXCentroids]
            [i**2 for i in rYCentroids]
            rCentroids = [sum(i) for i in zip(rXCentroids, rYCentroids)]
            
            [i**2 for i in bXCentroids]
            [i**2 for i in bYCentroids]
            bCentroids = [sum(i) for i in zip(bXCentroids, bYCentroids)]
            
            [i**0.5 for i in wCentroids]
            [i**0.5 for i in rCentroids]
            [i**0.5 for i in bCentroids]
            # legend_elements = [Line2D([0],[0], marker = 'o', color = 'gold', 
            #                         label = 'Core',
            #                         markerfacecolor = 'gold', markersize = 10),
            #                 Line2D([0],[0], marker = 'o', color = 'crimson',
            #                         label = 'Inner',
            #                         markerfacecolor = 'crimson', markersize = 10),
            #                 Line2D([0],[0], marker = 'o', color = 'blue',
            #                         label = 'Outer',
            #                         markerfacecolor = 'blue', markersize = 10)]
            
            # plt.figure(1)
            # plt.plot(frames, wCentroids, c = "gold")
            # plt.legend(handles = legend_elements)
            
            # plt.plot(frames, rCentroids, c = "crimson")
            # plt.plot(frames, bCentroids, c = "blue")
            # plt.xlabel('Time (frames)')
            # plt.ylabel('Centroid Position Absolute Difference (from average)')
            # plt.title('Centroid Position Fluctuation from Average vs Time of ' + fileName)
            # plt.show()
            
            # plt.figure(2) 
            # plt.plot(frames, wCentroids)
            # plt.show()
            
            # plt.figure(3)    
            N = len(wCentroids)
            fourier = fft(wCentroids) / N # 1 / N is a normalization factor
            fourier = fourier[range(int(N / 2))]
            
            # sampling rate
            S = 30 
            # frequency
            T = 1 / S 
            k = np.arange(N)
            Ts = N/S
            freq = k/Ts
            freq = freq[range(int(N/2))]
            
            f = np.linspace(0, N * T, N / 2) # time vector

            result = np.argmax(fourier)
            print("Result = ", str(result))
            frequency = f[result]
            print("Frequency = " + str(frequency))
            fouriers.append(frequency)
            videos.append(videoCount)
            print("Video Count = " + str(videoCount))
            
            wXCentroids = []
            rXCentroids = []
            bXCentroids = []
            wYCentroids = []
            rYCentroids = []
            bYCentroids = []
            
            plt.ylabel("Amplitude")
            plt.xlabel("Frequency (Hz)")
            plt.title("Fourier Transform for Core Centroid Mean Deviations")
            plt.plot(freq, np.abs(fourier))  
            plt.show()
            break
=======
                f = np.linspace(0, N * T, N / 2) # time vector
                # fourier = np.abs(fourier)
                # print(fourier)
                result = np.argmax(fourier)
                # print("Result = ", str(result))
                fourierMax.append(np.max(fourier))
                frequency = f[result]
                # print("Frequency = " + str(frequency))
                fouriers.append(frequency)
                videos.append(videoCount)
                # print("Video Count = " + str(videoCount))
                
                averages = []
                
                fourier = np.abs(fourier)
                fourierAverage = sum(fourier) / len(fourier)
                # print(result)
                print(fourierAverage)
                fourierMeans.append(fourierAverage)
                
                print(np.max(fourier) / fourierAverage)
                
                # plt.ylabel("Amplitude", fontsize = 24)
                # plt.xlabel("Frequency (Hz)", fontsize = 24)
                # plt.title("Fourier Transform for Bounding Box Luminosity Mean Deviations", fontsize = 24)
                # plt.plot(freq, np.abs(fourier))  
                # plt.show()
                break
>>>>>>> expert-opinions
    
    print(fouriers)
    for i in range(0, len(stabilities)):
        if stabilities[i] > 1.25:
            plt.scatter(fouriers[i], fourierMax[i] / fourierMeans[i], c = 'blue')
        elif stabilities[i] > .5:
            plt.scatter(fouriers[i], fourierMax[i] / fourierMeans[i], c = 'purple')
        else:
            plt.scatter(fouriers[i], fourierMax[i] / fourierMeans[i], c = 'red')
    
    plt.title('Fourier Transform Video Number vs Frequency', fontsize = 24)
    plt.xlabel('Frequency (Hz)', fontsize = 24)
    plt.ylabel('Normalized Maximum Amplitude', fontsize = 24)
    
    legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
                            label = 'Stable',
                            markerfacecolor = 'blue', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                            label = 'Unstable',
                            markerfacecolor = 'red', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                            label = 'Uncertain',
                            markerfacecolor = 'purple', markersize = 10)]

    plt.legend(handles = legend_elements, fontsize = 20)
    
    plt.show()
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    fourier()
    
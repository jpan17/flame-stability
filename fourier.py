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
    
    for i in range(0, len(df['File name'])):   
        
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
                    
                if cv2.waitKey(25) == ord('q'):
                    break
                    
            else:  
                
                average = sum(averages) / len(averages) 
                averages[:] = [x - average for x in averages]    
                
                N = len(averages)
                fourier = fft(averages) / N # 1 / N is a normalization factor
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
                # fourier = np.abs(fourier)
                # print(fourier)
                result = np.argmax(fourier)
                print("Result = ", str(result))
                fourierMax.append(result)
                frequency = f[result]
                print("Frequency = " + str(frequency))
                fouriers.append(frequency)
                videos.append(videoCount)
                print("Video Count = " + str(videoCount))
                
                averages = []
                
                fourier = np.abs(fourier)
                fourierAverage = sum(fouriers) / len(fouriers)
                fourierMeans.append(fourierAverage)
                
                # plt.ylabel("Amplitude", fontsize = 24)
                # plt.xlabel("Frequency (Hz)", fontsize = 24)
                # plt.title("Fourier Transform for Bounding Box Luminosity Mean Deviations", fontsize = 24)
                # plt.plot(freq, np.abs(fourier))  
                # plt.show()
                break
    
    print(fouriers)
    for i in range(0, len(stabilities)):
        if stabilities[i] > 1.25:
            plt.scatter(fourierMax[i], fourierMeans[i], c = 'blue')
        elif stabilities[i] > .5:
            plt.scatter(fourierMax[i], fourierMeans[i], c = 'purple')
        else:
            plt.scatter(fourierMax[i], fourierMeans[i], c = 'red')
    
    plt.title('Fourier Transform Max Amplitude vs Mean Amplitude', fontsize = 24)
    plt.xlabel('Max Amplitude', fontsize = 24)
    plt.ylabel('Mean Amplitude', fontsize = 24)
    
    legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
                            label = 'Stable',
                            markerfacecolor = 'blue', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                            label = 'Unstable',
                            markerfacecolor = 'red', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                            label = 'Uncertain',
                            markerfacecolor = 'purple', markersize = 10)]

    plt.legend(handles = legend_elements)
    
    plt.show()
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    fourier()
    
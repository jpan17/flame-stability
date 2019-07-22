import cv2
import csv
import numpy as np
import pandas
from sklearn.cluster import MiniBatchKMeans
# =========================================================================== #

def quantization():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    frameCount = 0 
    videoCount = 0
    
    for i in range(0, len(df['File name'])):
        
        videoCount += 1
        
        fileName = df['File name'][i]
        fire = cv2.VideoCapture('./fireFiles/' + fileName)
        
        ret, frame = fire.read()
        height, width, channels = frame.shape
        frameWidth = width
        frameHeight = height
        out = cv2.VideoWriter('quantization-' + fileName, cv2.VideoWriter_fourcc(*'XVID'),
                              30, (frameWidth, frameHeight))
        
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        while (fire.isOpened()):
            
            ret, frame = fire.read()
            
            if ret == True:
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                frame = frame.reshape((width * height, 3))
                clt = MiniBatchKMeans(n_clusters = 3)
                labels = clt.fit_predict(frame)
                quant = clt.cluster_centers_.astype('uint8')[labels]
                
                quant = quant.reshape((height, width, 3))
                frame = frame.reshape((height, width, 3))
                
                quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
                
                cv2.imshow('quant', quant)
                
                out.write(quant)
                
                frameCount += 1
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break
    
    fire.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    quantization()
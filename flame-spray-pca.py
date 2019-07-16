import cv2
import csv
import pandas
from sklearn.preprocessing import StandardScaler
import flameTest.luminance as luminance
import flameTest.twoComponentPCA as twoComponentPCA
# =========================================================================== #

# standardize the values in array
def standardize(array):
    standardized = StandardScaler().fit_transform(array)
    return standardized

# =========================================================================== #

# main function
def main():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    features = []
    frameCount = 0    
    videos = []
    
    for i in range(0, len(df['File name'])):
        
        numFrames = 0
        
        # read in video
        fire = cv2.VideoCapture('./fireFiles/' + df['File name'][i])
        
        # print error message if you can't read it in
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        # initialize video variables
        ret, frame = fire.read()
        height, width, channels = frame.shape
        vidHeight = height
        vidWidth = width 
        test = ''
        
        # display the video until 'q' is pressed or until it terminates
        while (fire.isOpened()):
            ret, frame = fire.read()
            frameCount += 1
            
            if ret == True:
                cv2.imshow('Fire', frame)
                
                if frameCount % 15 == 0: 
                    temp = luminance.lumArray(frame, vidHeight, vidWidth)
                    features.append(temp)
                    numFrames += 1
                
                # terminates the video before it finishes
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                videos.append(numFrames)
                break
            
    features = standardize(features)
    print(frameCount)
    print(features.shape)
    
    twoComponentPCA.applyPCA(features, frameCount, 'test', videos)
        
    fire.release()
    cv2.destroyAllWindows()
   
        
if __name__ == "__main__":
    main()
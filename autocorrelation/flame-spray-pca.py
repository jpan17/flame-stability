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
    stability = []
    temp = []
    tempStability = 1
    
    for i in range(0, len(df['File name']) - 1):
        
        numFrames = 0
        
        # read in video
        fire = cv2.VideoCapture('./fireFiles/' + df['File name'][i])
        print(df['File name'][i])
        # print(frameCount)
        # if i > 0 and frameCount % 10 == 0: 
        #     print(len(temp))
        #     features.append(temp)
        #     videos.append(1)
        #     stability.append(tempStability)
        
        # print error message if you can't read it in
        if (fire.isOpened() == False):
            print("Error opening video file or stream")
            
        # initialize video variables
        ret, frame = fire.read()
        height, width, channels = frame.shape
        vidHeight = height
        vidWidth = width 
        test = ''
        tempStability = int(df['boxStability'][i])
        
        # display the video until 'q' is pressed or until it terminates
        while (fire.isOpened() and numFrames < 250):
            ret, frame = fire.read()
            
            if ret == True:
                cv2.imshow('Fire', frame)
                
                frameCount += 1
                temp += luminance.lumArray(frame, vidHeight, vidWidth)
                
                if frameCount % 10 == 0: 
                    numFrames += 1
                    features.append(temp)
                    temp = []
                    videos.append(1)
                    stability.append(tempStability)
                
                # terminates the video before it finishes
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                # videos.append(numFrames)
                # temp = []
                break
            
    features = standardize(features)
    print(frameCount)
    print(features.shape)
    
    twoComponentPCA.applyPCA(features, frameCount, '', videos,
                             stability)
        
    fire.release()
    cv2.destroyAllWindows()
   
        
if __name__ == "__main__":
    main()
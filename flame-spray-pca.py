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
    
    for i in range(0, len(df['File name'])):
        
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
        frameCount = 0
        test = ''
        
        # display the video until 'q' is pressed or until it terminates
        while (fire.isOpened()):
            ret, frame = fire.read()
            frameCount += 1
            
            if ret == True:
                cv2.imshow('Fire', frame)
                
                temp = luminance.lumArray(frame, vidHeight, vidWidth)
                features.append(temp)
                
                # terminates the video before it finishes
                if cv2.waitKey(25) == ord('q') or frameCount > 2800:
                    break
                
            else:
                break
            
    features = standardize(features)
    
    print(features.shape)
    
    twoComponentPCA.applyPCA(features, frameCount, 'test')
        
    fire.release()
    cv2.destroyAllWindows()
   
        
if __name__ == "__main__":
    main()
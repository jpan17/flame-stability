import cv2
from sklearn.preprocessing import StandardScaler
import luminance
import rgb
import twoComponentPCA
import multiplePCA
# =========================================================================== #

# standardize values in array
def standardize (array):
    
    standardized = StandardScaler().fit_transform(array)
    return standardized

# =========================================================================== #

# main function
def main():
    
    features = []
    
    # read in video
    fire = cv2.VideoCapture('fire.mp4')
    
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
    
    choice = input("What features do you want to use? 1) luminance 2) r 3) g" +
                   " 4) b 5) RGB \n")
    
    if choice == '1': 
        test = "Luminance"
    elif choice == '2':
        test = "R"
    elif choice == '3':
        test = "G"
    elif choice == '4':
        test = 'B'
    elif choice == '5':
        test = 'RGB'
    else:
        test = ''
    
    # display the video until 'q' is pressed or until it terminates
    while (fire.isOpened()):
        ret, frame = fire.read()
        frameCount += 1
        
        if ret == True:
            cv2.imshow('Fire', frame)
            
            # handles user decision
            if choice == "1":
                temp = luminance.lumArray(frame, vidHeight, vidWidth)
                features.append(temp)
            elif choice == "2" or choice == "3" or choice == "4":
                temp = rgb.rgbArray(frame, vidHeight, vidWidth, test)
                features.append(temp)
            else:
                print("Your choice is invalid or RGB hasn't been implemented" +
                      " yet.")
                break
               
            # terminates the video before it finishes
            if cv2.waitKey(25) == ord('q'):
                break
            
        else:
            break
    
    features = standardize(features) 
    
    choice2 = input("What would you like to do with the data? 1) Apply " +
                    "2-component PCA 2) Apply 3-component PCA or 3) Examine " +
                    "explained variance number of components \n")
    
    if choice2 == '1':
        twoComponentPCA.applyPCA(features, frameCount, test)
    elif choice2 == '2':
        print("Not implemented yet")
        # threeComponentPCA.applyPCA(features, frameCount, test)
    elif choice2 == '3':
        componentCount = input("How many components do you want? \n")
        multiplePCA.applyPCA(features, int(componentCount), test)
    else:
        print("Gotta choose something else")
      
    
    fire.release()
    cv2.destroyAllWindows()
   
        
if __name__ == "__main__":
    main()
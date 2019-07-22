import cv2
import csv
import numpy as np
import pandas
# =========================================================================== #

def main():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    frameCount = 0
    videoCount = 0
    
    for i in range(0, len(df['File name'])):
        
        videoCount += 1
        
        background = 'threshold20-' + df['File name'][i]
        foreground = 'threshold60-' + df['File name'][i]
        
        blueFire = cv2.VideoCapture('./threshold20/' + background)
        redFire = cv2.VideoCapture('./threshold60/' + foreground)
        
        bRet, bFrame = blueFire.read()
        height, width, channels = bFrame.shape
        frameWidth = width
        frameHeight = height
        out = cv2.VideoWriter('together-' + df['File name'][i], cv2.VideoWriter_fourcc(*'XVID'),
                              30, (frameWidth, frameHeight))

        if (blueFire.isOpened() == False or redFire.isOpened() == False):
            print("Error opening video file or stream")
            
        while (blueFire.isOpened() and redFire.isOpened()):
            
            bRet, bFrame = blueFire.read()
            rRet, rFrame = redFire.read()
                        
            if bRet == True and rRet == True:
                
                cv2.imshow('Blue Fire', bFrame)
                cv2.imshow('Red Fire', rFrame)
                
                blended = cv2.addWeighted(bFrame, 0.5, rFrame, 0.5, 0)
                cv2.imshow('Blended', blended)
                
                out.write(blended)
                
                frameCount += 1
                
                if cv2.waitKey(25) == ord('q'):
                    break
                
            else:
                break

    blueFire.release()
    redFire.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
            
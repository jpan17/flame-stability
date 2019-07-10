import cv2
import numpy as num 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift 
from sklearn.cluster import AffinityPropagation
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
# =========================================================================== #

# for each pixel in an image, calculate luminance and store it in a 1D array
def lumArray (image, height, width): 
    # array to return
    luminances = []
    
    # extract luminances
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    frameL, frameA, frameB = cv2.split(frame)
    
    # iterate through all pixels and add to array
    for row in range(0, height):
        for col in range(0, width):
            luminances.append(frameL[row][col])
    
    return luminances

# =========================================================================== #

# for each pixel in an image, calculate RGB and store it in a 1D array 
def rgbArray (image, height, width, test):
    # array to return
    rgbs = []
    
    # extract RGB values
    frameB, frameG, frameR = cv2.split(image)
    
    # code for all RGB
    if test == 'RGB':
        for row in range(0, height):
            for col in range(0, width):
                rgbs.append(frameR[row][col])
                rgbs.append(frameG[row][col])
                rgbs.append(frameB[row][col])
        
    # iterate through all pixels and add to array
    # R
    if test == 'R':
        for rowR in range(0, height):
            for colR in range(0, width):
                rgbs.append(frameR[rowR][colR])
            
    # G
    if test == 'G':
        for rowG in range(0, height):
            for colG in range(0, width):
                rgbs.append(frameG[rowG][colG])
            
    # B
    if test == 'B':
        for rowB in range(0, height):
            for colB in range(0, width):
                rgbs.append(frameB[rowB][colB])

    return rgbs

# =========================================================================== #

# standardize the values in this array
def standardize (array):
    
    standardized = StandardScaler().fit_transform(array)

    return standardized
# =========================================================================== #

# apply PCA with 2 components
def applyPCA (array, frameCount):
    
    pca = PCA(n_components = 2)
    
    principalComponents = pca.fit_transform(array)
    
    # print(len(principalComponents))
    
    # plot the figure if it's not rgb
    if test != "luminance":
        for i in range (0, len(principalComponents)):
            if i >= 0 and i <= 100:
                plt.scatter(principalComponents[i,0], principalComponents[i,1],
                            c = 'yellow')
            elif i > 100 and i < 260:
                plt.scatter(principalComponents[i,0], principalComponents[i,1],
                            c = "orange")
            else:
                plt.scatter(principalComponents[i,0], principalComponents[i,1],
                            c = "crimson")
    # code for all rgb; shouldn't get used for now
    else:
        print("you shouldn't be here")

        for i in range (0, len(principalComponents)):
            if i >= 0 and i <= 3 * 100:
                plt.scatter(principalComponents[i,0], principalComponents[i,1],
                            c = 'yellow')
            elif i > 3 * 100 and i < 3 * 260:
                plt.scatter(principalComponents[i,0], principalComponents[i,1],
                            c = "orange")
            else:
                plt.scatter(principalComponents[i,0], principalComponents[i,1],
                            c = "crimson")              

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # decide how you want to cluster them
    choice = input("Do you want to apply 1) kmeans 2) affinity propogation" +
                   " or 3) mean shift to this data? Press enter to skip" +
                   " cluster step.\n")
    
    legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
                              label = 'Beginning (stable)',
                              markerfacecolor = 'yellow', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                              label = 'Middle (unstable)',
                              markerfacecolor = 'orange', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w', 
                              label = 'End (stable)',
                              markerfacecolor = 'crimson', markersize = 10)]
    
    plt.legend(handles = legend_elements)
    print(pca.explained_variance_ratio_)
    
    if choice == "1":
        clusterNum = input("How many clusters do you want? (no more than 5) \n")
        applyKmeans(principalComponents, int(clusterNum), test)
    elif choice == "2":
        applyAffinity(principalComponents, test)
    elif choice == "3":
        applyMeanShift(principalComponents, test)
    else:
        plt.title("2 Component PCA on Frame " + test + " Values (per pixel)")
        
    plt.show()
    return

# =========================================================================== #

# apply kmeans algorithm to data
def applyKmeans(array, clusterNumber, test):
    
    
    kmeans = KMeans(n_clusters = clusterNumber)
    kmeans.fit(array)
    
    # instnatiate  dictionaries
    clustersX = {}
    clustersY = {}
    
    for i in range(0, len(kmeans.cluster_centers_)):
        clustersX[i] = []
        clustersY[i] = []
    
    # .. and fill them
    for i in range(0, len(array)):
        label = kmeans.labels_[i]
        clustersX[label].append(array[i, 0])
        clustersY[label].append(array[i, 1])
    
  
    # encircle clusters
    for i in range(0, len(kmeans.cluster_centers_)):
        if len(clustersX[i]) > 2:
            encircle(clustersX[i], clustersY[i], ec = "orange", fc = "gold", 
                    alpha = 0.2)
    
    plt.title("2 Component PCA on Frame " + test + " Values (per pixel) with " + 
              "kmeans = " + str(clusterNumber))
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
                color = 'black')
    return

# =========================================================================== #

# apply affinity propagation algorithm to data
def applyAffinity(array, test):
    affinity = AffinityPropagation(damping = 0.5)
    affinity.fit(array)
    
    # instnatiate  dictionaries
    clustersX = {}
    clustersY = {}
    
    for i in range(0, len(affinity.cluster_centers_)):
        clustersX[i] = []
        clustersY[i] = []
    
    # .. and fill them
    for i in range(0, len(array)):
        label = affinity.labels_[i]
        clustersX[label].append(array[i, 0])
        clustersY[label].append(array[i, 1])
    
  
    # encircle clusters
    for i in range(0, len(affinity.cluster_centers_)):
        if len(clustersX[i]) > 2:
            encircle(clustersX[i], clustersY[i], ec = "orange", fc = "gold", 
                    alpha = 0.2)
    
    plt.title("2 Component PCA on Frame " + test + " Values (per pixel) with " + 
              "affinity propagation")
    plt.scatter(affinity.cluster_centers_[:,0], affinity.cluster_centers_[:,1],
                color = 'black')
    return

# =========================================================================== #

# apply mean shift algorithm to data
def applyMeanShift(array, test):
    meanshift = MeanShift(bandwidth = 80, min_bin_freq = 5)
    meanshift.fit(array)
    
    # instnatiate  dictionaries
    clustersX = {}
    clustersY = {}
    
    for i in range(0, len(meanshift.cluster_centers_)):
        clustersX[i] = []
        clustersY[i] = []
    
    # .. and fill them
    for i in range(0, len(array)):
        label = meanshift.labels_[i]
        clustersX[label].append(array[i, 0])
        clustersY[label].append(array[i, 1])
    
  
    # encircle clusters
    for i in range(0, len(meanshift.cluster_centers_)):
        if len(clustersX[i]) > 2:
            encircle(clustersX[i], clustersY[i], ec = "orange", fc = "gold", 
                    alpha = 0.2)
    
    plt.title("2 Component PCA on Frame " + test + " Values (per pixel) with " + 
              "mean shift")
    plt.scatter(meanshift.cluster_centers_[:,0], meanshift.cluster_centers_[:,1],
                color = 'black')
    
    return
# =========================================================================== #

# helper function to make cluster visualization easier
def encircle(x, y, ax = None, **kw):
    if not ax: ax = plt.gca()
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

# =========================================================================== #

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
                temp = lumArray(frame, vidHeight, vidWidth)
                features.append(temp)
            elif choice == "2" or choice == "3" or choice == "4":
                temp = rgbArray(frame, vidHeight, vidWidth, test)
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
    applyPCA(features, frameCount)
    return
    
    
    fire.release()
    cv2.destroyAllWindows()
   
        
if __name__ == "__main__":
    main()
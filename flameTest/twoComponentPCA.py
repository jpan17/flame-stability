import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import spatial
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.cm as cm
# =========================================================================== #

# apply PCA with 2 components
def applyPCA (array, frameCount, test, videos, stability):
    
    pca = PCA(n_components = 2)
    
    principalComponents = pca.fit_transform(array)
    frames = 0
    
    for i in range(0, len(videos) - 1):
        isStable = stability[i]
        for j in range(0, videos[i]):
            if isStable > 1.25: 
                plt.scatter(principalComponents[frames, 0],
                            principalComponents[frames, 1], c = 'blue')
            elif isStable > .5:
                plt.scatter(principalComponents[frames, 0],
                            principalComponents[frames, 1], c = 'purple')
            else:
                plt.scatter(principalComponents[frames, 0],
                            principalComponents[frames, 1], c = 'red')
            
            frames += 1
    # colors = cm.rainbow(np.linspace(0, 1, len(videos)))
    # frames = 0
    # print(len(colors))
    # print(colors[0])
    
    # for c in range(0, len(colors)):
    #     color = colors[c]
    #     for i in range(0, videos[c]):
    #         plt.scatter(principalComponents[frames, 0], 
    #                     principalComponents[frames, 1], c = color)
    #         frames += 1 
        
    # plot the figure if it's not rgb
    # if test != "luminance":
    #     for i in range (0, len(principalComponents)):
    #         if i >= 0 and i <= 100:
    #             plt.scatter(principalComponents[i,0], principalComponents[i,1],
    #                         c = 'yellow')
    #         elif i > 100 and i < 260:
    #             plt.scatter(principalComponents[i,0], principalComponents[i,1],
    #                         c = "orange")
    #         else:
    #             plt.scatter(principalComponents[i,0], principalComponents[i,1],
    #                         c = "crimson")
    # # code for all rgb; shouldn't get used for now
    # else:
    #     print("you shouldn't be here")

    #     for i in range (0, len(principalComponents)):
    #         if i >= 0 and i <= 3 * 100:
    #             plt.scatter(principalComponents[i,0], principalComponents[i,1],
    #                         c = 'yellow')
    #         elif i > 3 * 100 and i < 3 * 260:
    #             plt.scatter(principalComponents[i,0], principalComponents[i,1],
    #                         c = "orange")
    #         else:
    #             plt.scatter(principalComponents[i,0], principalComponents[i,1],
    #                         c = "crimson")              

    plt.xlabel("Principal Component 1", fontsize = 24)
    plt.ylabel("Principal Component 2", fontsize = 24)
    
    # decide how you want to cluster them
    choice = input("Do you want to apply 1) kmeans 2) affinity propogation" +
                   " or 3) mean shift to this data? Press enter to skip" +
                   " cluster step.\n")
    
    # legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
    #                           label = 'Beginning (stable)',
    #                           markerfacecolor = 'yellow', markersize = 10),
    #                    Line2D([0],[0], marker = 'o', color = 'w',
    #                           label = 'Middle (unstable)',
    #                           markerfacecolor = 'orange', markersize = 10),
    #                    Line2D([0],[0], marker = 'o', color = 'w', 
    #                           label = 'End (stable)',
    #                           markerfacecolor = 'crimson', markersize = 10)]
    
    # plt.legend(handles = legend_elements)
    
     
    legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
                              label = 'Stable',
                              markerfacecolor = 'blue', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                              label = 'Unstable',
                              markerfacecolor = 'red', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                              label = 'Uncertain',
                              markerfacecolor = 'purple', markersize = 10)]
    
    plt.legend(handles = legend_elements, fontsize = 18)
    print(pca.explained_variance_ratio_)
    
    if choice == "1":
        clusterNum = input("How many clusters do you want? (no more than 5) \n")
        applyKmeans(principalComponents, int(clusterNum), test)
    elif choice == "2":
        applyAffinity(principalComponents, test)
    elif choice == "3":
        applyMeanShift(principalComponents, test)
    else:
        # plt.title("2 Component PCA on " + test + " Pixel Values")
        plt.title("2 component PCA on Bounding Box Pixel Luminosity (per frame)", fontsize = 24)
        
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
    
    # plt.title("2 Component PCA on " + test + " Values (per pixel) with " + 
    #           "kmeans = " + str(clusterNumber))
    plt.title("2 component PCA on Bounding Box Pixel Luminosity (per frame)", fontsize = 24)
    
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
    
    plt.title("2 Component PCA on " + test + " Values (per pixel) with " + 
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
    
    plt.title("2 Component PCA on " + test + " Values (per pixel) with " + 
              "mean shift")
    plt.scatter(meanshift.cluster_centers_[:,0], meanshift.cluster_centers_[:,1],
                color = 'black')
    
    return

# =========================================================================== #
# helper function to make cluster visualization easier
def encircle(x, y, ax = None, **kw):
    if not ax: ax = plt.gca()
    p = np.c_[x, y]
    hull = spatial.qhull.ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)
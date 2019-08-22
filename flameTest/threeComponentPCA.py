from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import graph3d
# =========================================================================== #

def applyPCA (array, name):
    
    pca = PCA(n_components = 3)
    
    principalComponents = pca.fit_transform(array)
    
    fig = plt.figure()
    graph = fig.add_subplot(111, projection = '3d')
    
    for i in range(0, len(principalComponents)):
        if i >= 0 and i <= 100: 
            graph.scatter(principalComponents[i,0], principalComponents[i,1],
                        principalComponents[i,2], c = 'yellow')
        elif i > 100 and i <= 260:
            graph.scatter(principalComponents[i,0], principalComponents[i,1],
                        principalComponents[i,2], c = 'orange')
        else:
            graph.scatter(principalComponents[i,0], principalComponents[i,1],
                        principalComponents[i,2], c = 'crimson')
        
    user = input("Do you want to apply 1) kmeans 2) affinity propogation" + 
                 " or 3) mean shift to this data? Press enter to skip" + 
                 " cluster step. \n")
    
    graph.set_xlabel("Component 1")
    graph.set_ylabel("Component 2")
    graph.set_zlabel("Component 3")
    
    title = ''
    
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
    
    choice = ''
    if user == '1':
        choice = 'k-means'
        clusterNum = input("How many clusters do you want? (no more than 5) \n")
        applyKmeans(principalComponents, int(clusterNum), graph)
    elif user == '2':
        choice = 'affinity propogation'
        applyAffinity(principalComponents, graph)
    elif user == '3':
        choice = 'mean shift'
        applyMeanShift(principalComponents, graph)
    else:
        title = "3 Component PCA on Frame " + name + " Values (per pixel)"
        plt.title("3 Component PCA on Frame " + name + " Values (per pixel)")
        plt.show()
        return
        
    
    plt.title("3 Component PCA on Frame " + name + " Values (per pixel) with " +
              choice + " clustering")
    
    title = "3 Component PCA on Frame " + name + " Values (per pixel) with " + choice + " clustering"
    
    plt.show()
    return
    
# =========================================================================== #
    
# apply kmeans algorithm to data
def applyKmeans(array, clusterNumber, graph):
    
    
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
            
    graph.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
                kmeans.cluster_centers_[:,2], color = 'black')
    return

# =========================================================================== #

# apply affinity propagation algorithm to data
def applyAffinity(array, graph):
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
    
    graph.scatter(affinity.cluster_centers_[:,0], affinity.cluster_centers_[:,1],
                  affinity.cluster_centers_[:,2], color = 'black')
    return

# =========================================================================== #

# apply mean shift algorithm to data
def applyMeanShift(array, graph):
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
    
  
    graph.scatter(meanshift.cluster_centers_[:,0], meanshift.cluster_centers_[:,1],
                meanshift.cluster_centers_[:,2], color = 'black')
    
    return

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.lines import Line2D

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
        
    graph.set_xlabel("Component 1")
    graph.set_ylabel("Component 2")
    graph.set_zlabel("Component 3")
    plt.title("3 Component PCA on Frame " + name + " Values (per pixel)")
    
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
    
    plt.show()
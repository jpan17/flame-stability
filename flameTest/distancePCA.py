import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from scipy.spatial import distance
import statistics
import numpy as np

# plot explained variance vs component number
def applyPCA(array, componentCount, name):
    
    distances = []
    
    for i in range(0, componentCount):
        
        pca = PCA(n_components = componentCount + 1)
        principalComponents = pca.fit_transform(array)
        
        unstable = principalComponents[100:260]
        stable = principalComponents[0:100]
        
        stableMean = np.mean(stable)
        unstableMean = np.mean(unstable)
        
        dst = abs(distance.euclidean(stableMean, unstableMean))
    
        plt.scatter(i, dst, color = 'red')
    
    plt.title("Euclidean distance between stable and unstable conditions" + 
               " for " + str(componentCount) + " components")

    plt.xlabel("Component Number")
    plt.ylabel("Distance between the means")
    
    
    plt.show()
    return
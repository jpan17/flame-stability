import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# plot explained variance vs component number
def applyPCA (array, componentCount, name):
    pca = PCA(n_components = componentCount)
    
    principalComponents = pca.fit_transform(array)
    
    
    first = plt.subplot(121)
    # plot individual explained variances
    for i in range(0, len(pca.explained_variance_ratio_)):
        plt.scatter(i, pca.explained_variance_ratio_[i],
                    c = 'orange')    
    
    plt.title("Explained variance ratios for " + name + " for " +
              str(componentCount) + " components")
    plt.xlabel("Component Number")
    plt.ylabel("Explained variance ratio")
     
    cumulative = 0
    second = plt.subplot(122)
    # plot cumulative variances
    for i in range(0, len(pca.explained_variance_ratio_)):
        cumulative += pca.explained_variance_ratio_[i]
        plt.scatter(i, cumulative, c = 'red')
    
    plt.title("Cumulative explained variance ratios for " + name +
              " for " + str(componentCount) + " components")
    plt.xlabel("Component Number")
    plt.ylabel("Cumulative explained variance ratio")
    
    plt.show()
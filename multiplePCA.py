import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# plot explained variance vs component number
def applyPCA (array, componentCount):
    pca = PCA(n_components = componentCount)
    
    principalComponents = pca.fit_transform(array)
    
    
    plt.subplot(121)
    # plot individual pcas
    for i in range(0, len(pca.explained_variance_ratio_)):
        plt.scatter(i, pca.explained_variance_ratio_[i],
                    c = 'orange')
        
    
    cumulative = 0
    plt.subplot(122)
    for i in range(0, len(pca.explained_variance_ratio)):
        cumulative += pca.explained_variance_ratio_[i]
        plt.scatter(i, cumulative, c = 'red')
    
    plt.show()
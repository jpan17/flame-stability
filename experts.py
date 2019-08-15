import csv
import pandas
import statistics
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.lines import Line2D
# =========================================================================== #

def experts():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    joe = []
    debolina = []
    jess = []
    dante = []
    means = []
    boxStabilities = []
    centroidStabilities = []
    boxStabilities = []
    videos = []
    
    for i in range(0, len(df['File name'])):
        
        fileName = df['File name'][i]
        
        currMean = (df['jess'][i] + df['debolina'][i] + df['joe'][i] + df['dante'][i]) / 4
        means.append(currMean)    
        jess.append(df['jess'][i])
        debolina.append(df['debolina'][i])
        joe.append(df['joe'][i])
        dante.append(df['dante'][i])
        
        centroidStabilities.append(df['centroid'][i])
        boxStabilities.append(df['box'][i])
        videos.append(int(fileName[12:14]))

#     plt.scatter(videos, joe, c = 'cyan', alpha = .7, marker = 'o')
#     plt.scatter(videos, debolina, c = 'gold', alpha = .7, marker = 'o')
#     plt.scatter(videos, jess, c = 'crimson', alpha = 0.7, marker = 'o')
#     plt.scatter(videos, dante, c = 'green', alpha = 0.7, marker = 'o')
    axes = plt.gca()
    axes.set_xlim([0, 53])
    axes.set_ylim([-2, 4])
    
    legend_elements = [Line2D([0],[0], marker = 'o', color = 'w', 
                              label = 'Means',
                              markerfacecolor = 'black', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                              label = 'Bounding Box Stabilities (r = 0.89)',
                              markerfacecolor = 'violet', markersize = 10),
                       Line2D([0],[0], marker = 'o', color = 'w',
                              label = 'Centroid Stabilities (r = 0.40)',
                              markerfacecolor = 'orange', markersize = 10)]
                     #   Line2D([0],[0], marker = 'o', color = 'w',
                     #          label = 'Dante',
                     #          markerfacecolor = 'green', markersize = 10)]

    
    plt.legend(handles = legend_elements, fontsize = 18)
    
    plt.xlabel('Video Number', fontsize = 24)
    plt.ylabel('Stability (0 = unstable, 2 = stable)', fontsize = 24)
    plt.title('Flame Stability vs Video Number', fontsize = 24)
    
    plt.plot(videos, boxStabilities, c = 'violet')
    plt.plot(videos, centroidStabilities, c = 'orange')
    plt.plot(videos, means, c = 'black', linewidth = 2)

    print(np.corrcoef(means, boxStabilities))
    print(np.corrcoef(means, centroidStabilities))
    plt.show()
    
if __name__ == "__main__":
    experts()
        
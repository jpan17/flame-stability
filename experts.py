import csv
import pandas
import statistics
import matplotlib.pyplot as plt
# =========================================================================== #

def experts():
    
    df = pandas.read_csv('EtOH_flamemap.csv')
    
    joe = []
    debolina = []
    jess = []
    dante = []
    means = []
    
    videos = []
    
    for i in range(0, len(df['File name'])):
        
        fileName = df['File name'][i]
        
        currMean = (df['jess'][i] + df['debolina'][i] + df['joe'][i]) / 3
        means.append(currMean)    
        jess.append(df['jess'][i])
        debolina.append(df['debolina'][i])
        joe.append(df['joe'][i])
        videos.append(int(fileName[12:14]))

    plt.plot(videos, joe, c = 'blue')
    plt.plot(videos, debolina, c = 'gold')
    plt.plot(videos, jess, c = 'crimson')
    plt.plot(videos, means, c = 'black', linewidth = 2)
    print(means)
    plt.show()
    
if __name__ == "__main__":
    experts()
        
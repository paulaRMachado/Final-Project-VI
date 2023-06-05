import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def matrix_corr(df):
    """
    This finction show the correlation matrix/heatmap for the input data.
    :arg:
    df: It takes a dataframe with the input data for the model
    """
    plt.figure(figsize=(10, 8))
    sbn.heatmap(data=df.corr(), annot=True,cmap='coolwarm',fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig('image/plots/correlation_heatmap.png')
    plt.show()
    pass


def elbow_method(df):
    """
    This function calculate the sum of squared distances for each k value and plots graph
    :arg:
    df: dataframe with all the data.
    :returns:
    The function does not have a return. It saves the plot inside the plots folder.
    """
    k_values = range(1, 20)
    sse = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)  
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.plot(k_values, sse, linestyle='--', marker='o', color='red')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method')
    plt.savefig('image/plots/elbow_plot.png')
    plt.show()

def plot_results(centroid_dict, cluster_dict, X, epoch):
    nrows=1
    ncols=1
    fig, ax = plt.subplots(figsize=(10, 6), nrows=nrows, ncols=ncols)
    
    ax1 = plt.subplot(nrows, ncols, 1)
    ax1.scatter(X[:, 0], X[:, 1], c='tab:blue')
    for k in centroid_dict.keys():
        ax1.scatter(centroid_dict[k][0], centroid_dict[k][1], c='red', s=100)
    ax1.set_title("K-Means Clustering - Converged on Epoch: {}".format(epoch))
    plt.savefig('image/plots/K-Means_Clustering.png')
    plt.show()
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
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
    


def elbow_method(df):
    """
    This function calculate the sum of squared distances for each k value and plots graph
    
    :arg:
        df: dataframe with all the data.
    :returns:
        The function does not have a return. It saves the plot inside the plots folder.
    """
    k_values = range(1, 11)
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

def silhouette_plot(df):
    """
    """
    k_values = range(2, 11)
    silhouette_scores = []

    for k in k_values:
        # Perform clustering with K clusters
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(df)
        
        # Calculate the Silhouette Coefficient for the clustering result
        silhouette_coefficient = silhouette_score(df, cluster_labels)
        silhouette_scores.append(silhouette_coefficient)

    # Plot the line chart
    plt.figure(figsize=(10, 8))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs. Number of Clusters')
    plt.grid(True)
    plt.savefig('image/plots/Silhouette_Coefficient.png')
    plt.show()

def davies_boulding_plot(df):
    k_values = range(2, 11)
    davies_bouldin_scores = []
    for k in k_values:
        # Perform clustering with K clusters
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(df)
        
        # Calculate the Davies-Bouldin Index for the clustering result
        davies_bouldin_index = davies_bouldin_score(df, cluster_labels)
        davies_bouldin_scores.append(davies_bouldin_index)

    # Plot the line chart
    plt.figure(figsize=(10, 8))
    plt.plot(k_values, davies_bouldin_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index vs. Number of Clusters')
    plt.grid(True)
    plt.savefig('image/plots/Davies_Bouldin.png')
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
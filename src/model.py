import numpy as np
from sklearn.cluster import KMeans

def fit_Kmeans(X, k):

    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X)

    centroid_dict = {i: centroid for i, centroid in enumerate(kmeans.cluster_centers_)}
    cluster_dict = {i: np.where(kmeans.labels_ == i)[0].tolist() for i in range(k)}

    return centroid_dict, cluster_dict

def train(centroid_dict, cluster_dict, X):
    """
    Get the updated centroids and cluster assignments

    :args:
        centroid_dict (dict): A dictionary containing the initial centroids.
        cluster_dict (dict): A dictionary containing the initial cluster assignments.
        X (array-like): The original data points.
        
    :Returns:
        tuple: A tuple containing the final centroid dictionary, cluster dictionary, and the number of epochs trained.
    """
    k = len(centroid_dict)  # Get the number of clusters

    # Initialize a KMeans object with the initial centroids
    kmeans = KMeans(n_clusters=k, init=np.array(list(centroid_dict.values())))
    kmeans.fit(X)

    # Get the updated centroids and cluster assignments
    centroid_dict = {i: centroid for i, centroid in enumerate(kmeans.cluster_centers_)}
    cluster_dict = {i: np.where(kmeans.labels_ == i)[0].tolist() for i in range(k)}

    return centroid_dict, cluster_dict


def predict_and_find_nearest(centroid_dict, cluster_dict, new_data_point):
    """
    Predict the cluster index for a new data point and find the nearest point within the predicted cluster.

    Args:
        centroid_dict (dict): A dictionary containing the centroids.
        cluster_dict (dict): A dictionary containing the cluster assignments.
        new_data_point (array-like): The new data point.

    Returns:
        int: The predicted cluster index.
        array-like: The nearest point within the predicted cluster.
    """
    # Convert the new data point to a numpy array and ensure it's 2D
    new_data_point = np.array(new_data_point).reshape(1, -1)

    # Calculate the Euclidean distance between the new data point and each centroid
    distances = [np.linalg.norm(new_data_point - centroid.reshape(1, -1)) for centroid in centroid_dict.values()]

    # Find the index of the cluster with the minimum distance
    predicted_cluster_index = np.argmin(distances)

    # Get the data points within the predicted cluster
    data_points_in_cluster = cluster_dict[predicted_cluster_index]

    # Convert the data points to numpy arrays
    data_points_in_cluster = np.array(data_points_in_cluster)

    # Calculate the distances between the new data point and the data points in the cluster
    distances = [np.linalg.norm(new_data_point - data_point) for data_point in data_points_in_cluster]


  # Find the indices of the two points with the minimum distances
    nearest_point_indices = np.argsort(distances)[:2]

    # Get the nearest and second nearest points from the data points in the cluster
    nearest_point = data_points_in_cluster[nearest_point_indices[0]]
    second_nearest_point = data_points_in_cluster[nearest_point_indices[1]]

    return nearest_point, second_nearest_point
"""
    # Find the index of the point with the minimum distance
    nearest_point_index = np.argmin(distances)

    # Get the nearest point from the data points in the cluster
    nearest_point = data_points_in_cluster[nearest_point_index]

    return nearest_point

"""

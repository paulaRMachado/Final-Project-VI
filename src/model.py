import numpy as np
from sklearn.cluster import KMeans


def euclidian_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points in n-dimensional space.
    
    :Args:
        p1 (list or array-like): The coordinates of the first point.
        p2 (list or array-like): The coordinates of the second point.
        
    :Returns:
        float: The Euclidean distance between the two points.
    """
    
    return sum([(x-y)**2 for x,y in zip(p1, p2)])**(1/2)

def update_centroids(centroid_dict, cluster_dict, X):
    """
    Update the centroids based on the data points assigned to each cluster.
    
    :Args:
        centroid_dict (dict): A dictionary containing the current centroids.
        cluster_dict (dict): A dictionary containing the data points assigned to each cluster.
        X (array-like): The original data points.
        
    :Returns:
        tuple: A tuple containing the updated centroid dictionary and cluster dictionary.
    """
    
    for key in cluster_dict.keys():
        cluster_xs = [X[i] for i in cluster_dict[key]]

        if len(cluster_xs)!=0:
            new_cluster = np.mean(cluster_xs, axis=0)

        centroid_dict[key] = new_cluster
    return centroid_dict, cluster_dict

def update_clusters(centroid_dict, cluster_dict, X, initial=False):
    """
    Update the clusters by assigning data points to the nearest centroids.
    
    :Args:
        centroid_dict (dict): A dictionary containing the current centroids.
        cluster_dict (dict): A dictionary containing the data points assigned to each cluster.
        X (array-like): The original data points.
        initial (bool): Flag indicating whether it is the initial assignment of data points.
        
    Returns:
        tuple: A tuple containing the updated centroid dictionary and cluster dictionary.
    
    """
    if initial == False:
        for key in cluster_dict.keys():
            cluster_dict[key] = []

    for i,x in enumerate(X):
        min_dist = np.inf
        min_cluster = None
        for c in cluster_dict.keys():
            current_dist = euclidian_distance(x, centroid_dict[c])
            if current_dist < min_dist:
                min_cluster = c
                min_dist = current_dist
        cluster_dict[min_cluster].append(i)
    
    return centroid_dict, cluster_dict

def check_dicts_identical(centroid_dict, last_centroid_dict):
    """
    used
    """
    for x in centroid_dict.keys():
        if (centroid_dict[x] == last_centroid_dict[x]).all():
            print()
            continue
        else:
            return False
    return True

def train(centroid_dict, cluster_dict, X, epochs=100):
    """
    Train the k-means model by updating the centroids and clusters iteratively for a specified number of epochs.
    
    :args:
        centroid_dict (dict): A dictionary containing the initial centroids.
        cluster_dict (dict): A dictionary containing the initial cluster assignments.
        X (array-like): The original data points.
        epochs (int): The number of training epochs. Set default to 100 iterations
        
    :Returns:
        tuple: A tuple containing the final centroid dictionary, cluster dictionary, and the number of epochs trained.
    """
    for epoch in range(epochs):

        if epoch == 0:
            centroid_dict, cluster_dict = update_clusters(centroid_dict, cluster_dict, X, initial=True)
            continue

        last_centroid_dict = centroid_dict.copy()
        
        centroid_dict, cluster_dict = update_centroids(centroid_dict, cluster_dict, X)
        centroid_dict, cluster_dict = update_clusters(centroid_dict, cluster_dict, X)

        if check_dicts_identical(centroid_dict, last_centroid_dict):
            return centroid_dict, cluster_dict, epoch

        last_centroid_dict = centroid_dict.copy()
    return centroid_dict, cluster_dict, epoch



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

    # Find the index of the point with the minimum distance
    nearest_point_index = np.argmin(distances)

    # Get the nearest point from the data points in the cluster
    nearest_point = data_points_in_cluster[nearest_point_index]

    return nearest_point


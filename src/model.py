import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)  # Step 1: Initialize centroids
    
    for _ in range(max_iterations):
        # Step 2: Assign data points to the nearest centroids
        clusters = assign_data_points(data, centroids)
        
        # Step 3: Update centroids
        new_centroids = update_centroids(data, clusters, k)
        
        # Check convergence
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_data_points(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
    return new_centroids



def euclidian_distance(p1, p2):
    return sum([(x-y)**2 for x,y in zip(p1, p2)])**(1/2)

def update_centroids(centroid_dict, cluster_dict, X):
    X_dim = len(X[0])

    for key in cluster_dict.keys():
        cluster_xs = [X[i] for i in cluster_dict[key]]

        if len(cluster_xs)!=0:
            new_cluster = np.mean(cluster_xs, axis=0)

        centroid_dict[key] = new_cluster
    return centroid_dict, cluster_dict

def update_clusters(centroid_dict, cluster_dict, X, initial=False):
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
        for x in centroid_dict.keys():
            if (centroid_dict[x] == last_centroid_dict[x]).all():
                print()
                continue
            else:
                return False
        return True

def train(centroid_dict, cluster_dict, X, epochs):
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
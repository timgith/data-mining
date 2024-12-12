import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import seaborn as sns
import math, random

''' Gonzalez's algorithm, written by Tarun Sunkaraneni on Kaggle
link: https://www.kaggle.com/barelydedicated/gonzalez-algorithm.
This returns the centroids of the clusters'''
def gonzalez(data, cluster_num, technique = 'max'):
    clusters = []
    clusters.append(data[0]) # let us assign the first cluster point to be first point of the data
    while len(clusters) is not cluster_num:
        if technique is 'max':
            clusters.append(max_dist(data, clusters)) 
        if technique is 'norm':
            clusters.append(norm_dist(data, clusters)) 
        # we add the furthest point from ALL current clusters
    return (clusters)

def kMeans(data, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(data)
    cluster_centers = kmeans.cluster_centers_.tolist()
    return cluster_centers


'''Given the data and the centroids, return the clusters as indicies
of their respective data points'''
def find_clusters(centroids, data):
    clusters = []
    for i in range(len(centroids)):
        clusters.append([])
    cluster_lookup = {}
    i = 0
    for point1 in data:
        point1 = tuple(point1.tolist())
        min_dist = math.inf
        cluster = -1
        j = 0
        for point2 in centroids:
            point2 = tuple(point2)
            dist = distance.euclidean(point1, point2)
            if dist < min_dist:
                min_dist = dist
                cluster = j
            j += 1  
        clusters[cluster].append(i)
        cluster_lookup[i] = cluster
        i += 1
    return clusters, cluster_lookup

'''Given the data as an nparray and the precomputed clusters, find the
max distances between the points'''
def max_dist(data, clusters):
    distances = np.zeros(len(data)) # we will keep a cumulative distance measure for all points
    for cluster_id, cluster in enumerate(clusters):
        for point_id, point in enumerate(data):
            if distance.euclidean(point,cluster) == 0.0:
                distances[point_id] = -math.inf # this point is already a cluster (obselete)
            if not math.isinf(distances[point_id]):
                # if a point is not obselete, then we add the distance to its specific bin
                distances[point_id] = distances[point_id] + distance.euclidean(point,cluster) 
                # return the point which is furthest away from all the other clusters
    return data[np.argmax(distances)]

'''Greedy set cover algorithm, implemented by Martin Broadhurst
with a small refactor (explicit for loops rather than set comprehension)'''
def set_cover(universe, subsets,limit = None):
    if limit is None:
        limit = len(universe)
    """Find a family of subsets that covers the universal set"""
    elements = set()
    for s in subsets:
        for e in s:
            elements.add(e)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    compt = 0
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset
        compt+=1
        if compt==limit:
            break
    return cover

'''Create sets from the k-Nearest graph. This is referenced as
N in XXXXXX et al 2020'''
def find_N(kng):
    N = [None] * len(kng)
    for i in range(len(N)):
        # Find all indicies of 1, incidating a neighbor
        indices = [j for j, x in enumerate(kng[i]) if x == 1]
        N[i] = set(indices)
    return N

'''Find exemplar list based off set cover and epsilon neighbors
This is referenced as Y in XXXXXX et al 2002'''
def find_Y(S, N):
    Y = []
    i = 0
    for n in S:
        Y.append(N.index(n))
    return Y

'''Ground exemplars in respect to their partitions'''
def find_exemplars(Y, partitions):
    exemplars = []
    for i in range(len(partitions)):
        exemplars.append([])
    for exemplar in Y:
        i = 0
        for partition in partitions:
            if exemplar in partition:
                exemplars[i].append(exemplar)
                break
            i += 1
    count = 0
    for cluster in exemplars:
        count += len(cluster)
    return exemplars

'''For y in Y, reassign epsilon neighbors of y to the cluster of y so
long as they are not exemplars themselves'''
def reassign_clusters(Y, N, clusters, cluster_lookup):
    visited = []
    for exemplar in Y:
        neighbors = N[exemplar]
        for neighbor in neighbors:
            if neighbor not in Y and neighbor not in visited:
                clusters[cluster_lookup[neighbor]].remove(neighbor)
                clusters[cluster_lookup[exemplar]].append(neighbor)
                visited.append(neighbor)
    # Remove empty clusters
    filter(lambda a: a != [], clusters)
    return clusters

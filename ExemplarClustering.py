from ExemplarClusteringBackend import (gonzalez, find_clusters,
set_cover, find_N, find_Y, find_exemplars, reassign_clusters, kMeans)
from sklearn.neighbors import radius_neighbors_graph
import numpy as np
import sys


# REQUIREMENTS: sklearn, numpy, seaborn, scikit
class ExemplarCluster(object):
	'''
	Input:
		- Data:     nparray    Raw data in N-Space
		- Epsilon:  int        Number of neighbors in the epsilon graph
		- k:        int        k parameter for Gonzalez's method
	Output:
		- (array of int, array of int) clusters, exemplars
	'''
	def __init__(self, data, epsilon, k, algorithm='Gonzalez',limit=None,verbose=False):
		self.data = data
		self.epsilon = epsilon
		self.k = k
		self.verbose = verbose
		self.algorithm = algorithm
		self.exemplars = None
		self.clusters = None
		self.label_table = None
		self.limit = limit

	def fit(self,bounded = False):
		# Run Gonzalez to generate clusters
		if self.verbose:
			print ("Step 1/4: Running Gonzalez's Method")
		if self.algorithm == 'Gonzalez':
			centroids = gonzalez(self.data, self.k)
		if self.algorithm == 'kMeans':
			centroids = kMeans(self.data, self.k)
		clusters, cluster_lookup = find_clusters(centroids, self.data)
		# Create K-Nearest Graph
		if self.verbose:
			print ("Step 2/4: Creating Epsilon-Neighbor Graph")
		kng = radius_neighbors_graph(self.data, self.epsilon, mode='connectivity', include_self=True)
		kng = kng.toarray().tolist()
		# For some reason, ~0.1% of kngs do not include themselves
		if bounded == True:
						print("bounded version")
						for i in range(len(kng)):
								id_clust_i = cluster_lookup[i]
								kng[i][i] = 1.0
								for j in range(len(kng)):
										id_clust_j = cluster_lookup[j]
										if id_clust_i !=  id_clust_j:
												kng[i][j] = 0
		if self.verbose:
			max_sum = -1
			for row in kng:
				row_sum = sum(row)
				if row_sum > max_sum:
					max_sum = row_sum
			print ('INFO:')
			print ('    Largest neighborhood: ' + str(max_sum))
			print ('    Number of instances: ' + str(len(kng)))
		N = find_N(kng)
		self.neighbors_graph = N
		# Use Greedy set cover, keep track of the indicies of sets used to cover (exemplars)
		if self.verbose:
			print ("Step 3/4: Using Greedy Set Cover")
		universe = set(range(0, self.data.shape[0]))
		S = set_cover(universe, N,self.limit)
		Y = find_Y(S, N)
		self.exemplars = find_exemplars(Y, clusters)
		if self.verbose:
			print ("Step 4/4: Reassigning Clusters")
		# Reassign non-exemplar epsilon neighbords of exmplars to cluster of exemplars
		self.clusters = reassign_clusters(Y, N, clusters, cluster_lookup)

	def get_exemplars(self):
		if self.exemplars == None:
			print ("WARNING: Data not yet fit. To find exemplars, use .fit() method")
		return self.exemplars

	def get_clusters(self):
		if self.clusters == None:
			print ("WARNING: Data not yet fit. To find clusters, use .fit() method")
		return self.clusters

	def find_exemplar_neighborhood(self, unique=False, count=True):
		# Can only be done after data has been fit
		if self.exemplars == None:
			raise FitError('Must fit before finding neighborhood')
		if self.label_table is None:
			self.label_table = np.zeros([len(self.neighbors_graph),
				len(self.neighbors_graph)])
			for cluster in self.exemplars:
				for e in cluster:
					for n in self.neighbors_graph[e]:
						self.label_table[e][n] = 1
		counts = {}
		# Find the counts of occurances for rows and columns
		sum_array_r = np.sum(self.label_table, axis=1)
		sum_array_c = np.sum(self.label_table, axis=0)
		for cluster in self.exemplars:
			for e in cluster:
				if count:
					counts[e] = 0
				else:
					counts[e] = []
				if not unique:
					# Count number of neighbors
					if count:
						counts[e] = int(sum_array_r[e])
					else:
						counts[e] += list(np.nonzero(self.label_table[e]))
				else:
					for p in range(len(self.label_table[e])):
						# point p belongs to one exemplar nieghborhood
						if sum_array_c[p] == 1 and self.label_table[e][p] == 1:
							if count:
								counts[e] += 1
							else:
								counts[e] += [p]
		return counts

# Define exceptions
class Error(Exception):
	#Base class for exceptions in this module
	pass

class FitError(Error):
	def __init__(self, message):
		self.message = message

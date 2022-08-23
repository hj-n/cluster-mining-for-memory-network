from sklearn.neighbors import NearestNeighbors
import numpy as np

def run_knn(ts_list, metric):
	"""
	Run KNN algorithm on the time series data
	"""
	n_neighbors = int(np.sqrt(len(ts_list)))
	knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1).fit(ts_list)
	distances, indices = knn.kneighbors(ts_list)

	return distances[:, 1:], indices[:, 1:]

import numpy as np
import distance as dt
from tslearn.clustering import KShape 
import hdbscan


"""
List of functions that clusters the time series data.
Returns the (1) list of clusters and (2) list of seeds (centroid computed by the clustering algorithm).
If seed does not exists, return None
"""


def pm_memnet(ts_list, metric, threshold=0.25):
	"""
	The clustering and seed (key) extraction method used in PM_MEMNET paper
	"""
	cluster_list = []
	seed_list = []
	while(len(ts_list) > 0):
		target_idx = np.random.randint(len(ts_list))
		target_ts = ts_list[target_idx]
		if np.sum(np.abs(target_ts)) == 0:
			ts_list = np.delete(ts_list, target_idx, axis=0)
			continue
		dist_list = np.array([metric(target_ts, ts) < threshold for ts in ts_list])
		cluster = ts_list[dist_list]
		ts_list = ts_list[~dist_list]
		cluster_list.append(cluster)
		seed_list.append(target_ts)
	
	return cluster_list, seed_list

def hdbscan_clustering(ts_list, metric):
	"""
	The HDBSCAN cluster
	"""
	clusterer = hdbscan.HDBSCAN(metric=metric)
	labels = clusterer.fit_predict(ts_list)
	cluster_list = []
	for i in range(max(labels) + 1):
		cluster_list.append(ts_list[labels == i])
	return cluster_list, None

	


# def k_shape(ts_list, verbose=False, n_cluster=100):
# 	"""
# 	K-shape clustering algorithm
# 	"""
# 	k_shape_clustering = KShape(n_clusters=n_cluster, verbose=verbose)
# 	k_shape_clustering.fit(ts_list)
# 	return k_shape_clustering.labels_, k_shape_clustering.cluster_centers_



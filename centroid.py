import numpy as np

from tqdm import tqdm

def random_extraction(cluster_list):
	"""
	Randomly extract a seed from the cluster
	"""
	seed_list = []
	for cluster in cluster_list:
		seed_list.append(cluster[np.random.randint(len(cluster))])
	return seed_list

def medoid_extraction(cluster_list, metric):
	"""
	Extract the medoid from the cluster
	"""
	medoid_list = []
	for cluster in tqdm(cluster_list):
		dist_sum_list = []
		for ts1 in cluster:
			dist_sum = 0
			for ts2 in cluster:
				dist_sum += metric(ts1, ts2)
			dist_sum_list.append(dist_sum)
		medoid_list.append(cluster[np.argmin(dist_sum_list)])
		
	return medoid_list


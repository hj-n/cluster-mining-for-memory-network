import numpy as np
import distance as dt


def key_extraction(ts_list, threshold=0.25):
	cluster_list = []
	while(len(ts_list) > 0):
		target_idx = np.random.randint(len(ts_list))
		target_ts = ts_list[target_idx]
		if np.sum(np.abs(target_ts)) == 0:
			ts_list = np.delete(ts_list, target_idx, axis=0)
			continue
		dist_list = np.array([dt.cosine_similarity_with_fourier_distance(target_ts, ts) < threshold for ts in ts_list])
		cluster = ts_list[dist_list]
		ts_list = ts_list[~dist_list]
		#cluster_list.append(cluster)
		cluster_list.append(target_ts)
	
	return cluster_list


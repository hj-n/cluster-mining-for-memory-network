from _distance_metrics import *
from _read_data import *
from _window_list import *
from _compute_knn import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import os

def knn_distance_computation_single(identifier, window, stride, metric):
	"""
	Run KNN algorithm on the time series data
	"""
	data = read_data(identifier, window, stride)
	distances, indices = run_knn(data, metric)
	return distances, indices


def knn_distance_computation_all(identifier_list, window_list_dict, distance_metric_list, distance_metric_name_list):
	"""
	Run KNN algorithm for entire combination of time series data
	"""
	for identifier in identifier_list:
		for window in window_list_dict[identifier]:
			for stride in [int(window / 2), window]:
				for i, distance_metric in enumerate(distance_metric_list):
					path = f"{identifier}/{window}_{stride}_{distance_metric_name_list[i]}"
					index_path = f"./distances/{path}_index.npy"
					distance_path = f"./distances/{path}_distance.npy"
					if (os.path.exists(index_path)):
						print(f"{identifier}_{window}_{stride}_{distance_metric_name_list[i]} already exists" )
						continue
					distances, indices = knn_distance_computation_single(identifier, window, stride, distance_metric)
					## save distances and indices as npy files
					np.save(index_path, indices)
					np.save(distance_path, distances)

					print(f"{identifier}_{window}_{stride}_{distance_metric_name_list[i]} finished" )

knn_distance_computation_all(
	identifier_list=["elec", "exchange", "solar", "traffic"],
	window_list_dict=WINDOW_LIST_DICT,
	distance_metric_list=[cosine_similarity_distance, cosine_similarity_with_fourier_distance, dtw_distance],
	distance_metric_name_list=["cossim", "cossim_fft", "dtw"]
)
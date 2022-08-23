import numpy as np
from tslearn.metrics import dtw

from tqdm import tqdm

def pairwise_distance(ts_list, metric):
	pairwise_distance_list = []
	for i, ts_1 in tqdm(enumerate(ts_list)):
		for ts_2 in ts_list[i:]:
			pairwise_distance_list.append(metric(ts_1, ts_2))

	return pairwise_distance_list



def cosine_similarity_distance(series_1, series_2):
	## compute cosine similarity between two series
	sim = np.dot(series_1, series_2) / (np.linalg.norm(series_1) * np.linalg.norm(series_2))
	return 1 - sim


def cosine_similarity_with_fourier_distance(series_1, series_2):
	s1_fft = np.fft.fft(series_1)
	s2_fft = np.fft.fft(series_2)

	## stack the real and imaginary parts
	s1_fft_stack = np.hstack((s1_fft.real, s1_fft.imag))
	s2_fft_stack = np.hstack((s2_fft.real, s2_fft.imag))

	return cosine_similarity_distance(s1_fft_stack, s2_fft_stack)

def dtw_distance(series_1, series_2):
	return dtw(series_1, series_2)
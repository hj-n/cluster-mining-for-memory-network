import numpy as np
import pandas as pd

from tqdm import tqdm



def read_data(file_identifier, window=12, stride=6):
	if file_identifier == "elec":
		data = pd.read_csv("data/electricity/electricity.txt", header=None, sep=",")
	elif file_identifier == "exchange":
		data = pd.read_csv("data/exchange_rate/exchange_rate.txt", header=None, sep=",")
	elif file_identifier == "solar":
		data = pd.read_csv("data/solar_energe/solar_AL.txt", header=None, sep=",")
	elif file_identifier == "traffic":
		data = pd.read_csv("data/traffic/traffic.txt", header=None, sep=",")
	else:
		raise ValueError("Invalid file_identifier")


	data_np = data.to_numpy().T

	data_list = []
	for i in tqdm(range(0, data_np.shape[1], stride)):
		for j in range(0, data_np.shape[0]):
			if i + window <= data_np.shape[1]:
				data_list.append(data_np[j][i:i+window])
	
	return np.array(data_list)







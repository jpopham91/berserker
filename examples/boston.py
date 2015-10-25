from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

data = load_boston().data
n_samples = data.shape[0]
test_split = int(0.2*n_samples)
shuffled_data = data[np.random.permutation(n_samples)]

train = shuffled_data[:-test_split]
test = shuffled_data[test_split:]
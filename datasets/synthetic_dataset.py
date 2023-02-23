"""
Functions for generating nonlinear synthetic multiview data
"""
import numpy as np
from numpy.random import multivariate_normal, uniform

import torch
from torch.utils import data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_spd_matrix, make_low_rank_matrix


def random_nonlin_map(n_in, n_out, n_hidden, rank=1000):
	"""
	Reaturn a random nonlinear function parameterized by an MLP
	"""
	# Random MLP mapping
	W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
	W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
	W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
	# No biases
	b_0 = np.random.uniform(0, 0, (1, n_hidden))
	b_1 = np.random.uniform(0, 0, (1, n_hidden))
	b_2 = np.random.uniform(0, 0, (1, n_out))

	nlin_map = lambda x: np.matmul(
		ReLU(np.matmul(ReLU(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))), W_1) +
			 np.tile(b_1, (x.shape[0], 1))), W_2) + np.tile(b_2, (x.shape[0], 1))

	return nlin_map


def ReLU(x):
	return x * (x > 0)


def generate_synthetic_data(p: int, n_views: int, n: int, k: int, seed: int):
	"""
	Generate a nonlinear synthetic multiview dataset

	@param p: number of covariates per view
	@param n_views: number of views
	@param n: number of data points
	@param k: number of concepts
	@param seed: random generator seed
	@return: a design matrix of dimensions (@n, @n_views, @p), concept values and labels
	"""
	# Replicability
	np.random.seed(seed)

	# Generate covariates
	mu = uniform(-5, 5, p * n_views)
	sigma = make_spd_matrix(p * n_views, random_state=seed)
	X = multivariate_normal(mean=mu, cov=sigma, size=n)
	ss = StandardScaler()
	X = ss.fit_transform(X)
	# Produce different views
	X_views = np.zeros((n, n_views, p))
	for v in range(n_views):
		X_views[:, v] = X[:, (v * p):(v * p + p)]

	# Nonlinear maps
	g = random_nonlin_map(n_in=p * n_views, n_out=k, n_hidden=int((p * n_views + k) / 2))
	f = random_nonlin_map(n_in=k, n_out=1, n_hidden=int(k / 2))

	# Generate concepts
	c = g(X)
	tmp = np.tile(np.median(c, 0), (X.shape[0], 1))
	c = (c >= tmp) * 1.0

	# Generate labels
	y = f(c)
	tmp = np.tile(np.median(y, 0), (X.shape[0], 1))
	y = (y >= tmp) * 1.0

	return X_views, c, y


class SyntheticDataset(data.dataset.Dataset):
	"""
	Dataset class for the nonlinear synthetic multiview data
	"""
	def __init__(self, num_vars: int, num_views: int, num_points: int, num_predicates: int,
				 partial_predicates: bool = False, num_partial_predicates: int = None, indices: np.ndarray = None,
				 seed: int = 42):
		"""
		Initializes the dataset.

		@param num_vars: number of covariates per view
		@param num_views: number of views
		@param num_points: number of data points
		@param num_predicates: number of concepts
		@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
		@param num_partial_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
		@param indices: indices of the data points to be kept; the rest of the data points will be discarded
		@param seed: random generator seed
		"""
		# Shall a partial predicate set be used?
		if not partial_predicates:
			self.predicate_idx = np.arange(0, num_predicates)
		else:
			assert 0 < num_partial_predicates <= num_predicates
			np.random.seed(seed)
			self.predicate_idx = np.random.choice(
				a=np.arange(0, num_predicates), size=(num_partial_predicates, ), replace=False)

		self.X, self.c, self.y = generate_synthetic_data(p=num_vars, n_views=num_views, n=num_points, k=num_predicates,
														 seed=seed)

		if indices is not None:
			self.X = self.X[indices]
			self.c = self.c[indices]
			self.y = self.y[indices]

	def __getitem__(self, index):
		"""
		Returns points from the dataset

		@param index: index
		@return: a dictionary with the data; dict['images'] contains multiview features, dict['label'] contains
		target labels, dict['concepts'] contains concepts
		"""
		labels = self.y[index, 0]
		concepts = self.c[index, self.predicate_idx]
		features = self.X[index]

		return {'img_code': index, 'file_names': index, 'images': features, 'label': labels,
				'features': features, 'concepts': concepts}

	def __len__(self):
		return self.X.shape[0]


def get_synthetic_dataloaders(num_vars: int, num_views: int, num_points: int, num_predicates: int, batch_size: int,
							  num_workers: int, partial_predicates: bool = False, num_partial_predicates: int = None,
							  train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42):
	"""
	Constructs data loaders for the synthetic data

	@param num_vars: number of covariates per view
	@param num_views: number of views
	@param num_points: number of data points
	@param num_predicates: number of concepts
	@param batch_size: batch size
	@param num_workers: number of worker processes
	@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
	@param num_partial_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
	@param train_ratio: the ratio specifying the training set size in the train-validation-test split
	@param val_ratio: the ratio specifying the validation set size in the train-validation-test split
	@param seed: random generator seed
	@return: a dictionary with data loaders
	"""

	# Train-validation-test split
	indices_train, indices_valtest = train_test_split(np.arange(0, num_points), train_size=train_ratio,
													  random_state=seed)
	indices_val, indices_test = train_test_split(indices_valtest, train_size=val_ratio / (1. - train_ratio),
												 random_state=2 * seed)

	# Datasets
	synthetic_datasets = {'train': SyntheticDataset(num_vars=num_vars, num_views=num_views, num_points=num_points,
													num_predicates=num_predicates, partial_predicates=partial_predicates,
													num_partial_predicates=num_partial_predicates, indices=indices_train,
													seed=seed),
						  'val': SyntheticDataset(num_vars=num_vars, num_views=num_views, num_points=num_points,
												  num_predicates=num_predicates, partial_predicates=partial_predicates,
												  num_partial_predicates=num_partial_predicates, indices=indices_val,
												  seed=seed),
						  'test': SyntheticDataset(num_vars=num_vars, num_views=num_views, num_points=num_points,
												   num_predicates=num_predicates, partial_predicates=partial_predicates,
												   num_partial_predicates=num_partial_predicates, indices=indices_test,
												   seed=seed)}
	# Data loaders
	synthetic_loaders = {x: torch.utils.data.DataLoader(synthetic_datasets[x], batch_size=batch_size, shuffle=True,
														num_workers=num_workers) for x in ['train', 'val', 'test']}

	return synthetic_loaders


def get_synthetic_datasets(num_vars: int, num_views: int, num_points: int, num_predicates: int,
						   partial_predicates: bool = False, num_partial_predicates: int = None,
						   train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42):
	"""
	Constructs dataset objects for the synthetic data

	@param num_vars: number of covariates per view
	@param num_views: number of views
	@param num_points: number of data points
	@param num_predicates: number of concepts
	@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
	@param num_partial_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
	@param train_ratio: the ratio specifying the training set size in the train-validation-test split
	@param val_ratio: the ratio specifying the validation set size in the train-validation-test split
	@param seed: random generator seed
	@return: dataset objects for the training, validation and test sets
	"""
	# Train-validation-test split
	indices_train, indices_valtest = train_test_split(np.arange(0, num_points), train_size=train_ratio,
													  random_state=seed)
	indices_val, indices_test = train_test_split(indices_valtest, train_size=val_ratio / (1. - train_ratio),
												 random_state=2 * seed)

	# Datasets
	synthetic_datasets = {'train': SyntheticDataset(num_vars=num_vars, num_views=num_views, num_points=num_points,
													num_predicates=num_predicates, partial_predicates=partial_predicates,
													num_partial_predicates=num_partial_predicates, indices=indices_train,
													seed=seed),
						  'val': SyntheticDataset(num_vars=num_vars, num_views=num_views, num_points=num_points,
												  num_predicates=num_predicates, partial_predicates=partial_predicates,
												  num_partial_predicates=num_partial_predicates, indices=indices_val,
												  seed=seed),
						  'test': SyntheticDataset(num_vars=num_vars, num_views=num_views, num_points=num_points,
												   num_predicates=num_predicates, partial_predicates=partial_predicates,
												   num_partial_predicates=num_partial_predicates, indices=indices_test,
												   seed=seed)}

	return synthetic_datasets['train'], synthetic_datasets['val'], synthetic_datasets['test']

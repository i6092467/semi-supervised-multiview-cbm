"""
Utility functions for generating multiview animals with attributes data
"""
import numpy as np

import torch
from torch.utils import data

from torchvision import transforms
from torchvision.transforms.functional import crop, resize

from datasets.awa_dataset import (AnimalDataset, train_test_split_AwA)


class MultiViewAnimalDataset(data.dataset.Dataset):
	"""
	Dataset class for the multiview animals with attributes
	"""
	def __init__(self, n_views, width, height, classes_file, img_dir_list=None, transform=None, seed=42,
				 partial_predicates: bool = False, num_predicates: int = 85, preload: bool = False):
		"""
		Initializes a dataset object for the multiview animals with attributes

		@param n_views: number of views
		@param width: width of the crop (in pixels) to generate each view
		@param height: height of the crop (in pixels) to generate each view
		@param classes_file: the file listing all classes from the AwA dataset
		@param img_dir_list: list with the file names of images to be included (if set to None all images are included)
		@param transform: transformation applied to the images when loading
		@param seed: random generator seed
		@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
		@param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
		@param preload: flag identifying if the images should be preloaded into the CPU memory
		"""
		self.n_views = n_views

		self.awa = AnimalDataset(
			classes_file=classes_file, img_dir_list=img_dir_list, transform=transform,
			partial_predicates=partial_predicates, num_predicates=num_predicates, preload=preload, seed=seed)

		self.boxes = np.zeros((len(self.awa), self.n_views, 4))

		# Generate fixed crop boxes for the individual views
		np.random.seed(seed)
		for i in range(len(self.awa)):
			for j in range(self.n_views):
				left = np.random.randint(low=0, high=224 - width)
				top = np.random.randint(low=0, high=224 - height)
				self.boxes[i, j] = np.array([top, left, height, width])

	def __getitem__(self, index):
		"""
		Returns points from the dataset

		@param index: index
		@return: a dictionary with the data; dict['img_code'] contains indices, dict['file_names'] contains
		image file names, dict['images'] contains multiview images, dict['label'] contains target labels,
		dict['features'] contains multiview images, dict['concepts'] contains concept values.
		"""
		# Get data from the vanilla AwA
		awa_dict = self.awa.__getitem__(index=index)
		im = awa_dict['images']
		im_index = awa_dict['label']
		file_names = awa_dict['file_names']
		im_predicate = awa_dict['concepts']
		im_views = torch.zeros((self.n_views, im.shape[0], im.shape[1], im.shape[2]))

		# Create multiple views from a single image by cropping
		for j in range(self.n_views):
			tmp = crop(im, int(self.boxes[index, j][0]), int(self.boxes[index, j][1]),
					   int(self.boxes[index, j][2]), int(self.boxes[index, j][3]))
			im_views[j] = resize(tmp, [224, 224])

		return {'img_code': index, 'file_names': file_names, 'images': im_views, 'label': im_index,
				'features': im, 'concepts': im_predicate}

	def __len__(self):
		return self.boxes.shape[0]


def get_MAwA_dataloaders(n_views, width, height, classes_file, batch_size, num_workers, train_ratio=0.6, val_ratio=0.2,
						 seed=42, partial_predicates: bool = False, num_predicates: int = 85, preload: bool = False):
	"""
	Constructs data loaders for the multiview animals with attributes dataset

	@param n_views: number of views
	@param width: width of the crop (in pixels) to generate each view
	@param height: height of the crop (in pixels) to generate each view
	@param classes_file: the file listing all classes from the AwA dataset
	@param batch_size: batch size
	@param num_workers: number of worker processes
	@param train_ratio: the ratio specifying the training set size in the train-validation-test split
	@param val_ratio: the ratio specifying the validation set size in the train-validation-test split
	@param seed: random generator seed
	@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
	@param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
	@param preload: flag identifying if the images should be preloaded into the CPU memory
	@return: a dictionary with the data loaders for the training, validation and test sets
	"""
	# Train-validation-test split
	img_names_train, img_names_val, img_names_test = train_test_split_AwA(
		classes_file=classes_file, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
		partial_predicates=partial_predicates, num_predicates=num_predicates, preload=preload)

	# Transformations
	transform_list_train = []
	if not preload:
		transform_list_train.append(transforms.Resize(size=(224, 224)))
	transform_list_train.append(transforms.ToTensor())
	transform_train = transforms.Compose(transform_list_train)

	transform_list_val = []
	if not preload:
		transform_list_val.append(transforms.Resize(size=(224, 224)))
	transform_list_val.append(transforms.ToTensor())
	transform_val = transforms.Compose(transform_list_val)

	transform_list_test = []
	if not preload:
		transform_list_test.append(transforms.Resize(size=(224, 224)))
	transform_list_test.append(transforms.ToTensor())
	transform_test = transforms.Compose(transform_list_test)

	mawa_datasets = {'train': MultiViewAnimalDataset(n_views=n_views, width=width, height=height,
													 classes_file=classes_file, img_dir_list=img_names_train,
													 transform=transform_train, seed=seed,
													 partial_predicates=partial_predicates,
													 num_predicates=num_predicates, preload=preload),
					'val': MultiViewAnimalDataset(n_views=n_views, width=width, height=height,
												  classes_file=classes_file, img_dir_list=img_names_val,
												  transform=transform_val, seed=seed,
												  partial_predicates=partial_predicates,
												  num_predicates=num_predicates, preload=preload),
					'test': MultiViewAnimalDataset(n_views=n_views, width=width, height=height,
												   classes_file=classes_file, img_dir_list=img_names_test,
												   transform=transform_test, seed=seed,
												  partial_predicates=partial_predicates,
												  num_predicates=num_predicates, preload=preload)}
	# Data loaders
	mawa_loaders = {x: torch.utils.data.DataLoader(mawa_datasets[x], batch_size=batch_size, shuffle=True,
												   num_workers=num_workers) for x in ['train', 'val', 'test']}

	return mawa_loaders


def get_MAwA_datasets(n_views, width, height, classes_file, train_ratio=0.6, val_ratio=0.2, seed=42,
					  partial_predicates: bool = False, num_predicates: int = 85, preload: bool = False):
	"""
	Constructs dataset objects for the multiview animals with attributes

	@param n_views: number of views
	@param width: width of the crop (in pixels) to generate each view
	@param height: height of the crop (in pixels) to generate each view
	@param classes_file: the file listing all classes from the AwA dataset
	@param train_ratio: the ratio specifying the training set size in the train-validation-test split
	@param val_ratio: the ratio specifying the validation set size in the train-validation-test split
	@param seed: random generator seed
	@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
	@param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
	@param preload: flag identifying if the images should be preloaded into the CPU memory
	@return: dataset objects corresponding to the training, validation and test sets, respectively
	"""
	# Train-validation-test split
	img_names_train, img_names_val, img_names_test = train_test_split_AwA(
		classes_file=classes_file, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
		partial_predicates=partial_predicates, num_predicates=num_predicates, preload=preload)

	# Transformations
	transform_list_train = []
	if not preload:
		transform_list_train.append(transforms.Resize(size=(224, 224)))
	transform_list_train.append(transforms.ToTensor())
	transform_train = transforms.Compose(transform_list_train)

	transform_list_val = []
	if not preload:
		transform_list_val.append(transforms.Resize(size=(224, 224)))
	transform_list_val.append(transforms.ToTensor())
	transform_val = transforms.Compose(transform_list_val)

	transform_list_test = []
	if not preload:
		transform_list_test.append(transforms.Resize(size=(224, 224)))
	transform_list_test.append(transforms.ToTensor())
	transform_test = transforms.Compose(transform_list_test)

	mawa_datasets = {'train': MultiViewAnimalDataset(n_views=n_views, width=width, height=height,
													 classes_file=classes_file, img_dir_list=img_names_train,
													 transform=transform_train, seed=seed,
													 partial_predicates=partial_predicates,
													 num_predicates=num_predicates, preload=preload),
					 'val': MultiViewAnimalDataset(n_views=n_views, width=width, height=height,
												   classes_file=classes_file, img_dir_list=img_names_val,
												   transform=transform_val, seed=seed,
													 partial_predicates=partial_predicates,
													 num_predicates=num_predicates, preload=preload),
					 'test': MultiViewAnimalDataset(n_views=n_views, width=width, height=height,
													classes_file=classes_file, img_dir_list=img_names_test,
													transform=transform_test, seed=seed,
													 partial_predicates=partial_predicates,
													 num_predicates=num_predicates, preload=preload)}

	return mawa_datasets['train'], mawa_datasets['val'], mawa_datasets['test']

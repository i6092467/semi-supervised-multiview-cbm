"""
Animals with Attributes (AwA) 2 dataset: https://cvml.ist.ac.at/AwA2/
Code adapted from https://github.com/dfan/awa2-zero-shot-learning
"""
import numpy as np

import os
from glob import glob

import progressbar

from PIL import Image

import torch
from torch.utils import data

from torchvision import transforms

from sklearn.model_selection import train_test_split

# AwA dataset directory
# TODO: insert relevant directory here
DATA_DIR = '...'

imgs = {}


class AnimalDataset(data.dataset.Dataset):
	"""
	Animals with attributes dataset
	"""
	def __init__(self, classes_file, img_dir_list=None, transform=None, partial_predicates: bool = False,
				 num_predicates: int = 85, preload: bool = False, seed: int = 42):
		"""
		Initializes the dataset object

		@param classes_file: the file listing all classes from the AwA dataset
		@param img_dir_list: list with the file names of images to be included (if set to None all images are included)
		@param transform: transformation applied to the images when loading
		@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
		@param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
		@param preload:  flag identifying if the images should be preloaded into the CPU memory
		@param seed: random generator seed
		"""
		predicate_binary_mat = np.array(
			np.genfromtxt(os.path.join(DATA_DIR, 'predicate-matrix-binary.txt'), dtype='int'))
		self.predicate_binary_mat = predicate_binary_mat
		self.transform = transform

		# Shall a partial predicate set be used?
		if not partial_predicates:
			self.predicate_idx = np.arange(0, self.predicate_binary_mat.shape[1])
		else:
			np.random.seed(seed)
			self.predicate_idx = np.random.choice(a=np.arange(0, self.predicate_binary_mat.shape[1]),
												  size=(num_predicates, ), replace=False)

		class_to_index = dict()
		# Build dictionary of indices to classes
		with open(os.path.join(DATA_DIR, 'classes.txt')) as f:
			index = 0
			for line in f:
				class_name = line.split('\t')[1].strip()
				class_to_index[class_name] = index
				index += 1
		self.class_to_index = class_to_index

		img_names = []
		img_index = []
		with open(os.path.join(DATA_DIR, classes_file)) as f:
			for line in f:
				class_name = line.strip()
				FOLDER_DIR = os.path.join(os.path.join(DATA_DIR, 'JPEGImages'), class_name)
				file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
				files = glob(file_descriptor)

				class_index = class_to_index[class_name]
				for file_name in files:
					img_names.append(file_name)
					img_index.append(class_index)

		# If a list of images is pre-specified, use only them
		if img_dir_list is not None:
			inds = [img_names.index(x) for x in img_dir_list if x in img_names]
		else:
			inds = [_ for _ in range(len(img_names))]
		self.img_names = [img_names[i] for i in inds]
		self.img_index = [img_index[i] for i in inds]

		self.preload = preload

		# Preload images if necessary
		if preload:
			print('Pre-loading AwA images...')
			bar = progressbar.ProgressBar(maxval=len(img_names))
			bar.start()

			for i in range(len(img_names)):
				if img_names[i] in imgs:
					pass
				else:
					im = Image.open(self.img_names[i])
					if im.getbands()[0] == 'L':
						im = im.convert('RGB')
					im = im.resize((224, 224))
					imgs[img_names[i]] = np.array(im)
				bar.update(i)
			bar.finish()

	def __getitem__(self, index):
		"""
		Returns points from the dataset

		@param index: index
		@return: a dictionary with the data; dict['img_code'] contains indices, dict['file_names'] contains
		image file names, dict['images'] contains images, dict['label'] contains target labels,
		dict['features'] contains images, dict['concepts'] contains concept values.
		"""
		if not self.preload:
			im = Image.open(self.img_names[index])
			if im.getbands()[0] == 'L':
				im = im.convert('RGB')
		else:
			im = imgs[self.img_names[index]]

		if self.transform:
			im = self.transform(im)

		im_index = self.img_index[index]
		im_predicate = self.predicate_binary_mat[im_index, self.predicate_idx]

		return {'img_code': index, 'file_names': self.img_names[index], 'images': im, 'label': im_index,
				'features': im, 'concepts': im_predicate}

	def __len__(self):
		return len(self.img_names)


def train_test_split_AwA(classes_file, train_ratio=0.6, val_ratio=0.2, seed=42,
						 partial_predicates: bool = False, num_predicates: int = 85, preload: bool = False):
	"""
	Performs train-validation-test split and constructs dataset objects

	@param classes_file: the file listing all classes from the AwA dataset
	@param train_ratio: the ratio specifying the training set size in the train-validation-test split
	@param val_ratio: the ratio specifying the validation set size in the train-validation-test split
	@param seed: random generator seed
	@param partial_predicates: flag identifying whether only a subset of the ground-truth concepts will be observable
	@param num_predicates: if @partial_predicates is True, specifies the number of concepts to be observed
	@param preload: flag identifying if the images should be preloaded into the CPU memory
	@return: dataset objects corresponding to the training, validation and test sets, respectively
	"""
	assert train_ratio + val_ratio < 1.0
	np.random.seed(seed)
	awa_complete = AnimalDataset(
		classes_file=classes_file, transform=None, partial_predicates=partial_predicates, num_predicates=num_predicates,
		preload=preload, seed=seed)

	img_names_train, img_names_valtest = train_test_split(
		awa_complete.img_names, train_size=train_ratio, random_state=seed)
	img_names_val, img_names_test = train_test_split(
		img_names_valtest, train_size=val_ratio / (1. - train_ratio), random_state=2*seed)

	return img_names_train, img_names_val, img_names_test


def get_AwA_dataloaders(classes_file, batch_size, num_workers, train_ratio=0.6, val_ratio=0.2, seed=42,
						partial_predicates: bool = False, num_predicates: int = 85, preload: bool = False):
	"""
	Constructs data loaders for the AwA dataset

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

	# Datasets
	awa_datasets = {'train': AnimalDataset(classes_file=classes_file, img_dir_list=img_names_train,
										   transform=transform_train, partial_predicates=partial_predicates,
										   num_predicates=num_predicates, preload=preload, seed=seed),
					'val': AnimalDataset(classes_file=classes_file, img_dir_list=img_names_val,
										 transform=transform_val, partial_predicates=partial_predicates,
										 num_predicates=num_predicates, preload=preload, seed=seed),
					'test': AnimalDataset(classes_file=classes_file, img_dir_list=img_names_test,
										 transform=transform_test, partial_predicates=partial_predicates,
										  num_predicates=num_predicates, preload=preload, seed=seed)}
	# Data loaders
	awa_loaders = {x: torch.utils.data.DataLoader(awa_datasets[x], batch_size=batch_size, shuffle=True,
												  num_workers=num_workers) for x in ['train', 'val', 'test']}

	return awa_loaders

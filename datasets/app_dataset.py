import copy
import json
import math
import os
import random
import progressbar

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

# Dictionary mapping feature names to column indices
FEATURE_DICT = {"Age": 0, "Male_sex": 1, "Height": 2, "Weight": 3, "BMI": 4, "Alvarado": 5, "PAS": 6, "Peritonitis": 7,
				"Migration": 8, "RLQ_tender": 9, "Rebound": 10, "Cough": 11, "Psoas": 12, "Nausea": 13, "Anorexia": 14,
				"Temp": 15, "Dysuria": 16, "Stool": 17, "WBC": 18, "Neutrophils": 19, "CRP": 20, "Ketones_ur": 21,
				"Erythrocytes_ur": 22, "WBC_ur": 23, "Visibility_app": 24, "Diameter": 25, "Fluids": 26, "Layers": 27,
				"Kokarde": 28, "Perfusion": 29, "Perforation": 30, "Tissue_r": 31, "Pathological_lymph": 32,
				"Thickening": 33, "Ileus": 34, "Coprostasis": 35, "Meteorism": 36, "Enteritis": 37, "Abscess": 38,
				"Conglomerate": 39, "Gynecol": 40}

imgs = {}


class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		"""
		Transformation adding Gaussian noise to tensors

		@param mean: mean of the Gaussian noise
		@param std: standard deviation of the Gaussian noise
		"""
		self.std = std
		self.mean = mean

	def __call__(self, tensor):
		gaussian = torch.randn(tensor.size()) * self.std + self.mean
		return tensor + gaussian

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddSPNoise(object):
	def __init__(self, s_p_ratio=0.5, amount=0.02):
		"""
		Transformation adding salt-and-pepper noise to tensors

		@param s_p_ratio: ratio between the numbers of white and black pixels
		@param amount: share of noise pixels among the total number of pixels
		"""
		self.s_p_ratio = s_p_ratio
		self.amount = amount

	def __call__(self, tensor):
		shape = tensor.shape
		total = int(self.amount * tensor.nelement())
		num_salt = torch.randint(0, tensor.nelement(), (int(self.s_p_ratio * total),))
		flat = tensor.view(-1)
		flat[num_salt] = 1
		num_pepper = torch.randint(0, tensor.nelement(), (int((1 - self.s_p_ratio) * total),))
		flat[num_pepper] = 0
		return flat.view(shape)


class ChangeBrightness(object):
	def __init__(self, interval=0.5):
		"""
		Transformation changing the brightness of the image represented by a tensor

		@param interval: defines an interval for the random factor for brightness adjustment
		"""
		self.interval = interval
		assert self.interval >= 0 and self.interval <= 1

	def __call__(self, tensor):
		r = random.randint(0, 100) / 100
		factor = (1 - self.interval) + r * 2 * self.interval
		return T.functional.adjust_brightness(tensor, factor)


class RandomResize(object):

	def __init__(self, interval=(0.5, 1)):
		"""
		Transformation resizing a tensor by center-cropping

		@param interval: defines an interval for the random factor for center cropping
							(a value of 1 is equivalent to not cropping)
		"""
		self.interval = interval
		assert 0 <= self.interval[0] < self.interval[1] <= 1

	def __call__(self, tensor):
		r = random.randint(0, 100) / 100
		factor = self.interval[0] + (self.interval[1] - self.interval[0]) * r
		cropped = T.functional.center_crop(tensor, round(factor * tensor.shape[-1]))
		return T.functional.resize(cropped, tensor.shape[-1])


class RandomSharpness(object):
	def __init__(self):
		"""
		Transformation adjusting the sharpness of an image represented as a tensor
		"""
		pass

	def __call__(self, tensor):
		inc = random.randint(0, 2)
		if inc == 0:
			factor = random.randint(0, 100) / 100
		else:
			factor = random.randint(1, 8)
		return T.functional.adjust_sharpness(tensor, factor)


class RandomGamma(object):

	def __init__(self, interval=(0.5, 2)):
		"""
		Transformation adjusting the gamma value of an image represented as a tensor.

		@param interval: defines an interval for the random factor for adjusting the gamma
		"""
		self.interval = interval

	def __call__(self, tensor):
		r = random.randint(0, 100) / 100
		if r < 0.5:
			factor = self.interval[0] + r
		else:
			factor = (r - 0.5) * (self.interval[1] - 1) / 0.5 + 1
		return T.functional.adjust_gamma(tensor, factor)


class RandomZeroing(object):
	def __init__(self, frac=(0.05)):
		"""
		Transformation setting a randomly chosen rectangle with an image to zero.

		@param frac: defines the size of the rectangle as a fraction of the image size.
		"""
		self.frac = frac

	def __call__(self, tensor):
		_, h, w = tensor.shape
		t_size = h * w
		rect_size = round(t_size * self.frac)
		l = round(math.sqrt(rect_size))
		r = random.randint(0, 100) / 100
		if r < 0.5:
			lower = round(l * (0.666))
			factor = round((l - lower) * 2 * r + lower)
		else:
			upper = round(l * 1.5)
			factor = round((upper - l) * 2 * (r - 0.5) + l)
		rect_h = factor
		rect_w = rect_size // rect_h
		rect_x = random.randint(0, h - rect_h)
		rect_y = random.randint(0, w - rect_w)
		tensor[:, rect_x:rect_x + rect_h, rect_y:rect_y + rect_w] = 0
		return tensor


class AppendicitisDataset(torch.utils.data.Dataset):
	"""
	A class for the pediatric appendicitis dataset
	"""
	def __init__(self, config, augmentation=True, visualize=False, train_data=True):
		"""
		Initialize new dataset object

		@param config: stores all necessary configuration parameters
		@param augmentation: defines if augmentation should be applied to images
		@param visualize: defines if when accessing data point ultrasound images should be plotted
		@param train_data: indicates whether the dataset is used for training a model
		"""
		if train_data is False and augmentation is True:
			raise ValueError("Augmentation should not be done on test data!")
		if train_data:
			self.img_dir = config['images']
			with open(config['dict_file']) as f:
				self.labels = json.load(f)
		else:
			self.img_dir = config['test_images']
			with open(config['dict_file_test']) as f:
				self.labels = json.load(f)

		self.fusion = len(config["ex_features"]) > 0
		self.feat_idx = [FEATURE_DICT.get(key) for key in config["ex_features"]]
		self.hist_equal = config['hist_equal']
		self.augmentation = augmentation
		self.visualize = visualize
		try:
			self.concept_ids = config["concept_ids"]
		except KeyError:
			if config['dataset'] == 'app':
				self.concept_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
			else:
				self.concept_ids = [i for i in range(config['num_s_concepts'])]
		self.transforms = []
		if augmentation:
			self.gaussNoise = config['gaussian_noise']
			self.poissonNoise = config['poisson_noise']
			self.SPNoise = config['SP_noise']
			self.zero_rect = config['zero_rect']
			self.augment_per_sample = config['aug_per_sample']

			if config['normalize']:
				self.transforms.append('normalize')
			if config['brightness']:
				self.transforms.append('brightness')
			if config['rotate']:
				self.transforms.append('rotate')
			if config['shear']:
				self.transforms.append('shear')
			if config['resize']:
				self.transforms.append('resize')
			if config['sharpness']:
				self.transforms.append('sharpness')
			if config['gamma']:
				self.transforms.append('gamma')

		self.preload = config['preload']
		# Preload images if necessary
		if self.preload:
			print('Pre-loading images...')
			bar = progressbar.ProgressBar(maxval=len(self.labels))
			bar.start()

			for i in range(len(self.labels)):
				img_code = list(self.labels)[i]
				if img_code in imgs:
					pass
				else:
					# Load patient's images
					file_names_orig = list(self.labels.values())[i][0]
					label = list(self.labels.values())[i][1]
					concepts = np.array(list(self.labels.values())[i][3])[self.concept_ids]

					if self.fusion:
						all_tab_features = list(self.labels.values())[i][2]
						tab_features = [all_tab_features[i] for i in self.feat_idx]
					else:
						tab_features = []

					images = np.empty((20, 400, 400))
					for idx, name in enumerate(file_names_orig):
						img = cv2.imread(os.path.join(self.img_dir, name), cv2.IMREAD_GRAYSCALE)
						assert img is not None

						# CLAHE
						if self.hist_equal:
							clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
							img = clahe.apply(img)

						img = img / 255.

						images[idx] = img

					imgs[img_code] = images

				bar.update(i)
			bar.finish()

	def __getitem__(self, i):
		"""
		Retrieves data points form the dataset

		@param i: index
		@return: a dictionary with the data; dict['img_code'] contains image codes, dict['file_names'] contains
		image file names, dict['images'] contains ultrasound images, dict['label'] contains target labels,
		dict['features'] contains tabular features, dict['concepts'] contains concept values.
		"""
		# Load patient's images, label and if necessary, tabular features
		img_code = list(self.labels)[i]
		file_names_orig = list(self.labels.values())[i][0]
		label = list(self.labels.values())[i][1]
		concepts = np.array(list(self.labels.values())[i][3])[self.concept_ids]

		if self.fusion:
			all_tab_features = list(self.labels.values())[i][2]
			tab_features = [all_tab_features[i] for i in self.feat_idx]
		else:
			tab_features = []

		if self.preload:
			images = np.empty((20, 3, 400, 400))
			images_ = imgs[img_code]

			for idx, name in enumerate(file_names_orig):
				img = images_[idx]

				# Randomly choose a predefined number of transformations
				# Black rectangle is always applied if augmentation is enabled (and zero_rect > 0)
				if self.augmentation:
					apply_transforms = []
					augments = self.transforms.copy()
					taken_augments = []
					for i in range(self.augment_per_sample):
						assert len(augments) > 0, "Not enough distinct random transformations!"
						r = random.randint(0, len(augments) - 1)
						taken_augments.append(augments[r])
						del augments[r]

					if 'normalize' in taken_augments:
						mean = img.mean()
						std = img.std()
						apply_transforms.append(T.Normalize(mean, std))

					if 'brightness' in taken_augments:
						apply_transforms.append(ChangeBrightness(interval=0.5))

					if 'rotate' in taken_augments:
						apply_transforms.append(T.RandomRotation((-20, 20), interpolation=InterpolationMode.BILINEAR))

					if 'shear' in taken_augments:
						apply_transforms.append(T.RandomAffine(0, shear=20, interpolation=InterpolationMode.BILINEAR))

					if 'resize' in taken_augments:
						apply_transforms.append(RandomResize((0.6, 1)))

					if 'sharpness' in taken_augments:
						apply_transforms.append(RandomSharpness())

					if 'gamma' in taken_augments:
						apply_transforms.append(RandomGamma())

					if self.zero_rect > 0:
						apply_transforms.append(RandomZeroing(self.zero_rect))

					if self.gaussNoise:
						noise = AddGaussianNoise(0, 0.01)
						apply_transforms.append(noise)
					elif self.SPNoise:
						noise = AddSPNoise(0.5, 0.002)
						apply_transforms.append(noise)

					apply_transform = T.Compose(apply_transforms)

					img = torch.from_numpy(img).float().unsqueeze(0)
					img = apply_transform(img).squeeze()
				else:
					img = torch.from_numpy(img).float()

				img = torch.stack((img, img, img), 0)  						# network expects 3 channels
				norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
											 std=[0.229, 0.224, 0.225])  	# for PyTorch pretrained models
				images[idx] = norm_transform(img)

		else:
			images = np.empty((20, 3, 400, 400))
			for idx, name in enumerate(file_names_orig):
				img = cv2.imread(os.path.join(self.img_dir, name), cv2.IMREAD_GRAYSCALE)
				assert img is not None

				# CLAHE
				if self.hist_equal:
					clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
					img = clahe.apply(img)

				img = img / 255.
				if self.visualize:
					fig, ax = plt.subplots(1, 2)
					ax[0].imshow(img, cmap='gray', vmin=0, vmax=1)
					ax[0].set_title("Original")

				# Randomly choose a predefined number of transformations
				# Black rectangle is always applied if augmentation is enabled (and zero_rect > 0)
				if self.augmentation:
					apply_transforms = []
					augments = self.transforms.copy()
					taken_augments = []
					for i in range(self.augment_per_sample):
						assert len(augments) > 0, "Not enough distinct random transformations!"
						r = random.randint(0, len(augments) - 1)
						taken_augments.append(augments[r])
						del augments[r]

					if 'normalize' in taken_augments:
						mean = img.mean()
						std = img.std()
						apply_transforms.append(T.Normalize(mean, std))

					if 'brightness' in taken_augments:
						apply_transforms.append(ChangeBrightness(interval=0.5))

					if 'rotate' in taken_augments:
						apply_transforms.append(T.RandomRotation((-20, 20), interpolation=InterpolationMode.BILINEAR))

					if 'shear' in taken_augments:
						apply_transforms.append(T.RandomAffine(0, shear=20, interpolation=InterpolationMode.BILINEAR))

					if 'resize' in taken_augments:
						apply_transforms.append(RandomResize((0.6, 1)))

					if 'sharpness' in taken_augments:
						apply_transforms.append(RandomSharpness())

					if 'gamma' in taken_augments:
						apply_transforms.append(RandomGamma())

					if self.zero_rect > 0:
						apply_transforms.append(RandomZeroing(self.zero_rect))

					if self.gaussNoise:
						noise = AddGaussianNoise(0, 0.01)
						apply_transforms.append(noise)
					elif self.SPNoise:
						noise = AddSPNoise(0.5, 0.002)
						apply_transforms.append(noise)

					apply_transform = T.Compose(apply_transforms)

					img = torch.from_numpy(img).float().unsqueeze(0)
					img = apply_transform(img).squeeze()
				else:
					img = torch.from_numpy(img).float()

				img = torch.stack((img, img, img), 0)  						# network expects 3 channels
				norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
											 std=[0.229, 0.224, 0.225])  	# for PyTorch pretrained models
				images[idx] = norm_transform(img)

		padding_image = torch.zeros(img.shape)
		file_names = copy.deepcopy(file_names_orig)
		if len(file_names) < 20:
			file_names.extend(["padding.bmp"] * (20 - len(file_names_orig)))
			for i in range(len(images), 20):
				images[i] = padding_image

		assert len(images) == 20
		assert len(file_names) == 20
		return {'img_code': img_code, 'file_names': file_names, 'images': torch.tensor(images), 'label': label,
				'features': torch.FloatTensor(tab_features), 'concepts': torch.FloatTensor(concepts)}

	def __len__(self):
		return len(self.labels)

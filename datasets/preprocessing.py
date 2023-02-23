"""
Utility functions for preprocessing appendicitis data
"""
import os
import argparse
from pathlib import Path
from shutil import rmtree
import cv2
import re
import glob
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

ALLOWED_NAMES = ['app', 'rlq']
FOREGROUND_THRESHOLD = 0.2
SHADE_THRESHOLD = 6
TARGET_DIMS = (400, 400)
DEBUG = False


def find_image_border(img, threshold, shade_threshold):
	"""
	Finds a good border for cropping the image
	"""
	k = 0
	height, width = img.shape

	for row_idx, row in enumerate(img):
		foreground = len([px for px in row if px > shade_threshold])
		if foreground / height > threshold:
			if k == 0:
				row_idx_old = row_idx
			k += 1
			if k > 20:
				row_idx = row_idx_old
				return row_idx
		else:
			k = 0
	return row_idx


def filter_files(image_dir, allowed_names):
	"""
	Returns a filtered list of image files in the specified directory
	"""
	image_file_list = glob.glob(image_dir + '/*')

	filtered_file_list = []
	for file in image_file_list:
		name = file.split('/')[-1]
		if len(name.split('.')) < 3:
			print(name)
			continue
		for allowed_name in allowed_names:
			if re.search(allowed_name, name, flags=re.IGNORECASE):
				filtered_file_list.append(file)

	return image_file_list


def find_boundary(image_file, threshold, shade_threshold, no_black_triangles=False):
	"""
	Constructs boundaries for the crop
	"""
	img = cv2.imread(image_file)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	height, width = img.shape
	i_top = find_image_border(img, threshold, shade_threshold)

	flipped_img = cv2.flip(img, 0)

	i_bottom = find_image_border(flipped_img, threshold, shade_threshold)
	i_bottom = height - i_bottom

	rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

	j_left = find_image_border(rotated_img, threshold, shade_threshold)

	rotate_counter_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

	j_right = find_image_border(rotate_counter_img, threshold, shade_threshold)
	j_right = width - j_right

	if no_black_triangles:
		top_row = img[i_top]
		i = j_left
		while np.sum(top_row[i:i + 5]) < 10:
			i += 1
		j_left = i
		i = j_right
		while np.sum(top_row[i - 5:i]) < 10:
			i -= 1
		j_right = i

	if DEBUG:
		cv2.rectangle(img, (j_left, i_top), (j_right, i_bottom), (255), 1)
		fig = plt.figure()
		fig.set_size_inches(10, 10)
		plt.title(image_file)
		plt.imshow(img, cmap='gray')
		plt.show()

	return (i_top, i_bottom, j_left, j_right)


def crop_image(image_file, boundaries, output_size, resize=False, save_dir='', padding='constant', debug=False):
	"""
	Crops and pads/resizes the image

	@param image_file: image file directory
	@param boundaries: boundary coordinates for the crop
	@param output_size:	size of the output image
	@param resize:
	@param save_dir: directory where to save the output image
	@param padding: type of padding to apply ('constant', 'reflect', 'speckle' or 'resize')
	@param debug:
	@return: a preprocessed output image
	"""
	img = cv2.imread(image_file)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	i_top = boundaries[0]
	i_bottom = boundaries[1]
	j_left = boundaries[2]
	j_right = boundaries[3]

	cropped = np.copy(img[i_top:i_bottom, j_left:j_right])

	# Either resize the cropped image to the output size by interpolation
	# or pad the output image by black pixels
	if padding == 'resize':
		output = cv2.resize(cropped, output_size)
	elif padding in ['constant', 'reflect']:
		# output = np.zeros(output_size, dtype=np.ushort)
		cropped_h, cropped_w = cropped.shape
		if output_size[0] > cropped_h:
			top = (output_size[0] - cropped_h) // 2 + 1
			bottom = output_size[0] - top - cropped_h
		else:
			diff = cropped_h - output_size[0]
			i_h_top = diff // 2
			i_h_bottom = i_h_top + output_size[0]
			cropped = cropped[i_h_top:i_h_bottom, :]
			top = 0
			bottom = 0

		if output_size[1] > cropped_w:
			left = (output_size[1] - cropped_w) // 2 + 1
			right = output_size[1] - left - cropped_w
		else:
			diff = cropped_w - output_size[1]
			i_w_left = diff // 2
			i_w_right = i_w_left + output_size[1]
			cropped = cropped[:, i_w_left:i_w_right]
			left = 0
			right = 0
		output = np.pad(cropped, ((top, bottom), (left, right)), mode=padding)

	elif padding == 'speckle':
		noise = np.random.normal(1, 0.1, size=output_size)
		output = np.ones(output_size, dtype=np.ushort) * 64
		output = np.multiply(output, noise)
		cropped_h, cropped_w = cropped.shape
		if output_size[0] > cropped_h:
			top = (output_size[0] - cropped_h) // 2 + 1
			bottom = top + cropped_h
			i_top = 0
			i_bottom = cropped_h
		else:
			diff = cropped_h - output_size[0]
			i_top = diff // 2
			i_bottom = i_top + output_size[0]
			top = 0
			bottom = output_size[0]
		if output_size[1] > cropped_w:
			left = (output_size[1] - cropped_w) // 2 + 1
			right = left + cropped_w
			i_left = 0
			i_right = cropped_w
		else:
			diff = cropped_w - output_size[1]
			i_left = diff // 2
			i_right = i_left + output_size[1]
			left = 0
			right = output_size[1]
		output[top:bottom, left:right] = cropped[i_top:i_bottom, i_left:i_right]

	else:
		print("Padding mode unknown")
		quit()

	if DEBUG:
		fig = plt.figure()
		fig.set_size_inches(10, 10)
		plt.title('Cropped')
		plt.imshow(output, cmap='gray')
		plt.show()

	if save_dir != '':
		name = image_file.split('/')[-1]
		save_dir = Path(save_dir, name)
		cv2.imwrite(str(save_dir), output)

	return output


def preprocess(config):
	"""
	Runs preprocessing on ultrasound images

	@param config: configuration file with the parameters for preprocessing subroutines
	@return: None
	"""
	DEBUG = config['debug']

	image_file_list = filter_files(config['images'], ALLOWED_NAMES)
	target_dir = Path(config['target'])
	if target_dir.exists() and target_dir.is_dir():
		rmtree(target_dir)
	try:
		os.makedirs(target_dir)
		print('Created target directory')
	except OSError:
		pass

	image_file_list.sort()

	for image_file in tqdm(image_file_list):
		boundaries = find_boundary(image_file, FOREGROUND_THRESHOLD, SHADE_THRESHOLD,
								   no_black_triangles=config['no_black_triangles'])
		cropped_image = crop_image(image_file, boundaries, TARGET_DIMS, save_dir=target_dir, padding=config['padding'])


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--images')
	parser.add_argument('-t', '--target')
	parser.add_argument('-p', '--padding')
	parser.add_argument('-d', '--debug', action='store_true')
	parser.add_argument('-nbt', '--no_black_triangles')
	args = parser.parse_args()
	argsdict = vars(args)

	preprocess(argsdict)


if __name__ == "__main__":
	main()

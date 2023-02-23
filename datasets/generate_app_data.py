"""
Utility functions for handling and preprocessing pediatric appendicitis tabular data
"""
import argparse
import glob
import json
import re
import numpy as np

import pandas as pd
import yaml
import os

# List of the feature names to retrieve from the table
TABULAR_FEATURES = ['Age', 'Sex', 'Height', 'Weight', 'BMI', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score',
					'Peritonitis', 'Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness',
					'Coughing_Pain', 'Psoas_Sign', 'Nausea', 'Loss_of_Appetite', 'Body_Temperature', 'Dysuria',
					'Stool', 'WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Ketones_in_Urine', 'RBC_in_Urine',
					'WBC_in_Urine', 'Appendix_on_US', 'Appendix_Diameter', 'Free_Fluids', 'Appendix_Wall_Layers',
					'Target_Sign', 'Perfusion', 'Perforation', 'Surrounding_Tissue_Reaction',
					'Pathological_Lymph_Nodes', 'Bowel_Wall_Thickening', 'Ileus', 'Coprostasis', 'Meteorism',
					'Enteritis', 'Appendicular_Abscess', 'Conglomerate_of_Bowel_Loops', 'Gynecological_Findings']


def update_feature_values(features):
	"""
	Dichotomise features and replace missing ultrasonographic findings with negative results.

	@param features: pandas data frame with the pediatric appendicitis data
	@return: pandas data frame with dichotomised and partially imputed feature values
	"""
	features['Sex'] = features['Sex'].replace(['male', 'female', 'no'], [1, 0, np.nan])
	features['Peritonitis'] = features['Peritonitis'].replace(['generalized', 'local', 'no'], [1, 1, 0])
	features['Migratory_Pain'] = features['Migratory_Pain'].replace(['yes', 'no'], [1, 0])
	features['Lower_Right_Abd_Pain'] = features['Lower_Right_Abd_Pain'].replace(['yes', 'no'], [1, 0])
	features['Contralateral_Rebound_Tenderness'] = features['Contralateral_Rebound_Tenderness'].replace(
		['yes', 'no'], [1, 0])
	features['Coughing_Pain'] = features['Coughing_Pain'].replace(['yes', 'no'], [1, 0])
	features['Psoas_Sign'] = features['Psoas_Sign'].replace(['yes', 'no'], [1, 0])
	features['Nausea'] = features['Nausea'].replace(['yes', 'no'], [1, 0])
	features['Loss_of_Appetite'] = features['Loss_of_Appetite'].replace(['yes', 'no'], [1, 0])
	features['Dysuria'] = features['Dysuria'].replace(['yes', 'no'], [1, 0])
	features['Stool'] = features['Stool'].replace(
		['diarrhea', 'normal', 'constipation', 'constipation, diarrhea', 'no'],
		[1, 0, 1, 1, 0])
	features['Ketones_in_Urine'] = features['Ketones_in_Urine'].replace(['+', '++', '+++', 'no'], [1, 1, 1, 0])
	features['RBC_in_Urine'] = features['RBC_in_Urine'].replace(['+', '++', '+++', 'no'], [1, 1, 1, 0])
	features['WBC_in_Urine'] = features['WBC_in_Urine'].replace(['+', '++', '+++', 'no'], [1, 1, 1, 0])
	features['Appendix_on_US'] = features['Appendix_on_US'].replace(['yes', 'no'], [1, 0])
	features['Free_Fluids'] = features['Free_Fluids'].replace(['yes', 'no'], [1, 0])
	features['Appendix_Wall_Layers'] = features['Appendix_Wall_Layers'].replace(
		['raised', 'partially raised', 'upset', 'intact', 'no'], [1, 1, 1, 0, 0])
	features['Target_Sign'] = features['Target_Sign'].replace(['yes', 'no'], [1, 0])
	features['Perfusion'] = features['Perfusion'].replace(['hyperperfused', 'hypoperfused', 'no', 'present'],
														  [1, 0, 0, 1])
	features['Perforation'] = features['Perforation'].replace(
		['no', 'yes', 'not excluded', 'suspected'], [0, 1, np.nan, 1])
	features['Surrounding_Tissue_Reaction'] = features['Surrounding_Tissue_Reaction'].replace(['yes', 'no'], [1, 0])
	features['Pathological_Lymph_Nodes'] = features['Pathological_Lymph_Nodes'].replace(['yes', 'no'], [1, 0])
	features['Bowel_Wall_Thickening'] = features['Bowel_Wall_Thickening'].replace(['yes', 'no'], [1, 0])
	features['Ileus'] = features['Ileus'].replace(['yes', 'no'], [1, 0])
	features['Coprostasis'] = features['Coprostasis'].replace(['yes', 'no'], [1, 0])
	features['Meteorism'] = features['Meteorism'].replace(['yes', 'no'], [1, 0])
	features['Enteritis'] = features['Enteritis'].replace(['yes', 'no'], [1, 0])
	features['Appendicular_Abscess'] = features['Appendicular_Abscess'].replace(['yes', 'no', 'suspected'], [1, 0, 1])
	features['Conglomerate_of_Bowel_Loops'] = features['Conglomerate_of_Bowel_Loops'].replace(['yes', 'no'], [1, 0])
	features['Gynecological_Findings'] = \
		features['Gynecological_Findings'].replace(
		['Ausschluss gyn. Ursache der Beschwerden', 'Ausschluss Ovarialtorsion', 'Ausschluss Ovarialzyste',
		 'Ausschluss pathologischer Ovarialbefund',
		 'In beiden Ovarien Zysten darstellbar, links Ovar mit regelrechter Perfusion, rechts etwas vergrößert, keine eindeutige Perfusion nachweisbar. Retrovesikal freie Flüssigkeit mit Binnenecho',
		 'ja', 'kein Anhalt für eine gynäkologische Ursache der Beschwerden',
		 'kein sicherer Ausschluss einer Ovarialtorsion, Appendizitis wahrscheinlicher', 'keine',
		 'kleine Ovarzyste rechts', 'Ovar links vergrößert', 'Ovarialzyste', 'Ovarialzyste ', 'Ovarialzyste re.',
		 'Ovarialzysten', 'unauffällig', 'V. a. Ovarialtorsion', 'Zyste Uterus', 'no'],
		[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, np.nan, 1, 1, 0])

	# For the US variables update NaN with negative findings
	features['Surrounding_Tissue_Reaction'] = features['Surrounding_Tissue_Reaction'].replace(np.nan, 0)
	features['Free_Fluids'] = features['Free_Fluids'].replace(np.nan, 0)
	features['Bowel_Wall_Thickening'] = features['Bowel_Wall_Thickening'].replace(np.nan, 0)
	features['Enteritis'] = features['Enteritis'].replace(np.nan, 0)
	features['Appendix_on_US'] = features['Appendix_on_US'].replace(np.nan, 0)
	features['Pathological_Lymph_Nodes'] = features['Pathological_Lymph_Nodes'].replace(np.nan, 0)
	features['Ileus'] = features['Ileus'].replace(np.nan, 0)
	features['Coprostasis'] = features['Coprostasis'].replace(np.nan, 0)
	features['Meteorism'] = features['Meteorism'].replace(np.nan, 0)
	features['Appendix_Wall_Layers'] = features['Appendix_Wall_Layers'].replace(np.nan, 0)
	features['Target_Sign'] = features['Target_Sign'].replace(np.nan, 0)
	features['Perforation'] = features['Perforation'].replace(np.nan, 0)
	features['Appendicular_Abscess'] = features['Appendicular_Abscess'].replace(np.nan, 0)
	features['Conglomerate_of_Bowel_Loops'] = features['Conglomerate_of_Bowel_Loops'].replace(np.nan, 0)
	features['Gynecological_Findings'] = features['Gynecological_Findings'].replace(np.nan, 0)

	return features


def load_files(config):
	"""
	Loads tabular pediatric appendicitis data and a list of ultrasound image names to be discarded

	@param config: configuration
	@return: pandas data frame with tabular features and a list of excluded images
	"""
	# Read info file
	if config['info_file'][-3:] == 'csv':
		patient_info = pd.read_csv(config['info_file'], header=[0])
	else:
		patient_info = pd.read_excel(config['info_file'], header=[0], engine="openpyxl")

	# Read blacklist
	blacklist = []
	with open(config['blacklist'], 'r') as f:
		lines = f.readlines()
		for line in lines:
			blacklist.append(line.rstrip())

	return patient_info, blacklist


def build_img_dict(config, blacklist=None):
	"""
	Builds a dictionary of key-value pairs of the form
	img_code: [img_code_img_1, img_code_img_2, ..., img_code_img_k] from the images in the given directory.

	@param config: configuration
	@param blacklist: list containing excluded images
	@return: image file dictionary
	"""
	'''
		Builds a dictionary of key-value pairs of the form
		img_code: [img_code_img_1, img_code_img_2, ..., img_code_img_k] from the images in the given directory.
		Blacklist can be used to exclude images.
	'''
	img_codes = {}
	image_file_list = glob.glob(config['image_dir'] + '/*')
	for image_file in image_file_list:
		name = image_file.split('/')[-1]
		if blacklist is not None:
			if name in blacklist:
				continue
		number = re.split('_| |\.', name)[0]
		if number not in img_codes.keys():
			img_codes[number] = [name]
		else:
			img_codes[number].append(name)
	return img_codes


def generate_files(config):
	"""
	Constructs data dictionaries and saves them as .json files

	@param config: configuration
	@return: None
	"""
	mapping = {}
	patient_info, blacklist = load_files(config)
	img_codes = build_img_dict(config, blacklist)
	images_total = 0
	excluded = 0
	patient_info.fillna(np.nan, inplace=True)
	patient_info.replace('NULL', np.nan)
	for img_code, images in img_codes.items():
		patient_code = patient_info['US_Number'].values[
			list(np.argwhere(patient_info['US_Number'].values == int(img_code)))]

		if len(patient_code) == 0:  # in case image exists but mapping to patient isn't recorded
			continue

		patient_labels = patient_info[patient_info['US_Number'].values == patient_code[0]][
			config['label']].to_numpy(dtype=str)  # target labels

		group_left = np.setdiff1d(np.arange(len(patient_labels)), [])
		if len(group_left) == 0:
			continue  # all excluded
		elif len(group_left) > 1:
			raise ValueError(f"Duplicate patient (code {patient_code[0]})!")
		else:
			idx_keep = group_left[0]
			patient_label = patient_labels[idx_keep]

		if len(patient_label) == 0:  # empty target label cell
			continue
		if patient_label in config['true_label']:
			label_num = 1
		elif patient_label in config['false_label']:
			label_num = 0
		else:
			print(f"--- Label {patient_label} is not recognized ---")
			continue

		features = patient_info[patient_info['US_Number'].values == patient_code[0]][TABULAR_FEATURES].iloc[
			[idx_keep]]
		features = update_feature_values(features)
		concepts = features[config['concepts']]
		images_total += len(images)
		mapping[img_code] = [
			images, label_num, features.to_numpy(dtype=np.float32).flatten().tolist(),
			concepts.to_numpy(dtype=np.float32).flatten().tolist()]

	# Print statistics
	print("\n---Statistics---")
	print("\nBlacklist includes ", len(blacklist), " images")
	print("\nNumber of excluded patients = ", excluded)
	print("Number of patients in info file: ", len(patient_info))
	print("-> Number of patients used = ", len(mapping))
	print("-> Number of images used = ", images_total)

	if not os.path.exists(config['output_dir']):
		os.makedirs(config['output_dir'])

	with open(os.path.join(config['output_dir'], config['output_file']), 'w') as f:
		json.dump(mapping, f)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config')
	args = parser.parse_args()
	argsdict = vars(args)

	with open(argsdict['config'], 'r') as f:
		config = yaml.safe_load(f)

	config['filename'] = argsdict['config']

	generate_files(config)


if __name__ == "__main__":
	main()

"""
Run this file, giving a configuration file as input, to train models, e.g.:
	python train.py --config configfile.yaml
"""

import argparse
import os
import random
import sys
from collections import Counter
from os.path import join
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.app_dataset import AppendicitisDataset
from datasets.mawa_dataset import MultiViewAnimalDataset, get_MAwA_datasets
from datasets.synthetic_dataset import get_synthetic_datasets

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from utils.logging import save_data_split
from utils.printing import print_epoch_val_scores, print_epoch_val_scores_
from utils.metrics import PRC, ROC

from loss import create_loss, calc_concept_weights, calc_concept_sample_weights
from networks import create_model
from validate import validate_epoch_mvcbm, validate_epoch_ssmvcbm

import progressbar


def create_optimizer(config, model, mode):
	"""
	Parse the configuration file and return a relevant optimizer object
	"""
	assert config["optimizer"] in ["sgd", "adam"], "Only SGD and Adam optimizers are available!"

	optim_params = [
		{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': config[mode + "_learning_rate"],
		 'weight_decay': config['weight_decay']}
	]

	if config["optimizer"] == "sgd":
		return torch.optim.SGD(optim_params)
	elif config["optimizer"] == "adam":
		return torch.optim.Adam(optim_params)


def _create_data_loaders(config, gen, trainset, train_ids, validset=None, val_ids=None):
	"""
	Construct dataloaders based on the given datasets and config arguments
	"""
	train_subsampler = SubsetRandomSampler(train_ids, gen)
	if val_ids is not None:
		val_subsampler = SubsetRandomSampler(val_ids, gen)

	pm = config["device"] == "cuda"
	train_loader = DataLoader(trainset, batch_size=config["train_batch_size"], sampler=train_subsampler,
							  num_workers=config["workers"], pin_memory=pm, generator=gen, drop_last=True)
	if validset is not None and val_ids is not None:
		val_loader = DataLoader(validset, batch_size=config["val_batch_size"], sampler=val_subsampler,
								num_workers=config["workers"], pin_memory=pm, generator=gen)
	else:
		val_loader = None
	return train_loader, val_loader


def set_bn_to_eval(m):
	if isinstance(m, nn.BatchNorm2d):
		m.eval()


def freeze_module(m):
	m.eval()
	for param in m.parameters():
		param.requires_grad = False


def unfreeze_module(m):
	m.train()
	for param in m.parameters():
		param.requires_grad = True


def _get_data(config):
	"""
	Parse the configuration file and return a relevant dataset
	"""
	if config['dataset'] == 'mawa':
		if config['model'] == 'MVCBM' or config['model'] == 'CBM':
			num_concepts = config['num_concepts']
		elif config['model'] == 'SSMVCBM':
			num_concepts = config['num_s_concepts']

		trainset, validset, testset = get_MAwA_datasets(
			n_views=config['num_views'], width=60, height=60, classes_file='all_classes.txt', train_ratio=0.6,
			val_ratio=0.2, seed=config['seed'], partial_predicates=config['partial_concepts'],
			num_predicates=num_concepts, preload=config['preload'])

	elif config['dataset'] == 'app':
		trainset = AppendicitisDataset(config, augmentation=config["augmentation"], visualize=False, train_data=True)
		testset = AppendicitisDataset(config, augmentation=False, visualize=False, train_data=False)
		validset = AppendicitisDataset(config, augmentation=False, visualize=False, train_data=True)

	elif config['dataset'] == 'synthetic':
		if config['model'] == 'MVCBM' or config['model'] == 'CBM':
			num_concepts = config['num_synthetic_concepts']
			num_partial_concepts = config['num_concepts']
		elif config['model'] == 'SSMVCBM':
			num_concepts = config['num_synthetic_concepts']
			num_partial_concepts = config['num_s_concepts']

		trainset, validset, testset = get_synthetic_datasets(
			num_vars=config['num_vars'], num_views=config['num_views'], num_points=config['num_points'],
			num_predicates=num_concepts, partial_predicates=config['partial_concepts'],
			num_partial_predicates=num_partial_concepts, train_ratio=0.6, val_ratio=0.2, seed=config['seed'])

	else:
		NotImplementedError('ERROR: Dataset not supported!')

	return trainset, validset, testset


def _train_one_epoch_mvcbm(mode, epoch, config, model, optimizer, loss_fn, train_loader, target_class_weights,
						   concepts_class_weights):
	"""
	Train an MVCBM for one epoch
	"""
	running_len = 0
	running_target_loss = 0
	running_concepts_loss = [0] * config['num_concepts']
	running_summed_concepts_loss = 0
	running_total_loss = 0
	# Training mode and the number of epochs
	if mode == "j":
		num_epochs = config["j_epochs"]
	elif mode == "c":
		num_epochs = config["c_epochs"]
	elif mode == "t":
		num_epochs = config["t_epochs"]
	else:
		raise ValueError("Training mode unknown!")

	# Decrease the learning rate, if applicable
	if epoch >= 0 and config["decrease_every"] > 0 and (epoch + 1) % config["decrease_every"] == 0:
		for g in optimizer.param_groups:
			g["lr"] = g["lr"] / config["lr_divisor"]

	with tqdm(total=len(train_loader) * config["train_batch_size"], desc=f"Epoch {epoch + 1}/{num_epochs}",
			  unit="img", position=0, leave=True) as pbar:
		model.train()
		if config["model"] == "MVCBM" and config["training_mode"] == "sequential" and mode == "t":
			model.apply(set_bn_to_eval)

		for k, batch in enumerate(train_loader):
			# Address the class imbalance for target variable and categorical concepts
			labels_temp = batch["label"].numpy()
			concepts_temp = batch["concepts"].numpy()
			target_sample_weights = [target_class_weights[int(labels_temp[i])] for i in range(len(labels_temp))]
			target_sample_weights = torch.FloatTensor(target_sample_weights).to(config["device"])
			concepts_sample_weights = calc_concept_sample_weights(config, concepts_class_weights, concepts_temp)
			# NOTE: class weights have to be specified per class and per sample
			loss_fn.target_sample_weight = target_sample_weights
			loss_fn.target_class_weight = torch.FloatTensor(target_class_weights).to(config["device"])
			loss_fn.c_weights = concepts_sample_weights

			batch_images, target_true, batch_names = batch["images"].to(config["device"]), batch["label"].float().to(
				config["device"]), batch["file_names"]  # put the data on the device
			concepts_true = batch["concepts"].to(config["device"])
			additional_features = batch["features"].to(config["device"])
			if config['dataset'] == 'app':
				batch_names = np.array(list(map(list, zip(*batch_names))), dtype=object)  # transpose list

			# Mask tensor indicating which views were added for padding
			if config['dataset'] == 'app':
				mask = torch.tensor(batch_names != "padding.bmp").to(config["device"])
			elif config['dataset'] == 'mawa':
				mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])
			elif config['dataset'] == 'synthetic':
				mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])

			# Forward pass
			concepts_pred, target_pred_probs, target_pred_logits, attn_weights = model(
				batch_images, mask, additional_features)
			target_pred_probs = target_pred_probs.squeeze(1)
			target_pred_logits = target_pred_logits.squeeze(1)

			# Backward pass depends on the training mode of the model
			optimizer.zero_grad()
			# Compute the loss
			target_loss, concepts_loss, summed_concepts_loss, total_loss = loss_fn(
				concepts_pred, concepts_true, target_pred_probs, target_pred_logits, target_true)

			running_target_loss += target_loss.item() * batch_images.size(0)
			for concept_idx in range(len(concepts_loss)):
				running_concepts_loss[concept_idx] += concepts_loss[concept_idx].item() * batch_images.size(0)
			running_summed_concepts_loss += summed_concepts_loss.item() * batch_images.size(0)
			running_total_loss += total_loss.item() * batch_images.size(0)

			running_len += batch_images.size(0)
			if mode == "j":
				total_loss.backward()
			elif mode == "c":
				summed_concepts_loss.backward()
			else:
				target_loss.backward()
			optimizer.step()  # perform an update

			# Update the progress bar
			pbar.set_postfix(**{"Target loss": running_target_loss / running_len,
								"Concepts loss": running_summed_concepts_loss / running_len,
								"Total loss": running_total_loss / running_len, "lr": optimizer.param_groups[0]["lr"]})

			pbar.update(config["train_batch_size"])

	return None


def _train_one_epoch_ssmvcbm(mode, epoch, config, model, optimizer, loss_fn, train_loader, target_class_weights,
							concepts_class_weights, beta, gamma, adv_it=None):
	"""
	Train an SSMVCBM for on epoch
	"""
	running_len = 0
	running_loss = 0
	running_s_concepts_loss = [0] * config["num_s_concepts"] if mode == "sc" else None
	try:  # if sample size for computing correlation matrix calculation isn't specified, batch size will be used
		corr_sample_size = config["corr_sample_size"]
	except KeyError:
		corr_sample_size = config["train_batch_size"]

	if mode == "sc":
		num_epochs = config["sc_epochs"]
	elif mode == "usc":
		num_epochs = config["usc_epochs"]
	elif mode == "d":
		num_epochs = config["d_epochs"]
	elif mode == "t":
		num_epochs = config["t_epochs"]
	else:
		raise ValueError("Training mode unknown!")

	# Decrease the learning rate, if applicable
	if epoch >= 0 and config["decrease_every"] > 0 and (epoch + 1) % config["decrease_every"] == 0:
		for g in optimizer.param_groups:
			g["lr"] = g["lgr"] / config["lr_divisor"]

	# Keep track of the past concept values
	past_us_concepts = torch.zeros(corr_sample_size - config["train_batch_size"], config["num_us_concepts"]).to(
		config["device"]) if corr_sample_size > config["train_batch_size"] else None

	with tqdm(total=len(train_loader) * config["train_batch_size"], desc=f"Epoch {epoch + 1}/{num_epochs}",
			  unit="img", position=0, leave=True) as pbar:
		model.train()
		for name, child in model.named_children():
			if mode != "sc" and name == "sc_model":
				child.apply(set_bn_to_eval)
			if mode != "usc" and name == "usc_model":
				child.apply(set_bn_to_eval)

		for k, batch in enumerate(train_loader):
			# Address the class imbalance for target variable and categorical concepts
			labels_temp = batch["label"].numpy()
			concepts_temp = batch["concepts"].numpy()
			target_sample_weights = [target_class_weights[int(labels_temp[i])] for i in range(len(labels_temp))]
			target_sample_weights = torch.FloatTensor(target_sample_weights).to(config["device"])
			concepts_sample_weights = calc_concept_sample_weights(config, concepts_class_weights, concepts_temp)
			# NOTE: class weights have to be specified per class and per sample
			loss_fn.target_sample_weight = target_sample_weights
			loss_fn.target_class_weight = torch.FloatTensor(target_class_weights).to(config["device"])
			loss_fn.c_weights = concepts_sample_weights

			batch_images, target_true, batch_names = batch["images"].to(config["device"]), batch["label"].float().to(
				config["device"]), batch["file_names"]  # put the data on the specified device
			concepts_true = batch["concepts"].to(config["device"])
			additional_features = batch["features"].to(config["device"])

			if config['dataset'] == 'app':
				batch_names = np.array(list(map(list, zip(*batch_names))), dtype=object)  	# transpose list

			# Mask tensor indicating which views were added for padding
			if config['dataset'] == 'app':
				mask = torch.tensor(batch_names != "padding.bmp").to(config["device"])
			elif config['dataset'] == 'mawa':
				mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])
			elif config['dataset'] == 'synthetic':
				mask = torch.ones((batch_images.shape[0], config['num_views'])).to(config['device'])

			# Forward pass
			s_concepts_pred, us_concepts_pred, s_attn_weights, us_attn_weights, discr_concepts_pred, \
			target_pred_probs, target_pred_logits = model(batch_images, mask, additional_features)
			target_pred_probs = target_pred_probs.squeeze(1)
			target_pred_logits = target_pred_logits.squeeze(1)

			if (k + 1) * config["train_batch_size"] < corr_sample_size:
				if k == 0:
					us_concepts_sample = us_concepts_pred
				else:
					us_concepts_sample = torch.cat(
						(past_us_concepts[-k * config["train_batch_size"]:, :].detach(), us_concepts_pred), dim=0)
			else:
				us_concepts_sample = torch.cat((past_us_concepts.detach(), us_concepts_pred),
											   dim=0) if past_us_concepts is not None else us_concepts_pred

			if past_us_concepts is not None:
				past_us_concepts = past_us_concepts.roll(shifts=-config["train_batch_size"], dims=0)
				past_us_concepts[-config["train_batch_size"]:, :] = us_concepts_pred

			# Backward pass depends on the training mode of the model
			optimizer.zero_grad()
			# Compute the loss
			target_loss, s_concepts_loss, summed_s_concepts_loss, summed_discr_concepts_loss, summed_gen_concepts_loss, us_corr_loss = \
				loss_fn(s_concepts_pred, discr_concepts_pred, concepts_true, target_pred_probs, target_pred_logits,
						target_true, us_concepts_sample)

			running_len += batch_images.size(0)
			if mode == "t":
				running_loss += target_loss.item() * batch_images.size(0)
				target_loss.backward()
			elif mode == "sc":
				for concept_idx in range(len(s_concepts_loss)):
					running_s_concepts_loss[concept_idx] += s_concepts_loss[concept_idx].item() * batch_images.size(0)
				running_loss += summed_s_concepts_loss.item() * batch_images.size(0)
				summed_s_concepts_loss.backward()
			elif mode == "usc":
				running_loss += target_loss.item() * batch_images.size(
					0) + beta * summed_gen_concepts_loss.item() * batch_images.size(0)
				total_loss = target_loss + beta * summed_gen_concepts_loss + gamma * us_corr_loss
				total_loss.backward()
			else:
				running_loss += summed_discr_concepts_loss.item() * batch_images.size(0)
				if config['adversary']:
					summed_discr_concepts_loss.backward()

			optimizer.step()  # perform an update

			# Update the progress bar
			pbar.set_postfix(**{f"{mode} loss": running_loss / running_len, "lr": optimizer.param_groups[0]["lr"]})
			pbar.update(config["train_batch_size"])

	return None


def train_mvcbm_kfold(config, gen):
	"""
	Run the k-fold cross-validation for the MVCBM
	"""
	# Log the print-outs
	old_stdout = sys.stdout
	log_file = open(os.path.join(
		config["log_directory"], config['run_name'] + '_' + config['experiment_name'] + '_' +
								 str(config['seed']) + '.log'), 'w')
	sys.stdout = log_file

	# ---------------------------------
	#       Prepare data
	# ---------------------------------
	kfold = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])

	trainset, validset, testset = _get_data(config=config)

	labels = []
	for x in iter(DataLoader(trainset)):
		labels.append(x["label"].item())
	labels = np.array(labels)

	# ---------------------------------
	# Create temporary directories for models and data splits
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	splits_dir = os.path.join(config['log_directory'], 'splits')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(splits_dir):
		os.makedirs(splits_dir)

	# Numbers of training epochs for different modules
	if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
			config["training_mode"] == "joint":
		c_epochs = config["j_epochs"]
		t_epochs = config["j_epochs"]
	elif (config["model"] == "MVCBM" or config["model"] == "CBM") and \
			config["training_mode"] == "sequential":
		c_epochs = config["c_epochs"]
		t_epochs = config["t_epochs"]
	else:
		c_epochs = None
		t_epochs = config["t_epochs"]

	c_results = np.empty((config["k_folds"], c_epochs, config["num_concepts"], (4 + 1))) if c_epochs is not None else None
	t_results = np.empty((config["k_folds"], t_epochs, (11 + 3)))  # 11 metrics + 3 loss functions
	roc = ROC(config["k_folds"], range(1, t_epochs + 1))
	pr = PRC(config["k_folds"], range(1, t_epochs + 1))

	# Iterate over all folds
	for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset, labels)):
		print()
		print("------------------------")
		print(f'FOLD {fold + 1}/{config["k_folds"]}')
		print("------------------------")
		# Instantiate dataloaders
		train_loader, val_loader = _create_data_loaders(config, gen, trainset, train_ids=train_ids,
														validset=validset, val_ids=val_ids)
		test_loader = DataLoader(
			testset, batch_size=config["val_batch_size"], num_workers=config["workers"], generator=gen)
		save_data_split(splits_dir, trainset, train_ids, validset, val_ids)

		# Retrieve labels
		train_labels = []
		val_labels = []
		all_c = []
		for concept_idx in range(config["num_concepts"]):
			c = []
			for x in iter(train_loader):
				if concept_idx == 0:
					train_labels.extend(x["label"].tolist())
				concepts = x["concepts"].numpy()
				c.extend(concepts[:, concept_idx])
			all_c.append(c)

		print("Length of training array", len(train_labels))
		print("Train target class distribution: ", Counter(train_labels))
		print("Train concepts class distribution: ")
		for concept_idx in range(len(all_c)):
			print("...", Counter(all_c[concept_idx]))

		val_all_c = []
		for concept_idx in range(config["num_concepts"]):
			c = []
			for x in iter(val_loader):
				if concept_idx == 0:
					val_labels.extend(x["label"].tolist())
				concepts = x["concepts"].numpy()
				c.extend(concepts[:, concept_idx])
			val_all_c.append(c)
		print("\nLength of validation array", len(val_labels))
		print("Validation target class distribution: ", Counter(val_labels))
		print("Validation concepts class distribution: ")
		for concept_idx in range(len(val_all_c)):
			print("...", Counter(val_all_c[concept_idx]))

		target_class_weights = compute_class_weight(
			class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
		concepts_class_weights = calc_concept_weights(all_c)

		# Initialize the model and training objects
		model = create_model(config)
		model.to(config["device"])
		loss_fn = create_loss(config)

		# Training the concept prediction model
		if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
				config["training_mode"] == "sequential":
			print("\nStarting concepts training!\n")
			mode = "c"

			for name, child in model.named_children():
				if name.split("_")[0] == "t":
					child.apply(freeze_module)

			c_optimizer = create_optimizer(config, model, mode)

			for epoch in range(0, c_epochs):

				_train_one_epoch_mvcbm(mode, epoch, config, model, c_optimizer, loss_fn, train_loader,
									   target_class_weights, concepts_class_weights)
				target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, _, _, _ = \
					validate_epoch_mvcbm(epoch, config, model, val_loader, loss_fn)

				for concept_idx in range(len(concepts_loss)):
					c_results[fold, epoch, concept_idx, 0] = concepts_loss[concept_idx]
					c_results[fold, epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

				print_epoch_val_scores(
					config, mode, target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics,
					all_cMetrics, print_all_c=(config['dataset'] == 'app'))

			# Prepare parameters for the target model training
			for name, child in model.named_children():
				if name.split("_")[0] == "t":
					child.apply(unfreeze_module)
				else:
					child.apply(freeze_module)

		# Sequential vs. joint optimisation
		if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
				config["training_mode"] == "sequential":
			print("\nStarting target training!\n")
			mode = "t"
			optimizer = create_optimizer(config, model, mode)
		else:
			print("\nStarting joint training!\n")
			mode = "j"
			optimizer = create_optimizer(config, model, mode)

		# Training the target prediction model / performing joint training, depending on the given configuration
		for epoch in range(0, t_epochs):

			if config["model"] != "Dummy":
				_train_one_epoch_mvcbm(mode, epoch, config, model, optimizer, loss_fn,
									   train_loader, target_class_weights, concepts_class_weights)

			target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, conf_matrix, \
			FP_names, FN_names = validate_epoch_mvcbm(epoch, config, model, val_loader, loss_fn, fold, roc, pr)

			t_results[fold, epoch, 0] = target_loss
			t_results[fold, epoch, 1] = summed_concepts_loss
			t_results[fold, epoch, 2] = total_loss
			t_results[fold, epoch, 3:] = tMetrics.get_tMetrics()

			if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
					config["training_mode"] == "joint":
				for concept_idx in range(len(concepts_loss)):
					c_results[fold, epoch, concept_idx, 0] = concepts_loss[concept_idx]
					c_results[fold, epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

			print_epoch_val_scores(
				config, mode, target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics,
				all_cMetrics, print_all_c=(config['dataset'] == 'app'))

		torch.save(model.state_dict(), join(checkpoint_dir, f"fold{fold}_model.pth"))
		print("\nTraining finished, model saved!", flush=True)
		print("Confusion matrix:")
		print(conf_matrix)

	print()
	print()
	print("------------------------")
	print(f'{config["k_folds"]}-FOLD CROSS VALIDATION COMPLETED')
	print("------------------------")
	print()

	roc.save(Path("ROC_curves"))
	pr.save(Path("PR_curves"))

	# Stop logging print-outs
	sys.stdout = old_stdout
	log_file.close()

	return None


def train_ssmvcbm_kfold(config, gen):
	"""
	Run the k-fold cross-validation for the SSMVCBM
	"""
	# Log the print-outs
	old_stdout = sys.stdout
	log_file = open(
		os.path.join(config["log_directory"], config['run_name'] + '_' + config['experiment_name'] + '_' +
					 str(config['seed']) + '.log'), 'w')
	sys.stdout = log_file

	# ---------------------------------
	#       Prepare data
	# ---------------------------------
	kfold = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])

	trainset, validset, testset = _get_data(config=config)

	labels = []
	for x in iter(DataLoader(trainset)):
		labels.append(x["label"].item())
	labels = np.array(labels)

	# ---------------------------------
	# Create temporary directories for models and data splits
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	splits_dir = os.path.join(config['log_directory'], 'splits')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(splits_dir):
		os.makedirs(splits_dir)

	# Important configuration parameters
	sc_epochs = config["sc_epochs"]
	adv_it = config["adversarial_it"]
	usc_epochs = config["usc_epochs"]
	d_epochs = config["d_epochs"]
	t_epochs = config["t_epochs"]
	try:
		beta = config["beta"]
	except KeyError:
		beta = 1
	try:
		gamma = config["usc_gamma"]
	except KeyError:
		gamma = 0

	# 4 metrics + loss for every concept, and summed concept loss
	sc_results = np.empty((config["k_folds"], sc_epochs, config["num_s_concepts"] + 1, (4 + 1)))
	t_results = np.empty((config["k_folds"], t_epochs, (11 + 1)))  # 11 metrics + target loss
	roc = ROC(config["k_folds"], range(1, t_epochs + 1))
	pr = PRC(config["k_folds"], range(1, t_epochs + 1))

	# Iterate over all folds
	for fold, (train_ids, val_ids) in enumerate(kfold.split(trainset, labels)):
		print()
		print("------------------------")
		print(f'FOLD {fold + 1}/{config["k_folds"]}')
		print("------------------------")
		# Instantiate dataloaders
		train_loader, val_loader = _create_data_loaders(config, gen, trainset, train_ids=train_ids,
														validset=validset, val_ids=val_ids)
		test_loader = DataLoader(
			testset, batch_size=config["val_batch_size"], num_workers=config["workers"], generator=gen)
		save_data_split(splits_dir, trainset, train_ids, validset, val_ids)

		# Retrieve labels
		train_labels = []
		val_labels = []
		all_c = []
		for concept_idx in range(config["num_s_concepts"]):
			c = []
			for x in iter(train_loader):
				if concept_idx == 0:
					train_labels.extend(x["label"].tolist())
				concepts = x["concepts"].numpy()
				c.extend(concepts[:, concept_idx])
			all_c.append(c)

		print("Length of training array", len(train_labels))
		print("Train target class distribution: ", Counter(train_labels))
		print("Train concepts class distribution: ")
		for concept_idx in range(len(all_c)):
			print("...", Counter(all_c[concept_idx]))

		val_all_c = []
		for concept_idx in range(config["num_s_concepts"]):
			c = []
			for x in iter(val_loader):
				if concept_idx == 0:
					val_labels.extend(x["label"].tolist())
				concepts = x["concepts"].numpy()
				c.extend(concepts[:, concept_idx])
			val_all_c.append(c)
		print("\nLength of validation array", len(val_labels))
		print("Validation target class distribution: ", Counter(val_labels))
		print("Validation concepts class distribution: ")
		for concept_idx in range(len(val_all_c)):
			print("...", Counter(val_all_c[concept_idx]))

		target_class_weights = compute_class_weight(
			class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
		concepts_class_weights = calc_concept_weights(all_c)

		# Initialize model and training objects
		model = create_model(config)
		model.to(config["device"])
		loss_fn = create_loss(config)

		###########################################################################################################
		# Supervised concept learning
		###########################################################################################################
		mode = "sc"
		print("\nStarting supervised concepts training!\n")
		# Prepare parameters
		for name, child in model.named_children():
			if name == "sc_model":
				child.apply(unfreeze_module)
			else:
				print(f"Freezing {name}...")
				child.apply(freeze_module)

		sc_optimizer = create_optimizer(config, model, mode)

		for epoch in range(0, sc_epochs):

			_train_one_epoch_ssmvcbm(
				mode, epoch, config, model, sc_optimizer, loss_fn, train_loader, target_class_weights,
				concepts_class_weights, beta, gamma)
			val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, _, _, _, _ = validate_epoch_ssmvcbm(
				epoch, mode, config, model, val_loader, loss_fn, beta, gamma)

			for concept_idx in range(len(val_s_concepts_loss)):
				sc_results[fold, epoch, concept_idx, 0] = val_s_concepts_loss[concept_idx]
				sc_results[fold, epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

			sc_results[fold, epoch, -1, 0] = val_loss
			print_all_c = (config['dataset'] != 'mawa')
			print_epoch_val_scores_(config, mode, val_loss, val_s_concepts_loss, tMetrics, all_cMetrics,
										print_all_c)

		###########################################################################################################
		# Representation learning
		###########################################################################################################
		print("\nStarting representation learning!\n")
		for k in range(adv_it):
			# Train the representation encoder module
			print(f"\nAdversarial training iteration: {k + 1}/{adv_it}\n")
			mode = "usc"
			print("\nFitting representation encoder...\n")
			# prepare parameters
			for name, child in model.named_children():
				if name in ["usc_model", "t_model"]:
					child.apply(unfreeze_module)
				else:
					print(f"Freezing {name}...")
					child.apply(freeze_module)
			usc_optimizer = create_optimizer(config, model, mode)
			for epoch in range(0, usc_epochs):
				_train_one_epoch_ssmvcbm(
					mode, epoch, config, model, usc_optimizer, loss_fn, train_loader, target_class_weights,
					concepts_class_weights, beta, gamma)
				val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, _, _, _, us_cov = validate_epoch_ssmvcbm(
					epoch, mode, config, model, val_loader, loss_fn, beta, gamma)
				print(f"    -- Epoch {epoch} val loss: ", val_loss)

				us_cov = us_cov.cpu().detach().numpy()

			# Train the adversary
			mode = "d"
			print("\nFitting the adversary...\n")
			# prepare parameters
			for name, child in model.named_children():
				if name == "discriminator":
					child.apply(unfreeze_module)
				else:
					print(f"Freezing {name}...")
					child.apply(freeze_module)
			d_optimizer = create_optimizer(config, model, mode)
			for epoch in range(0, d_epochs):
				_train_one_epoch_ssmvcbm(
					mode, epoch, config, model, d_optimizer, loss_fn, train_loader, target_class_weights,
					concepts_class_weights, beta, gamma)
				val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, _, _, _, _ = validate_epoch_ssmvcbm(
					epoch, mode, config, model, val_loader, loss_fn, beta, gamma)
				print(f"    -- Epoch {epoch} val loss: ", val_loss)

		###########################################################################################################
		# Target learning
		###########################################################################################################
		mode = "t"
		print("\nStarting target training!\n")
		# Prepare parameters
		for name, child in model.named_children():
			if name == "t_model":
				child.apply(unfreeze_module)
			else:
				print(f"Freezing {name}...")
				child.apply(freeze_module)

		# Re-initialise target model weights
		for name, child in model.named_children():
			if name == "t_model":
				for t_name, t_child in child.named_children():
					t_child.reset_parameters()

		t_optimizer = create_optimizer(config, model, mode)

		for epoch in range(0, t_epochs):
			_train_one_epoch_ssmvcbm(
				mode, epoch, config, model, t_optimizer, loss_fn, train_loader, target_class_weights,
				concepts_class_weights, beta, gamma)

			val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, conf_matrix, FP_names, FN_names, _ = \
				validate_epoch_ssmvcbm(epoch, mode, config, model, val_loader, loss_fn, beta, gamma, fold, roc, pr)

			t_results[fold, epoch, 0] = val_loss
			t_results[fold, epoch, 1:] = tMetrics.get_tMetrics()

			print_all_c = (config['dataset'] != 'mawa')
			print_epoch_val_scores_(config, mode, val_loss, val_s_concepts_loss, tMetrics, all_cMetrics,
									print_all_c)

		torch.save(model.state_dict(), join(checkpoint_dir, f"fold{fold}_model.pth"))
		print("\nTraining finished, model saved!", flush=True)
		print("Confusion matrix:")
		print(conf_matrix)
		print("List of false positive images: ", FP_names)
		print("List of false negative images: ", FN_names)

	print()
	print()
	print("------------------------")
	print(f'{config["k_folds"]}-FOLD CROSS VALIDATION COMPLETED')
	print("------------------------")
	print()

	roc.save(Path("ROC_curves"))
	pr.save(Path("PR_curves"))

	# Stop logging print-outs
	sys.stdout = old_stdout
	log_file.close()

	return None


def train_mvcbm(config, gen):
	"""
	Train and test an MVCBM model on a single train-test split
	"""
	# Log the print-outs
	old_stdout = sys.stdout
	log_file = open(
		os.path.join(config["log_directory"], config['run_name'] + '_' + config['experiment_name'] + '_' +
					 str(config['seed']) + '.log'), 'w')
	sys.stdout = log_file

	# ---------------------------------
	#       Prepare data
	# ---------------------------------
	trainset, validset, testset = _get_data(config=config)

	# Retrieve labels
	train_labels = []
	test_labels = []
	all_c = [[] for _ in range(config['num_concepts'])]

	tmp_loader = DataLoader(trainset, batch_size=config['train_batch_size'])
	bar = progressbar.ProgressBar(maxval=len(tmp_loader))
	bar.start()
	cnt = 0
	for x in iter(tmp_loader):
		train_labels.extend(x["label"].cpu().numpy().tolist())
		concepts = x["concepts"].cpu().numpy()
		for concept_idx in range(len(all_c)):
			all_c[concept_idx].extend(concepts[:, concept_idx].tolist())
		bar.update(cnt)
		cnt += 1
	bar.finish()

	tmp_loader = DataLoader(testset, batch_size=config['train_batch_size'])
	bar = progressbar.ProgressBar(maxval=len(tmp_loader))
	bar.start()
	cnt = 0
	for x in iter(tmp_loader):
		test_labels.extend(x["label"].cpu().numpy().tolist())
		bar.update(cnt)
		cnt += 1
	bar.finish()

	train_labels = np.array(train_labels)
	test_labels = np.array(test_labels)

	print("Length of training array", len(train_labels))
	print("Length of test array", len(test_labels))
	print("Train target class distribution: ", Counter(train_labels))
	print("Train concepts class distribution: ")
	for concept_idx in range(len(all_c)):
		print("...", Counter(all_c[concept_idx]))
	print("Test target class distribution: ", Counter(test_labels))
	print()

	# ---------------------------------
	# Create temporary directory for models
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# Numbers of training epochs
	if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
			config["training_mode"] == "joint":
		c_epochs = config["j_epochs"]
		t_epochs = config["j_epochs"]
	elif (config["model"] == "MVCBM" or config["model"] == "CBM") and \
			config["training_mode"] == "sequential":
		c_epochs = config["c_epochs"]
		t_epochs = config["t_epochs"]
	else:
		c_epochs = None
		t_epochs = config["t_epochs"]

	c_results = np.empty((c_epochs, config["num_concepts"], (4 + 1))) if c_epochs is not None else None
	t_results = np.empty((t_epochs, (11 + 3)))  # 11 metrics + 3 losses

	# Instantiate dataloaders
	train_loader, _ = _create_data_loaders(config, gen, trainset, train_ids=np.arange(len(train_labels)))
	test_loader = DataLoader(testset, batch_size=config["val_batch_size"], num_workers=config["workers"],
							 generator=gen)

	# Concept and class weights
	target_class_weights = compute_class_weight(
		class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
	concepts_class_weights = calc_concept_weights(all_c)

	# Initialize model and training objects
	model = create_model(config)
	model.to(config["device"])
	loss_fn = create_loss(config)

	print("STARTING FINAL MODEL TRAINING!")
	print()

	# Concept learning
	if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
			config["training_mode"] == "sequential":
		print("\nStarting concepts training!\n")
		mode = "c"

		for name, child in model.named_children():
			if name.split("_")[0] == "t":
				child.apply(freeze_module)

		c_optimizer = create_optimizer(config, model, mode)

		for epoch in range(0, c_epochs):

			if config["model"] != "Dummy":
				_train_one_epoch_mvcbm(mode, epoch, config, model, c_optimizer, loss_fn,
									   train_loader, target_class_weights, concepts_class_weights)

			# Validate the model (either every epoc or at the very end)
			if config['validate_every_epoch'] or epoch == c_epochs - 1:
				target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, _, _, _ = \
					validate_epoch_mvcbm(epoch, config, model, test_loader, loss_fn)

				for concept_idx in range(len(concepts_loss)):
					c_results[epoch, concept_idx, 0] = concepts_loss[concept_idx]
					c_results[epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

				print_epoch_val_scores(config, mode, target_loss, concepts_loss,
									   summed_concepts_loss, total_loss, tMetrics, all_cMetrics,
									   print_all_c=(config['dataset'] == 'app'))

		# Prepare parameters for target training
		for name, child in model.named_children():
			if name.split("_")[0] == "t":
				child.apply(unfreeze_module)
			else:
				child.apply(freeze_module)

	# Sequential vs. joint optimisation
	if (config["model"] == "MVCBM" or config["model"] == "CBM") and \
			config["training_mode"] == "sequential":
		print("\nStarting target training!\n")
		mode = "t"
		optimizer = create_optimizer(config, model, mode)
	else:
		print("\nStarting joint training!\n")
		mode = "j"
		optimizer = create_optimizer(config, model, mode)

	for epoch in range(0, t_epochs):

		_train_one_epoch_mvcbm(
			mode, epoch, config, model, optimizer, loss_fn, train_loader, target_class_weights, concepts_class_weights)

		# Validate the model (either every epoch or at the very end)
		if config['validate_every_epoch'] or epoch == t_epochs - 1:
			target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics, conf_matrix, \
			FP_names, FN_names = validate_epoch_mvcbm(epoch, config, model, test_loader, loss_fn)

			t_results[epoch, 0] = target_loss
			t_results[epoch, 1] = summed_concepts_loss  # 0 for USVarMLP
			t_results[epoch, 2] = total_loss  # = target_loss for USVarMLP
			t_results[epoch, 3:] = tMetrics.get_tMetrics()

			if config["model"] in ["MVCBM", "CBM", "Dummy"] and \
					config["training_mode"] == "joint":
				for concept_idx in range(len(concepts_loss)):
					c_results[epoch, concept_idx, 0] = concepts_loss[concept_idx]
					c_results[epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

			print_epoch_val_scores(
				config, mode, target_loss, concepts_loss, summed_concepts_loss, total_loss, tMetrics, all_cMetrics,
				print_all_c=(config['dataset'] == 'app'))

	torch.save(model.state_dict(), join(checkpoint_dir, "final_model_" + config['run_name'] + '_' +
										config['experiment_name'] + ".pth"))
	print("\nTraining finished, model saved!", flush=True)

	print("\nEVALUATION ON THE TEST SET:\n")
	t_metric_names = ["t_loss", "c_loss", "total_loss", "t_ppv", "t_npv", "t_sensitivity",
					  "t_specificity", "t_accuracy", "t_balanced_accuracy", "t_f1_1", "t_f1_0", "t_f1_macro",
					  "t_auroc", "t_aupr"]

	if c_results is not None:
		print(f"Summed concepts loss on last epoch: {c_results[-1, -1, 0]}")

		if config['dataset'] != 'mawa':
			for concept_idx in range(config["num_concepts"]):
				c_metric_names = [f"sc{concept_idx}_loss", f"sc{concept_idx}_accuracy",
								  f"sc{concept_idx}_f1_macro", f"sc{concept_idx}_auroc", f"sc{concept_idx}_aupr"]
				print(f"(Concept {concept_idx}) Test results on last epoch: ")
				for metric_idx, metric_name in enumerate(c_metric_names):
					print(f"    {metric_name}: {c_results[-1, concept_idx, metric_idx]}")

		print(f"Averaged concepts test metrics on last epoch:")
		c_metric_names = [f"loss", f"accuracy", f"f1_macro", f"auroc", f"aupr"]
		for metric_idx, metric_name in enumerate(c_metric_names):
			print(f"    {metric_name}: {sum([c_results[-1, i, metric_idx] for i in range(config['num_concepts'])]) / config['num_concepts']}")

	print(f"(Target) Test results on last epoch: ")
	for metric_idx, metric_name in enumerate(t_metric_names):
		print(f"    {metric_name}: {t_results[-1, metric_idx]}")
	print("Confusion matrix:")
	print(conf_matrix)
	print("List of false positive images: ", FP_names)
	print("List of false negative images: ", FN_names)

	# Stop logging print-outs
	sys.stdout = old_stdout
	log_file.close()

	return None


def train_ssmvcbm(config, gen):
	"""
	Train and test an SSMVCBM model on a single train-test split
	"""
	# Log the print-outs
	old_stdout = sys.stdout
	log_file = open(
		os.path.join(
			config["log_directory"], config['run_name'] + '_' + config['experiment_name'] + '_' +
									 str(config['seed']) + '.log'), 'w')
	sys.stdout = log_file

	# ---------------------------------
	#       Prepare data
	# ---------------------------------
	trainset, validset, testset = _get_data(config=config)

	# Retrieve the labels
	train_labels = []
	test_labels = []
	all_c = [[] for _ in range(config['num_s_concepts'])]

	tmp_loader = DataLoader(trainset, batch_size=config['train_batch_size'])
	bar = progressbar.ProgressBar(maxval=len(tmp_loader))
	bar.start()
	cnt = 0
	for x in iter(tmp_loader):
		train_labels.extend(x["label"].cpu().numpy().tolist())
		concepts = x["concepts"].cpu().numpy()
		for concept_idx in range(len(all_c)):
			all_c[concept_idx].extend(concepts[:, concept_idx].tolist())
		bar.update(cnt)
		cnt += 1
	bar.finish()

	tmp_loader = DataLoader(testset, batch_size=config['train_batch_size'])
	bar = progressbar.ProgressBar(maxval=len(tmp_loader))
	bar.start()
	cnt = 0
	for x in iter(tmp_loader):
		test_labels.extend(x["label"].cpu().numpy().tolist())
		bar.update(cnt)
		cnt += 1
	bar.finish()

	train_labels = np.array(train_labels)
	test_labels = np.array(test_labels)

	print("Length of training array", len(train_labels))
	print("Length of test array", len(test_labels))
	print("Train target class distribution: ", Counter(train_labels))
	print("Train concepts class distribution: ")
	for concept_idx in range(len(all_c)):
		print("...", Counter(all_c[concept_idx]))
	print("Test target class distribution: ", Counter(test_labels))
	print()

	# ---------------------------------
	# Create temporary directory for models
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# Important parameters
	sc_epochs = config["sc_epochs"]
	adv_it = config["adversarial_it"]
	usc_epochs = config["usc_epochs"]
	d_epochs = config["d_epochs"]
	t_epochs = config["t_epochs"]
	try:
		beta = config["beta"]
	except KeyError:
		beta = 1
	try:
		gamma = config["usc_gamma"]
	except KeyError:
		gamma = 0

	sc_results = np.empty((sc_epochs, config["num_s_concepts"] + 1, (4 + 1)))
	t_results = np.empty((t_epochs, (11 + 1)))  # 11 metrics + the target loss

	# Instantiate dataloaders
	train_loader, _ = _create_data_loaders(config, gen, trainset, train_ids=np.arange(len(train_labels)))
	test_loader = DataLoader(testset, batch_size=config["val_batch_size"], num_workers=config["workers"],
							 generator=gen)

	# Class and concept weights
	target_class_weights = compute_class_weight(
		class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
	concepts_class_weights = calc_concept_weights(all_c)

	# Initialize model and training objects
	model = create_model(config)
	model.to(config["device"])
	loss_fn = create_loss(config)

	print("STARTING FINAL MODEL TRAINING!")
	print()

	###########################################################################################################
	# Supervised concept learning
	###########################################################################################################
	mode = "sc"
	print("\nStarting supervised concepts training!\n")
	# Prepare parameters
	for name, child in model.named_children():
		if name == "sc_model":
			child.apply(unfreeze_module)
		else:
			print(f"Freezing {name}...")
			child.apply(freeze_module)

	sc_optimizer = create_optimizer(config, model, mode)

	for epoch in range(0, sc_epochs):
		_train_one_epoch_ssmvcbm(
			mode, epoch, config, model, sc_optimizer, loss_fn, train_loader, target_class_weights,
			concepts_class_weights, beta, gamma)

		# Validate the model either every epoch or at the very end
		if config['validate_every_epoch'] or epoch == sc_epochs - 1:
			val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, _, _, _, _ = validate_epoch_ssmvcbm(
				epoch, mode, config, model, test_loader, loss_fn, beta, gamma)

			for concept_idx in range(len(val_s_concepts_loss)):
				sc_results[epoch, concept_idx, 0] = val_s_concepts_loss[concept_idx]
				sc_results[epoch, concept_idx, 1:5] = all_cMetrics[concept_idx].get_cMetrics()

			sc_results[epoch, -1, 0] = val_loss
			print_all_c = (config['dataset'] != 'mawa')
			print_epoch_val_scores_(config, mode, val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, print_all_c)

	###########################################################################################################
	# Representation learning
	###########################################################################################################
	print("\nStarting representation learning!\n")
	for k in range(adv_it):
		print(f"\nAdversarial training iteration: {k + 1}/{adv_it}\n")
		mode = "usc"
		print("\nFitting representation encoder...\n")
		# Prepare parameters
		for name, child in model.named_children():
			if name in ["usc_model", "t_model"]:
				child.apply(unfreeze_module)
			else:
				print(f"Freezing {name}...")
				child.apply(freeze_module)
		usc_optimizer = create_optimizer(config, model, mode)
		for epoch in range(0, usc_epochs):
			_train_one_epoch_ssmvcbm(
				mode, epoch, config, model, usc_optimizer, loss_fn, train_loader, target_class_weights,
				concepts_class_weights, beta, gamma, k)

			# Validate the model either every epoch or at the very end
			if config['validate_every_epoch']:
				val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, _, _, _, us_cov = validate_epoch_ssmvcbm(
					epoch, mode, config, model, test_loader, loss_fn, beta, gamma)
				print(f"    -- Epoch {epoch} test loss: ", val_loss)

				us_cov = us_cov.cpu().detach().numpy()

		# Train the adversary used for regularisation
		mode = "d"
		print("\nFitting the adversary...\n")
		# Prepare parameters
		for name, child in model.named_children():
			if name == "discriminator":
				child.apply(unfreeze_module)
			else:
				print(f"Freezing {name}...")
				child.apply(freeze_module)
		d_optimizer = create_optimizer(config, model, mode)
		for epoch in range(0, d_epochs):
			_train_one_epoch_ssmvcbm(
				mode, epoch, config, model, d_optimizer, loss_fn, train_loader, target_class_weights,
				concepts_class_weights, beta, gamma, k)

			if config['validate_every_epoch']:
				val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, _, _, _, _ = validate_epoch_ssmvcbm(
					epoch, mode, config, model, test_loader, loss_fn, beta, gamma)
				print(f"    -- Epoch {epoch} test loss: ", val_loss)

	###########################################################################################################
	# Target learning
	###########################################################################################################
	mode = "t"
	print("\nStarting target training!\n")
	# Prepare parameters
	for name, child in model.named_children():
		if name == "t_model":
			child.apply(unfreeze_module)
		else:
			print(f"Freezing {name}...")
			child.apply(freeze_module)
	# Re-initialize target model weights
	for name, child in model.named_children():
		if name == "t_model":
			for t_name, t_child in child.named_children():
				t_child.reset_parameters()

	t_optimizer = create_optimizer(config, model, mode)

	for epoch in range(0, t_epochs):
		_train_one_epoch_ssmvcbm(
			mode, epoch, config, model, t_optimizer, loss_fn, train_loader, target_class_weights,
			concepts_class_weights, beta, gamma)

		# Validate the model either every epoch or at the very end
		if config['validate_every_epoch'] or epoch == t_epochs - 1:
			val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, conf_matrix, FP_names, FN_names, _ = \
				validate_epoch_ssmvcbm(epoch, mode, config, model, test_loader, loss_fn, beta, gamma)
			t_results[epoch, 0] = val_loss
			t_results[epoch, 1:] = tMetrics.get_tMetrics()

			print_all_c = (config['dataset'] != 'mawa')
			print_epoch_val_scores_(config, mode, val_loss, val_s_concepts_loss, tMetrics, all_cMetrics, print_all_c)

	torch.save(model.state_dict(), join(checkpoint_dir, "final_model_" + config['run_name'] + '_' +
										config['experiment_name'] + ".pth"))
	print("\nTraining finished, model saved!", flush=True)

	# Evaluate on the test data
	print("\nEVALUATION ON TEST SET:\n")
	t_metric_names = ["t_loss", "t_ppv", "t_npv", "t_sensitivity",
					  "t_specificity", "t_accuracy", "t_balanced_accuracy", "t_f1_1", "t_f1_0", "t_f1_macro",
					  "t_auroc", "t_aupr"]

	if sc_results is not None:
		print(f"Summed concepts loss on last epoch: {sc_results[-1, -1, 0]}")

		if config['dataset'] != 'mawa':
			for concept_idx in range(config["num_s_concepts"]):
				c_metric_names = [f"sc{concept_idx}_loss", f"sc{concept_idx}_accuracy",
								  f"sc{concept_idx}_f1_macro", f"sc{concept_idx}_auroc", f"sc{concept_idx}_aupr"]
				print(f"(Concept {concept_idx}) Test results on last epoch: ")
				for metric_idx, metric_name in enumerate(c_metric_names):
					print(f"    {metric_name}: {sc_results[-1, concept_idx, metric_idx]}")

		print(f"Averaged concepts test metrics on last epoch:")
		c_metric_names = [f"loss", f"accuracy", f"f1_macro", f"auroc", f"aupr"]
		for metric_idx, metric_name in enumerate(c_metric_names):
			print(f"    {metric_name}: {sum([sc_results[-1, i, metric_idx] for i in range(config['num_s_concepts'])]) / config['num_s_concepts']}")

	print(f"(Target) Test results on last epoch: ")
	for metric_idx, metric_name in enumerate(t_metric_names):
		print(f"    {metric_name}: {t_results[-1, metric_idx]}")
	print("Confusion matrix:")
	print(conf_matrix)

	# Stop logging print-outs
	sys.stdout = old_stdout
	log_file.close()

	return None


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config")
	args = parser.parse_args()
	argsdict = vars(args)

	with open(argsdict["config"], "r") as f:
		config = yaml.safe_load(f)
	config["filename"] = argsdict["config"]

	# Ensure reproducibility
	random.seed(config["seed"])
	np.random.seed(config["seed"])
	gen = torch.manual_seed(config["seed"])
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)

	# Choose the relevant model training routine
	if config['model'] == 'SSMVCBM':
		train = train_ssmvcbm_kfold
		train_final = train_ssmvcbm
	elif config['model'] == 'MVCBM':
		train = train_mvcbm_kfold
		train_final = train_mvcbm
	elif config['model'] == 'CBM':
		train = train_mvcbm_kfold
		train_final = train_mvcbm

	if config["validate"]:
		train(config, gen)
	else:
		train_final(config, gen)


if __name__ == "__main__":
	main()

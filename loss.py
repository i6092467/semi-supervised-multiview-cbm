"""
Utility methods for constructing loss functions
"""
from typing import Optional

from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# TODO: replace this constant with an argument from the config
DECORRELATE = False


def create_loss(config):
	"""
	Parse configuration file and return a relevant loss function
	"""
	if config['model'] == 'MVCBM' or config['model'] == 'CBM':
		if config['training_mode'] == 'sequential':
			return MVCBLoss(num_classes=config['num_classes'])
		elif config['training_mode'] == 'joint':
			return MVCBLoss(num_classes=config['num_classes'], alpha=config['alpha'])
	elif config["model"] in ["SSMVCBM"]:
		return SSMVCBLoss(num_classes=config['num_classes'])
	else:
		return nn.BCELoss()


class MVCBLoss(nn.Module):
	"""
	Loss function for the (multiview) concept bottleneck model
	"""

	# NOTE: this loss function is also applicable to the vanilla CBMs
	def __init__(
			self,
			num_classes: Optional[int] = 2,
			target_class_weight: Optional[Tensor] = None,
			target_sample_weight: Optional[Tensor] = None,
			c_weights: Optional[Tensor] = None,
			reduction: str = "mean",
			alpha: float = 1) -> None:
		"""
		Initializes the loss object

		@param num_classes: the number of the classes of the target variable
		@param target_class_weight: weight per target's class
		@param target_sample_weight: target class weights per data point
		@param c_weights: concepts weights
		@param reduction: reduction to apply to the output of the CE loss
		@param alpha: parameter controlling the trade-off between the target and concept prediction during the joint
						optimization. The higher the @alpha, the high the weight of the concept prediction loss
		"""
		super(MVCBLoss, self).__init__()
		self.num_classes = num_classes
		self.target_class_weight = target_class_weight
		# NOTE: these weights will need to be updated every time before the loss is computed
		self.target_sample_weight = target_sample_weight
		self.c_weights = c_weights
		self.reduction = reduction
		self.alpha = alpha

	def forward(self, concepts_pred: Tensor, concepts_true: Tensor,
				target_pred_probs: Tensor, target_pred_logits: Tensor, target_true: Tensor) -> Tensor:
		"""
		Computes the loss for the given predictions

		@param concepts_pred: predicted concept values
		@param concepts_true: ground-truth concept values
		@param target_pred_probs: predicted probabilities, aka normalized logits, for the target variable
		@param target_pred_logits: predicted logits for the target variable
		@param target_true: ground-truth target variable values
		@return: target prediction loss, a tensor of prediction losses for each of the concepts, summed concept
					prediction loss and the total loss
		"""

		summed_concepts_loss = 0
		concepts_loss = []

		# NOTE: all concepts are assumed to be binary-valued
		# TODO: introduce continuously- and categorically-valued concepts
		for concept_idx in range(concepts_true.shape[1]):
			w = self.target_sample_weight * self.c_weights[concept_idx] if self.target_sample_weight is not None else None
			c_loss = F.binary_cross_entropy(
				concepts_pred[:, concept_idx], concepts_true[:, concept_idx].float(), weight=w, reduction=self.reduction)
			concepts_loss.append(c_loss)
			summed_concepts_loss += c_loss

		if self.num_classes == 2:
			target_loss = F.binary_cross_entropy(
				target_pred_probs, target_true, weight=self.target_sample_weight, reduction=self.reduction)
		else:
			target_loss = F.cross_entropy(
				target_pred_logits, target_true.long(), weight=self.target_class_weight, reduction=self.reduction)

		total_loss = target_loss + self.alpha * summed_concepts_loss

		return target_loss, concepts_loss, summed_concepts_loss, total_loss


class SSMVCBLoss(nn.Module):
	"""
	Loss function for the semi-supervised multiview concept bottleneck model
	"""

	def __init__(
			self,
			num_classes: Optional[int] = 2,
			target_class_weight: Optional[Tensor] = None,
			target_sample_weight: Optional[Tensor] = None,
			c_weights: Optional[Tensor] = None,
			reduction: str = "mean"
	) -> None:
		"""
		Initializes the loss object

		@param num_classes: the number of the classes of the target variable
		@param target_class_weight: weight per target's class
		@param target_sample_weight: target class weights per data point
		@param c_weights: concepts weights
		@param reduction: reduction to apply to the output of the CE loss
		"""
		super(SSMVCBLoss, self).__init__()
		self.num_classes = num_classes
		self.target_class_weight = target_class_weight
		# NOTE: these weights will need to be updated every time before the loss is computed
		self.target_sample_weight = target_sample_weight
		self.c_weights = c_weights
		self.reduction = reduction

	def forward(self, s_concepts_pred: Tensor, discr_concepts_pred: Tensor, concepts_true: Tensor,
				target_pred_probs: Tensor, target_pred_logits: Tensor, target_true: Tensor,
				us_concepts_sample: Tensor) -> Tensor:
		"""
		Computes the loss for the given predictions

		@param s_concepts_pred: predicted concept values
		@param discr_concepts_pred: concept predictions made by the adversary
		@param concepts_true: ground-truth concept values
		@param target_pred_probs: predicted probabilities, aka normalized logits, for the target variable
		@param target_pred_logits: predicted logits for the target variable
		@param target_true: ground-truth target variable values
		@param us_concepts_sample: a sample of the unsupervised representations
		@return: target prediction loss, a tensor of concept prediction losses, summed concept prediction loss,
					summed concept prediction loss for the adversary (its positive and negative values) and
					summed representation de-correlation loss
		"""
		summed_discr_concepts_loss = 0
		summed_s_concepts_loss = 0
		s_concepts_loss = []

		# Supervised concepts loss
		# NOTE: all concepts are assumed to be binary-valued
		# TODO: introduce continuously- and categorically-valued concepts
		for concept_idx in range(concepts_true.shape[1]):
			w = self.target_sample_weight * self.c_weights[concept_idx] if self.target_sample_weight is not None else None
			c_loss = F.binary_cross_entropy(
				s_concepts_pred[:, concept_idx], concepts_true[:, concept_idx].float(), weight=w, reduction=self.reduction)
			s_concepts_loss.append(c_loss)
			summed_s_concepts_loss += c_loss

		# Adversarial loss term
		for concept_idx in range(concepts_true.shape[1]):
			w = self.c_weights[concept_idx] if self.c_weights is not None else None
			c_loss = F.binary_cross_entropy(
				discr_concepts_pred[:, concept_idx], s_concepts_pred[:, concept_idx], weight=w, reduction=self.reduction)
			summed_discr_concepts_loss += c_loss

		# Unsupervised representation loss term
		summed_gen_concepts_loss = -summed_discr_concepts_loss

		# Compute covariance among the dimensions of the unsupervised representation
		# NOTE: can cause issues during the optimisation
		# NOTE: this loss term is disabled in the current implementation
		if DECORRELATE:
			cov = torch.cov(us_concepts_sample.T)
		else:
			cov = torch.zeros((us_concepts_sample.shape[1], us_concepts_sample.shape[1]))
		cov = cov.fill_diagonal_(0)
		us_corr_loss = torch.square(torch.linalg.matrix_norm(cov))

		# Target prediction loss term
		if self.num_classes == 2:
			target_loss = F.binary_cross_entropy(
				target_pred_probs, target_true, weight=self.target_sample_weight, reduction=self.reduction)
		else:
			target_loss = F.cross_entropy(
				target_pred_logits, target_true.long(), weight=self.target_class_weight, reduction=self.reduction)

		return target_loss, s_concepts_loss, summed_s_concepts_loss, summed_discr_concepts_loss, \
			   summed_gen_concepts_loss, us_corr_loss


def calc_concept_weights(all_c):
	"""
	Computes class weights for every list element all_c[i] corresponding to a set of the i-th concept values
	"""
	concepts_class_weights = []
	for concept_idx in range(len(all_c)):
		c_class_weights = compute_class_weight(class_weight="balanced", classes=[0, 1], y=all_c[concept_idx])
		concepts_class_weights.append(c_class_weights)
	return concepts_class_weights


def calc_concept_sample_weights(config, concepts_class_weights, batch_concepts):
	"""
	Assigns precomputed concept class weights to every sample in a batch.
	"""
	concepts_sample_weights = []
	for concept_idx in range(len(concepts_class_weights)):
		c_sample_weights = [concepts_class_weights[concept_idx][0] if int(
			batch_concepts[i][concept_idx]) == 0 else concepts_class_weights[concept_idx][1] for i in
							range(len(batch_concepts))]

		concepts_sample_weights.append(torch.FloatTensor(c_sample_weights).to(config["device"]))

	return concepts_sample_weights

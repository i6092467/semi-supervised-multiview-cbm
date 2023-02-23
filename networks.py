"""
Neural network architectures and concept bottleneck models
"""
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import load, nn
from torch.nn import utils
from torchvision import models


def create_model(config):
	"""
	Parse the configuration file and return a relevant model
	"""
	if config['model'] == 'MVCBM' or config['model'] == 'CBM':
		return MVCBM(config)
	elif config['model'] == 'SSMVCBM':
		return SSMVCBM(config)
	else:
		print("Could not create model with name ", config["model"], "!")
		quit()


class ResNet18Encoder(nn.Module):
	"""
	Extracts a vectorial representation from an image using ResNet-18
	"""

	def __init__(self, model_directory: str = None):
		super(ResNet18Encoder, self).__init__()

		# ResNet
		self.img_feature_extractor = ResNet18(model_directory=model_directory)

		# Register forward hook to save output of convolutional part of ResNet
		self.img_feature_extractor.model.avgpool.register_forward_hook(self.get_features("img_features"))
		self.embedding_dim = self.img_feature_extractor.model.fc.in_features
		self.features = {}

	def get_features(self, name):
		def hook(model, input, output):
			self.features[name] = output.squeeze(3).squeeze(2)

		return hook

	def forward(self, image):
		self.img_feature_extractor(image)
		return self.features["img_features"]


class FCNNEncoder(nn.Module):
	"""
	Extracts a vectorial representation from a view using a simple fully connected network
	"""

	def __init__(self, num_inputs: int, num_outputs: int, num_hidden: int, num_deep: int):
		super(FCNNEncoder, self).__init__()

		self.fc0 = nn.Linear(num_inputs, num_hidden)
		self.bn0 = nn.BatchNorm1d(num_hidden)
		self.fcs = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)])
		self.bns = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_deep)])
		self.dp = nn.Dropout(0.05)
		self.out = nn.Linear(num_hidden, num_outputs)

		self.embedding_dim = num_outputs

	def forward(self, t):
		t = self.bn0(self.dp(F.relu(self.fc0(t))))
		for bn, fc in zip(self.bns, self.fcs):
			t = bn(self.dp(F.relu(fc(t))))
		return self.out(t)


class CModel(nn.Module):
	"""
	Implements an encoder mapping to representations or supervised concepts from multiple input views
	"""

	def __init__(self, num_concepts, aggregator, attention, device, out_activation,
				 encoder_arch: str = 'ResNet18', model_directory: str = None):
		super(CModel, self).__init__()
		self.name = "CModel"
		self.device = device
		self.encoder_arch = encoder_arch
		self.aggregator = aggregator
		self.num_concepts = num_concepts
		self.attention = attention
		self.out_activation = out_activation

		if encoder_arch == 'ResNet18':
			self.c_encoder = ResNet18Encoder(model_directory=model_directory).to(device)
		elif encoder_arch == 'FCNN':
			self.c_encoder = FCNNEncoder(num_inputs=500, num_outputs=128, num_hidden=256, num_deep=2).to(device)
		else:
			NotImplementedError('ERROR: encoder architecture not supported!')

		self.embedding_dim = self.c_encoder.embedding_dim
		self.hidden_size = self.embedding_dim
		if self.attention:
			self.W_q = nn.Linear(self.num_concepts, self.embedding_dim)
			self.W_k = nn.Linear(self.embedding_dim, self.embedding_dim)
			self.W_v = nn.Linear(self.embedding_dim, self.embedding_dim)
			self.c_attention = ScaledDotProduct(dropout=0, batch_first=True)

		self.c_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
		self.c_fc1 = nn.Linear(self.hidden_size, 256)
		self.c_fc2 = nn.Linear(256, 64)
		self.c_fc3 = nn.Linear(64, self.num_concepts)

	def forward(self, images_seq, mask):

		encoded_seq = torch.empty((images_seq.size(0), images_seq.size(1), self.embedding_dim), device=self.device)

		if self.encoder_arch == 'ResNet18':
			images_seq_flat = torch.reshape(
				images_seq, (images_seq.size(0) * images_seq.size(1),
							 images_seq.size(2),
							 images_seq.size(3),
							 images_seq.size(4))).float()
		elif self.encoder_arch == 'FCNN':
			images_seq_flat = torch.reshape(
				images_seq, (images_seq.size(0) * images_seq.size(1), images_seq.size(2))).float()

		encoded_seq_flat = self.c_encoder(images_seq_flat)
		encoded_seq = torch.reshape(encoded_seq_flat, (images_seq.size(0), images_seq.size(1), self.embedding_dim)).float()

		if self.aggregator == "lstm":
			if self.attention:
				q_embedding = torch.diag(torch.ones(self.num_concepts)).repeat(images_seq.size(0), 1, 1).to(
					device=self.device)
				lstm_output, (hn, cn) = self.c_rnn(encoded_seq)
				q = self.W_q(q_embedding)
				k = self.W_k(lstm_output)
				v = self.W_v(lstm_output)
				mask = (~mask).unsqueeze(1).repeat(1, self.num_concepts, 1)
				joint_repr, attn_weights = self.c_attention(q, k, v, attn_mask=mask)
			else:
				lengths = torch.sum(mask, dim=1).cpu()
				encoded_seq_packed = utils.rnn.pack_padded_sequence(encoded_seq, lengths, batch_first=True,
																	enforce_sorted=False)
				packed_output, (hn, cn) = self.c_rnn(encoded_seq_packed)
				output, input_sizes = utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
				joint_repr = hn.squeeze(0)  # last hidden state element (padding is ignored)
		else:
			if self.attention:
				q_embedding = torch.diag(torch.ones(self.num_concepts)).repeat(images_seq.size(0), 1, 1).to(
					device=self.device)
				q = self.W_q(q_embedding)
				k = self.W_k(encoded_seq)
				v = self.W_v(encoded_seq)
				mask = (~mask).unsqueeze(1).repeat(1, self.num_concepts, 1)
				joint_repr, attn_weights = self.c_attention(q, k, v, attn_mask=mask)
			else:
				mask = mask.unsqueeze(-1)
				encoded_seq = encoded_seq * mask.float()
				joint_repr = encoded_seq.sum(1) / mask.sum(1)

		concepts_pred = F.relu(self.c_fc1(joint_repr))
		concepts_pred = F.relu(self.c_fc2(concepts_pred))
		if self.attention:
			if self.out_activation == "sigmoid":
				concepts_pred = torch.diagonal(torch.sigmoid(self.c_fc3(concepts_pred)), dim1=1, dim2=2)
			elif self.out_activation == "tanh":
				concepts_pred = torch.diagonal(torch.tanh(self.c_fc3(concepts_pred)), dim1=1, dim2=2)
			elif self.out_activation == "relu":
				concepts_pred = torch.diagonal(F.relu(self.c_fc3(concepts_pred)), dim1=1, dim2=2)
			else:
				concepts_pred = torch.diagonal(self.c_fc3(concepts_pred), dim1=1, dim2=2)
		else:
			if self.out_activation == "sigmoid":
				concepts_pred = torch.sigmoid(self.c_fc3(concepts_pred))
			elif self.out_activation == "tanh":
				concepts_pred = torch.tanh(self.c_fc3(concepts_pred))
			elif self.out_activation == "relu":
				concepts_pred = F.relu(self.c_fc3(concepts_pred))
			else:
				concepts_pred = self.c_fc3(concepts_pred)

		if not self.attention:
			attn_weights = None

		return concepts_pred, attn_weights


class TModel(nn.Module):
	"""
	Implements a neural network predicting the target variable from the concept bottleneck layer
	"""

	def __init__(self, num_classes, num_s_concepts, num_us_concepts, num_ex_feat, t_hidden_dim, norm_bottleneck, fusion):
		super(TModel, self).__init__()
		self.name = "TModel"
		self.num_classes = num_classes
		self.num_s_concepts = num_s_concepts
		self.num_us_concepts = num_us_concepts
		self.num_ex_feat = num_ex_feat
		self.fusion = fusion
		self.t_hidden_dim = t_hidden_dim
		self.norm_bottleneck = norm_bottleneck
		self.bn = nn.BatchNorm1d(self.num_s_concepts + self.num_us_concepts)
		self.t_fc1 = nn.Linear(self.num_s_concepts + self.num_us_concepts + self.num_ex_feat, self.t_hidden_dim)
		if self.num_classes == 2:
			self.t_fc2 = nn.Linear(t_hidden_dim, 1)
		else:
			self.t_fc2 = nn.Linear(t_hidden_dim, self.num_classes)

	def forward(self, s_concepts_pred, us_concepts_pred, ex_feat):
		concepts_pred = torch.cat((s_concepts_pred, us_concepts_pred), 1)
		if self.norm_bottleneck:
			concepts_pred = self.bn(concepts_pred)
		if self.fusion:
			concepts_pred = torch.cat((concepts_pred, ex_feat), dim=1)
		target_pred_logits = F.relu(self.t_fc1(concepts_pred))
		target_pred_logits = self.t_fc2(target_pred_logits)

		if self.num_classes == 2:
			target_pred_probs = torch.sigmoid(target_pred_logits)
		else:
			target_pred_probs = torch.softmax(target_pred_logits, 1)

		return target_pred_probs, target_pred_logits


class Adversary(nn.Module):
	"""
	Implements a neural network predicting supervised concepts from the unsupervised representation,
	used for adversarial regularisation in the SSMVCBM
	"""

	def __init__(self, num_s_concepts, num_us_concepts):
		super(Adversary, self).__init__()
		self.name = "SupUnsupDiscriminator"
		self.num_s_concepts = num_s_concepts
		self.num_us_concepts = num_us_concepts
		self.fc1 = nn.Linear(self.num_us_concepts, 8)
		self.fc2 = nn.Linear(8, 12)
		self.fc3 = nn.Linear(12, 8)
		self.fc4 = nn.Linear(8, num_s_concepts)

	def forward(self, us_concepts):
		s_concept_pred = F.relu(self.fc1(us_concepts))
		s_concept_pred = F.relu(self.fc2(s_concept_pred))
		s_concept_pred = F.relu(self.fc3(s_concept_pred))
		s_concept_pred = torch.sigmoid(self.fc4(s_concept_pred))
		return s_concept_pred


class MVCBM(nn.Module):
	"""
	Multiview concept bottleneck model (MVCBM)
	"""

	# NOTE: a single-view CBM is a special case
	def __init__(self, config):
		super(MVCBM, self).__init__()
		self.name = config['model']
		self.device = config["device"]
		self.aggregator = config["aggregator"]
		self.num_concepts = config["num_concepts"]
		self.num_classes = config['num_classes']
		try:
			self.fusion = config["fusion"]
		except KeyError:
			self.fusion = False
		try:
			self.num_ex_feat = config["num_ex_feat"]
		except KeyError:
			self.num_ex_feat = 0
		try:
			self.t_hidden_dim = config["t_hidden_dim"]
		except KeyError:
			self.t_hidden_dim = 5
		try:
			self.attention = config["attention"]
		except KeyError:
			self.attention = False  # older configuration files may not have "attention" key
		self.encoder_arch = config['encoder_arch']

		if config['encoder_arch'] == 'ResNet18':
			self.c_encoder = ResNet18Encoder(model_directory=config['model_directory']).to(config["device"])
		elif config['encoder_arch'] == 'FCNN':
			self.c_encoder = FCNNEncoder(num_inputs=500, num_outputs=128, num_hidden=256, num_deep=2).to(config["device"])
		else:
			NotImplementedError('ERROR: encoder architecture not supported!')

		self.embedding_dim = self.c_encoder.embedding_dim
		self.hidden_size = self.embedding_dim
		if self.attention:
			self.W_q = nn.Linear(self.num_concepts, self.embedding_dim)
			self.W_k = nn.Linear(self.embedding_dim, self.embedding_dim)
			self.W_v = nn.Linear(self.embedding_dim, self.embedding_dim)
			self.c_attention = ScaledDotProduct(dropout=0, batch_first=True)

		self.c_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
		self.c_fc1 = nn.Linear(self.hidden_size, 256)
		self.c_fc2 = nn.Linear(256, 64)
		self.c_fc3 = nn.Linear(64, self.num_concepts)
		self.t_fc1 = nn.Linear(self.num_concepts + self.num_ex_feat, self.t_hidden_dim)

		if self.num_classes == 2:
			self.t_fc2 = nn.Linear(self.t_hidden_dim, 1)
		else:
			self.t_fc2 = nn.Linear(self.t_hidden_dim, self.num_classes)

	def forward(self, images_seq, mask, ex_feat, intervention_concept_ids=None, concepts_true=None):
		"""
		Forward pass of the MVCBM

		@param images_seq: an input sequence of views
		@param mask: a mas indicating which views in the sequence were not padded
		@param ex_feat: additional tabular features to append to the concept bottleneck layer
		@param intervention_concept_ids: indices of te concepts to intervene on; if None, no intervention is performed
		@param concepts_true: the ground-truth concept values, need to specify them for the intervention
		@return: predicted concepts values, predicted probabilities for the target variable, predicted logits for the
					target variable, attention weights (if the attention mechanism is enabled)
		"""
		if self.name == 'CBM':
			images_seq = images_seq[:, 0:1]
			mask = mask[:, 0:1]

		encoded_seq = torch.empty((images_seq.size(0), images_seq.size(1), self.embedding_dim), device=self.device)

		if self.encoder_arch == 'ResNet18':
			images_seq_flat = torch.reshape(
				images_seq, (images_seq.size(0) * images_seq.size(1),
							 images_seq.size(2),
							 images_seq.size(3),
							 images_seq.size(4))).float()
		elif self.encoder_arch == 'FCNN':
			images_seq_flat = torch.reshape(
				images_seq, (images_seq.size(0) * images_seq.size(1),
							 images_seq.size(2))).float()

		encoded_seq_flat = self.c_encoder(images_seq_flat)

		encoded_seq = torch.reshape(encoded_seq_flat, (images_seq.size(0), images_seq.size(1), self.embedding_dim)).float()

		if self.aggregator == "lstm":
			if self.attention:
				q_embedding = torch.diag(torch.ones(self.num_concepts)).repeat(images_seq.size(0), 1, 1).to(
					device=self.device)
				lstm_output, (hn, cn) = self.c_rnn(encoded_seq)
				q = self.W_q(q_embedding)
				k = self.W_k(lstm_output)
				v = self.W_v(lstm_output)
				mask = (~mask).unsqueeze(1).repeat(1, self.num_concepts, 1)
				joint_repr, attn_weights = self.c_attention(q, k, v, attn_mask=mask)
			else:
				lengths = torch.sum(mask, dim=1).cpu()
				encoded_seq_packed = utils.rnn.pack_padded_sequence(encoded_seq, lengths, batch_first=True,
																	enforce_sorted=False)
				packed_output, (hn, cn) = self.c_rnn(encoded_seq_packed)
				output, input_sizes = utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
				joint_repr = hn.squeeze(0)  # last hidden state element (padding is ignored)
		else:
			if self.attention:
				q_embedding = torch.diag(torch.ones(self.num_concepts)).repeat(images_seq.size(0), 1, 1).to(
					device=self.device)
				q = self.W_q(q_embedding)
				k = self.W_k(encoded_seq)
				v = self.W_v(encoded_seq)
				mask = (~mask).unsqueeze(1).repeat(1, self.num_concepts, 1)
				joint_repr, attn_weights = self.c_attention(q, k, v, attn_mask=mask)
			else:
				mask = mask.unsqueeze(-1)
				encoded_seq = encoded_seq * mask.float()
				joint_repr = encoded_seq.sum(1) / mask.sum(1)

		concepts_pred = F.relu(self.c_fc1(joint_repr))
		concepts_pred = F.relu(self.c_fc2(concepts_pred))
		if self.attention:
			concepts_pred = torch.diagonal(torch.sigmoid(self.c_fc3(concepts_pred)), dim1=1, dim2=2)
		else:
			concepts_pred = torch.sigmoid(self.c_fc3(concepts_pred))

		if intervention_concept_ids is not None:
			for concept_idx in range(self.num_concepts):
				if concept_idx in intervention_concept_ids:
					concepts_pred[:, concept_idx] = concepts_true[:, concept_idx]

		if self.fusion:
			concepts_pred = torch.cat((concepts_pred, ex_feat), dim=1)
		target_pred_logits = F.relu(self.t_fc1(concepts_pred))
		target_pred_logits = self.t_fc2(target_pred_logits)
		if self.num_classes == 2:
			target_pred_probs = torch.sigmoid(target_pred_logits)
		else:
			target_pred_probs = torch.softmax(target_pred_logits, 1)

		if not self.attention:
			attn_weights = None

		return concepts_pred, target_pred_probs, target_pred_logits, attn_weights


class SSMVCBM(nn.Module):
	"""
	Semi-supervised multiview concept bottleneck model (SSMVCBM)
	"""
	def __init__(self, config):
		super(SSMVCBM, self).__init__()
		self.name = "SSMVCBM"
		self.device = config["device"]
		self.aggregator = config["aggregator"]
		self.num_s_concepts = config["num_s_concepts"]
		self.num_us_concepts = config["num_us_concepts"]
		self.t_hidden_dim = config["t_hidden_dim"]
		self.usc_out_activation = config["usc_out_activation"]
		try:
			self.fusion = config["fusion"]
		except KeyError:
			self.fusion = False
		try:
			self.num_ex_feat = config["num_ex_feat"]
		except KeyError:
			self.num_ex_feat = 0
		try:
			self.attention = config["attention"]
		except KeyError:
			self.attention = False
		try:
			self.usc_attention = config["usc_attention"]
		except KeyError:
			self.usc_attention = self.attention
		try:
			self.norm_bottleneck = config["norm_bottleneck"]
		except KeyError:
			self.norm_bottleneck = False
		try:
			self.num_classes = config['num_classes']
		except KeyError:
			self.num_classes = 2
		self.sc_model = CModel(self.num_s_concepts, self.aggregator, self.attention, self.device,
							   out_activation="sigmoid", model_directory=config['model_directory'],
							   encoder_arch=config['encoder_arch'])
		self.usc_model = CModel(self.num_us_concepts, self.aggregator, self.usc_attention,
								self.device, out_activation=self.usc_out_activation,
								model_directory=config['model_directory'], encoder_arch=config['encoder_arch'])
		self.discriminator = Adversary(self.num_s_concepts, self.num_us_concepts)
		self.t_model = TModel(self.num_classes, self.num_s_concepts, self.num_us_concepts, self.num_ex_feat,
							  self.t_hidden_dim, self.norm_bottleneck, self.fusion)

	def forward(self, images_seq, mask, ex_feat, intervention_concept_ids=None, concepts_true=None):
		"""
		Forward pass of the SSMVCBM

		@param images_seq: an input sequence of views
		@param mask: a mas indicating which views in the sequence were not padded
		@param ex_feat: additional tabular features to append to the concept bottleneck layer
		@param intervention_concept_ids: indices of te concepts to intervene on; if None, no intervention is performed
		@param concepts_true: the ground-truth concept values, need to specify them for the intervention
		@return: predicted concepts values, representations, concept attention weights, representation attention weights,
					concept prediction from the adversary, predicted probabilities for the target variable,
					predicted logits for the target variable
		"""
		s_concepts_pred, s_attn_weights = self.sc_model(images_seq, mask)
		us_concepts_pred, us_attn_weights = self.usc_model(images_seq, mask)
		discr_concepts_pred = self.discriminator(us_concepts_pred)

		if intervention_concept_ids is not None:
			for concept_idx in range(self.num_s_concepts):
				if concept_idx in intervention_concept_ids:
					s_concepts_pred[:, concept_idx] = concepts_true[:, concept_idx]

		target_pred_probs, target_pred_logits = self.t_model(s_concepts_pred, us_concepts_pred, ex_feat)

		if not self.attention:
			s_attn_weights = None
		if not self.usc_attention:
			us_attn_weights = None

		return s_concepts_pred, us_concepts_pred, s_attn_weights, us_attn_weights, discr_concepts_pred, \
			   target_pred_probs, target_pred_logits


class ResNet18(nn.Module):
	"""
	Pretrained ResNet-18 from PyTorch
	"""
	def __init__(self, model_directory: str = None):
		super(ResNet18, self).__init__()
		self.name = "ResNet18"
		if model_directory is not None:
			os.environ["TORCH_HOME"] = os.path.join(model_directory, 'resnet')
		else:
			os.environ["TORCH_HOME"] = "models/resnet"  # setting the environment variable
		self.model = models.resnet18(pretrained=False)
		# NOTE: weights can pre-downloaded from https://github.com/fregu856/deeplabv3
		if model_directory is not None:
			self.model.load_state_dict(load(os.path.join(model_directory, 'resnet/resnet18-5c106cde.pth')))
		else:
			self.model.load_state_dict(load("./models/resnet/resnet18-5c106cde.pth"))
		ftrs = self.model.fc.in_features

		# Change network output to binary classification
		self.model.fc = nn.Linear(ftrs, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		y = self.model(x)
		return self.sigmoid(y)


class ScaledDotProduct(torch.nn.Module):
	"""
	Taken from https://pytorch.org/text/stable/_modules/torchtext/nn/modules/multiheadattention.html#ScaledDotProduct
	"""

	def __init__(self, dropout=0.0, batch_first=False):

		super(ScaledDotProduct, self).__init__()
		self.dropout = dropout
		self.batch_first = batch_first

	def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
				attn_mask: Optional[torch.Tensor] = None,
				bias_k: Optional[torch.Tensor] = None,
				bias_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

		if self.batch_first:
			query, key, value = query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2)

		if bias_k is not None and bias_v is not None:
			assert key.size(-1) == bias_k.size(-1) and key.size(-2) == bias_k.size(-2) and bias_k.size(-3) == 1, \
				"Shape of bias_k is not supported"
			assert value.size(-1) == bias_v.size(-1) and value.size(-2) == bias_v.size(-2) and bias_v.size(-3) == 1, \
				"Shape of bias_v is not supported"
			key = torch.cat([key, bias_k])
			value = torch.cat([value, bias_v])
			if attn_mask is not None:
				attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))

		tgt_len, head_dim = query.size(-3), query.size(-1)
		assert query.size(-1) == key.size(-1) == value.size(-1), "The feature dim of query, key, value must be equal."
		assert key.size() == value.size(), "Shape of key, value must match"
		src_len = key.size(-3)
		batch_heads = max(query.size(-2), key.size(-2))

		# Scale query
		query, key, value = query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
		query = query * (float(head_dim) ** -0.5)
		if attn_mask is not None:
			if attn_mask.dim() != 3:
				raise RuntimeError('attn_mask must be a 3D tensor.')
			if (attn_mask.size(-1) != src_len) or (attn_mask.size(-2) != tgt_len) or \
					(attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads):
				raise RuntimeError('The size of the attn_mask is not correct.')
			if attn_mask.dtype != torch.bool:
				raise RuntimeError('Only bool tensor is supported for attn_mask')

		# Dot product of q, k
		attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
		if attn_mask is not None:
			attn_output_weights.masked_fill_(attn_mask, -1e8, )
		attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
		attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
		attn_output = torch.matmul(attn_output_weights, value)

		if self.batch_first:
			return attn_output, attn_output_weights
		else:
			return attn_output.transpose(-3, -2), attn_output_weights

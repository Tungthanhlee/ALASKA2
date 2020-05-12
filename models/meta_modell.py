
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import os
import torchvision
import timm
from timm.models.layers.activations import Swish, Mish
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from config import get_cfg_defaults

def convert_swish_to_mish(model):
	for child_name, child in model.named_children():
		if isinstance(child, Swish):
			setattr(model, child_name, Mish(inplace=True))
		else:
			convert_swish_to_mish(child)

def convert_relu_to_mish(model):
	for child_name, child in model.named_children():
		if isinstance(child, nn.ReLU):
			setattr(model, child_name, Mish(inplace=True))
		else:
			convert_relu_to_mish(child)

class Srnet(nn.Module):
	def __init__(self, cfg):
		super(Srnet, self).__init__()
		# Layer 1
		self.layer1 = nn.Conv2d(in_channels=cfg.DATA.INP_CHANNEL, out_channels=64,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		# Layer 2
		self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(16)
		# Layer 3
		self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn31 = nn.BatchNorm2d(16)
		self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn32 = nn.BatchNorm2d(16)
		# Layer 4
		self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn41 = nn.BatchNorm2d(16)
		self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn42 = nn.BatchNorm2d(16)
		# Layer 5
		self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn51 = nn.BatchNorm2d(16)
		self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn52 = nn.BatchNorm2d(16)
		# Layer 6
		self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn61 = nn.BatchNorm2d(16)
		self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn62 = nn.BatchNorm2d(16)
		# Layer 7
		self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn71 = nn.BatchNorm2d(16)
		self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn72 = nn.BatchNorm2d(16)
		# Layer 8
		self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn81 = nn.BatchNorm2d(16)
		self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn82 = nn.BatchNorm2d(16)
		self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn83 = nn.BatchNorm2d(16)
		self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		# Layer 9
		self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn91 = nn.BatchNorm2d(64)
		self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn92 = nn.BatchNorm2d(64)
		self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn93 = nn.BatchNorm2d(64)
		self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		# Layer 10
		self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn101 = nn.BatchNorm2d(128)
		self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn102 = nn.BatchNorm2d(128)
		self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn103 = nn.BatchNorm2d(128)
		self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		# Layer 11
		self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn111 = nn.BatchNorm2d(256)
		self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn112 = nn.BatchNorm2d(256)
		self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn113 = nn.BatchNorm2d(256)
		self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

		# Layer 12
		self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,
			kernel_size=3, stride=2, padding=0, bias=False)
		self.bn121 = nn.BatchNorm2d(512)
		self.layer122 = nn.Conv2d(in_channels=512, out_channels=512,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn122 = nn.BatchNorm2d(512)
		# avgp = torch.mean() in forward before fc
		# Fully Connected layer
		self.fc = nn.Linear(512*1*1, cfg.TRAIN.NUM_CLASSES)

	def forward(self, inputs):
		# Layer 1
		conv = self.layer1(inputs)
		actv = F.relu(self.bn1(conv))
		# Layer 2
		conv = self.layer2(actv)
		actv = F.relu(self.bn2(conv))
		# Layer 3
		conv1 = self.layer31(actv)
		actv1 = F.relu(self.bn31(conv1))
		conv2 = self.layer32(actv1)
		bn = self.bn32(conv2)
		res = torch.add(actv, bn)
		# Layer 4
		conv1 = self.layer41(res)
		actv1 = F.relu(self.bn41(conv1))
		conv2 = self.layer42(actv1)
		bn = self.bn42(conv2)
		res = torch.add(res, bn)
		# Layer 5
		conv1 = self.layer51(res)
		actv1 = F.relu(self.bn51(conv1))
		conv2 = self.layer52(actv1)
		bn = self.bn52(conv2)
		res = torch.add(res, bn)
		# Layer 6
		conv1 = self.layer61(res)
		actv1 = F.relu(self.bn61(conv1))
		conv2 = self.layer62(actv1)
		bn = self.bn62(conv2)
		res = torch.add(res, bn)
		# Layer 7
		conv1 = self.layer71(res)
		actv1 = F.relu(self.bn71(conv1))
		conv2 = self.layer72(actv1)
		bn = self.bn72(conv2)
		res = torch.add(res, bn)
		# Layer 8
		convs = self.layer81(res)
		convs = self.bn81(convs)
		conv1 = self.layer82(res)
		actv1 = F.relu(self.bn82(conv1))
		conv2 = self.layer83(actv1)
		bn = self.bn83(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 9
		convs = self.layer91(res)
		convs = self.bn91(convs)
		conv1 = self.layer92(res)
		actv1 = F.relu(self.bn92(conv1))
		conv2 = self.layer93(actv1)
		bn = self.bn93(conv2)
		pool = self.pool2(bn)
		res = torch.add(convs, pool)
		# Layer 10
		convs = self.layer101(res)
		convs = self.bn101(convs)
		conv1 = self.layer102(res)
		actv1 = F.relu(self.bn102(conv1))
		conv2 = self.layer103(actv1)
		bn = self.bn103(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 11
		convs = self.layer111(res)
		convs = self.bn111(convs)
		conv1 = self.layer112(res)
		actv1 = F.relu(self.bn112(conv1))
		conv2 = self.layer113(actv1)
		bn = self.bn113(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 12
		conv1 = self.layer121(res)
		actv1 = F.relu(self.bn121(conv1))
		conv2 = self.layer122(actv1)
		bn = self.bn122(conv2)
		# print("L12:",res.shape)
		avgp = torch.mean(bn, dim=(2,3), keepdim=True)
		# fully connected
		flatten = avgp.view(avgp.size(0),-1)
		# print("flatten:", flatten.shape)
		logit = self.fc(flatten)
		# print("FC:",fc.shape)
		# softmax = F.log_softmax(logit, dim=1)
		return logit

class EfficientNet(nn.Module):
	"""
	EfficientNet B0-B8.
	Args:
		cfg (CfgNode): configs
	"""
	def __init__(self, cfg):
		super(EfficientNet, self).__init__()
		model_name = cfg.TRAIN.MODEL
		pretrained = cfg.TRAIN.PRETRAINED
		input_channels = cfg.DATA.INP_CHANNEL
		pool_type = cfg.TRAIN.POOL_TYPE
		drop_connect_rate = cfg.MODEL.DROP_CONNECT
		self.drop_rate = cfg.TRAIN.DROPOUT
		num_classes = cfg.TRAIN.NUM_CLASSES

		backbone = timm.create_model(
			model_name=model_name,
			pretrained=pretrained,
			in_chans=input_channels,
			drop_connect_rate=drop_connect_rate,
		)
		self.conv_stem = backbone.conv_stem
		self.bn1 = backbone.bn1
		self.act1 = backbone.act1
		### Original blocks ###
		for i in range(len((backbone.blocks))):
			setattr(self, "block{}".format(str(i)), backbone.blocks[i])
		self.conv_head = backbone.conv_head
		self.bn2 = backbone.bn2
		self.act2 = backbone.act2
		self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
		self.num_features = backbone.num_features * self.global_pool.feat_mult()
		### Baseline head ###
		self.alaska_fc = nn.Linear(self.num_features, num_classes)
		# Replace with Mish activation
		if cfg.TRAIN.ACTIVATION == "mish":
			convert_swish_to_mish(self)
		del backbone

	def _features(self, x):
		x = self.conv_stem(x)
		x = self.bn1(x)
		x = self.act1(x)
		x = self.block0(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		x = self.block6(x)
		x = self.conv_head(x)
		x = self.bn2(x)
		x = self.act2(x)
		return x

	def forward(self, x):
		x = self._features(x)
		x = self.global_pool(x)
		x = torch.flatten(x, 1)
		if self.drop_rate > 0.:
			x = F.dropout(x, p=self.drop_rate, training=self.training)
		alaska_logits = self.alaska_fc(x)
		return alaska_logits

class ResNet(nn.Module):
	"""
	Generic ResNets.
	Args:
		cfg (CfgNode): configs
	"""
	def __init__(self, cfg):
		super(ResNet, self).__init__()
		model_name = cfg.TRAIN.MODEL
		pretrained = cfg.TRAIN.PRETRAINED
		input_channels = cfg.DATA.INP_CHANNEL
		pool_type = cfg.TRAIN.POOL_TYPE
		self.drop_rate = cfg.TRAIN.DROPOUT
		num_classes = cfg.TRAIN.NUM_CLASSES

		backbone = timm.create_model(
			model_name=model_name,
			pretrained=pretrained,
			in_chans=input_channels
		)
		self.conv1 = backbone.conv1
		self.bn1 = backbone.bn1
		self.act1 = backbone.act1
		self.maxpool = backbone.maxpool
		### Original blocks ###
		self.block1 = backbone.layer1
		self.block2 = backbone.layer2
		self.block3 = backbone.layer3
		self.block4 = backbone.layer4
		self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
		self.num_features = backbone.num_features * self.global_pool.feat_mult()
		### Baseline head ###
		self.alaska_fc = nn.Linear(self.num_features, num_classes)
		# Replace with Mish activation
		if cfg.TRAIN.ACTIVATION == "mish":
			convert_relu_to_mish(self)
		del backbone

	def _features(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act1(x)
		x = self.maxpool(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		return x

	def forward(self, x):
		x = self._features(x)
		x = self.global_pool(x)
		x = torch.flatten(x, 1)
		if self.drop_rate > 0.:
			x = F.dropout(x, p=self.drop_rate, training=self.training)
		alaska_logits = self.alaska_fc(x)
		return alaska_logits


class DenseNet(nn.Module):
	"""
	Generic DenseNets.
	Args:
		cfg (CfgNode): configs
	"""
	def __init__(self, cfg):
		super(DenseNet, self).__init__()
		model_name = cfg.TRAIN.MODEL
		pretrained = cfg.TRAIN.PRETRAINED
		input_channels = cfg.DATA.INP_CHANNEL
		pool_type = cfg.TRAIN.POOL_TYPE
		self.drop_rate = cfg.TRAIN.DROPOUT
		num_classes = cfg.TRAIN.NUM_CLASSES 

		backbone = timm.create_model(
			model_name=model_name,
			pretrained=pretrained,
			in_chans=input_channels
		)

		### Original blocks ###
		self.features = backbone.features
		self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
		self.num_features = backbone.features.norm5.num_features * self.global_pool.feat_mult()
		### Baseline head ###
		self.alaska_fc = nn.Linear(self.num_features, num_classes)
		# Replace with Mish activation
		if cfg.TRAIN.ACTIVATION == "mish":
			convert_relu_to_mish(self)
		del backbone

	def _features(self, x):
		x = self.features(x)
		return x

	def forward(self, x):
		x = self._features(x)
		x = self.global_pool(x)
		x = torch.flatten(x, 1)
		if self.drop_rate > 0.:
			x = F.dropout(x, p=self.drop_rate, training=self.training)
		alaska_logits = self.alaska_fc(x)
		return alaska_logits

if __name__ == "__main__":
	
	cfg = get_cfg_defaults()
	model = Srnet(cfg)
	print(model)
	inpu = torch.rand((5,3,244,244))
	out = model(inpu)
	print(out.shape)
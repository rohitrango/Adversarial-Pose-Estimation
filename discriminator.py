import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import modules

class PoseDiscriminator(nn.Module):

	def __init__(self, in_channels, num_channels, num_joints):
	'''
		Initialisation of the Confidence Discriminator network
		Contains the necessary modules
		Input is pose and confidence heatmaps
		in_channels = num_joints x 2 + 1 (for the image)
	'''	
		super(PoseDiscriminator, self).__init__()
		## Define Layers Here ##

		self.residual1 = Residual(in_channels, num_channels)
		self.residual2 = Residual(num_channels, num_channels)
		self.maxpool   = nn.MaxPool2d(2,2)
		self.residual3 = Residual(num_channels, num_channels)
		self.residual4 = Residual(num_channels, num_channels)
		self.residual5 = Residual(num_channels, num_channels)
		self.fc1 	   = nn.Linear(num_channels*16*16, 128)
		self.relu  	   = nn.ReLU()
		self.fc2 	   = nn.Linear(128, num_joints)

	def forward(self, x):
	"""
		Assuming num channels is 512 and in_channels is 2*num_joints = 32
	"""
		# N x 33 x 256 x 256
		x = self.residual1(x)
		# N x 512 x 256 x 256
		x = self.residual2(x)
		# N x 512 x 256 x 256
		x = self.maxpool(x)
		# N x 512 x 128 x 128
		x = self.residual3(x)
		# N x 512 x 128 x 128
		x = self.maxpool(x)
		# N x 512 x 64 x 64
		x = self.residual4(x)
		# N x 512 x 64 x 64
		x = self.maxpool(x)
		# N x 512 x 32 x 32
		x = self.residual5(x)
		# N x 512 x 32 x 32
		x = self.maxpool(x)
		# N x 512 x 16 x 16
		x = x.view(x.shape[0], -1)
		# N x (512 * 16 * 16)
		x = x.fc1(x)
		# N x 128
		x = x.relu(x)
		# N x 128
		x = x.fc2(x)
		# N x 16
		x = F.sigmoid(x)

		return x


class ConfidenceDiscriminator(nn.Module):

	def __init__(self, in_channels, num_channels, num_joints):
	'''
		Initialisation of the Confidence Discriminator network
		Contains the necessary modules
		Input is pose and confidence heatmaps
		in_channels = num_joints x 2
	'''	
		super(ConfidenceDiscriminator, self).__init__()
		## Define Layers Here ##

		self.residual1 = Residual(in_channels, num_channels)
		self.residual2 = Residual(num_channels, num_channels)
		self.maxpool   = nn.MaxPool2d(2,2)
		self.residual3 = Residual(num_channels, num_channels)
		self.residual4 = Residual(num_channels, num_channels)
		self.residual5 = Residual(num_channels, num_channels)
		self.fc1 	   = nn.Linear(num_channels*16*16, 128)
		self.relu  	   = nn.ReLU()
		self.fc2 	   = nn.Linear(128, num_joints)

	def forward(self, x):
	"""
		Assuming num channels is 512 and in_channels is 2*num_joints = 32
	"""
		# N x 32 x 256 x 256
		x = self.residual1(x)
		# N x 512 x 256 x 256
		x = self.residual2(x)
		# N x 512 x 256 x 256
		x = self.maxpool(x)
		# N x 512 x 128 x 128
		x = self.residual3(x)
		# N x 512 x 128 x 128
		x = self.maxpool(x)
		# N x 512 x 64 x 64
		x = self.residual4(x)
		# N x 512 x 64 x 64
		x = self.maxpool(x)
		# N x 512 x 32 x 32
		x = self.residual5(x)
		# N x 512 x 32 x 32
		x = self.maxpool(x)
		# N x 512 x 16 x 16
		x = x.view(x.shape[0], -1)
		# N x (512 * 16 * 16)
		x = x.fc1(x)
		# N x 128
		x = x.relu(x)
		# N x 128
		x = x.fc2(x)
		# N x 16
		x = F.sigmoid(x)
		
		return x
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from modules import Residual

class Discriminator(nn.Module):

	def __init__(self, in_channels, num_channels, num_joints, num_residuals=5):
		'''
			Initialisation of the Discriminator network
			Contains the necessary modules
			Input is pose and confidence heatmaps
			in_channels = num_joints x 2 + 3 (for the image) (Pose network)
			in_channels = num_joints x 2 (Confidence network)
		'''
		
		super(Discriminator, self).__init__()
		## Define Layers Here ##

		self.residual  = []
		self.num_residuals = num_residuals
		self.residual.append(Residual(in_channels, num_channels))
		for _ in range(num_residuals-1):
			self.residual.append(Residual(num_channels, num_channels))

		self.max_pool  = nn.MaxPool2d(2,2)
		self.fc1 	   = nn.Linear(num_channels*16*16, 128)
		self.relu  	   = nn.ReLU()
		self.fc2 	   = nn.Linear(128, num_joints)

	def forward(self, x):
		"""
			Assuming num channels is 512 and in_channels, num_residuals = 5

		"""
		# N x in_channels x 256 x 256
		x = self.residual[0](x)
		# N x 512 x 256 x 256
		
		for i in range(1, self.num_residuals):
			x = self.residual[i](x)
			x = self.max_pool(x)

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
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class PoseDiscriminator(nn.Module):

	def __init__(self):
	'''
		Initialisation of the Pose Discriminator network
		Contains the necessary modules
	'''	
		super(PoseDiscriminator, self).__init__()
		## Define Layers Here ##


	def forward(self, x):

		return x


class ConfidenceDiscriminator(nn.Module):

	def __init__(self):
	'''
		Initialisation of the Confidence Discriminator network
		Contains the necessary modules
	'''	
		super(ConfidenceDiscriminator, self).__init__()
		## Define Layers Here ##


	def forward(self, x):
		
		return x
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvReluBn(nn.Module):
	'''
		A block of convolution, relu, batchnorm
	'''	

	def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0):

		super(ConvReluBn, self).__init__()

		self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
		self.relu = nn.ReLU()
		self.bn   = nn.BatchNorm2d(out_channels)

	def forward(self, x):

		x = self.conv(x)
		x = self.relu(x)
		x = self.bn(x)

		return x




class Generator(nn.Module):
	'''
		Must be fully convolutional with conv deconv architecture
	'''	

	def __init__(self, num_joints, num_stacks):
		super(Generator, self).__init__()
		
		self.num_stacks = num_stacks
		self.num_joints = num_joints

		## Define Layers Here ##
		self.start = ConvReluBn(3,64, kernel_size = 7, stride = 2, padding = 3)


	def forward(self, x):
	'''
		The forward pass of the network
		Format : batch size x num chan x h x w
	'''

		# N x 3 x 256 x 256
		x = self.start(x)
		# N x 64 x 64 x 64
		




		# N x (2*num_joints) x 64 x 64

		# N x (2*num_joints) x 256 x 256
		
		return x
import torch
import torchvision
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


class ConvTripleBlock(nn.Module):
	'''
		A block of 3 ConvReluBn blocks. 
		This triple block makes up a residual block as described in the paper
		Resolution h x w does not change across this block
	'''	

	def __init__(self, in_channels, out_channels):

		super(ConvTripleBlock, self).__init__()

		out_channels_half = out_channels // 2

		self.convblock1 = ConvReluBn(in_channels,out_channels_half)
		self.convblock2 = ConvReluBn(out_channels_half,out_channels_half,3,1,1)
		self.convblock3 = ConvReluBn(out_channels_half,out_channels)

	def forward(self, x):

		x = self.convblock1(x)
		x = self.convblock2(x)
		x = self.convblock3(x)

		return x

class SkipLayer(nn.Module):
	'''
		The skip connections are necessary for transferring global and local context
		Resolution h x w does not change across this block
	'''

	def __init__(self, in_channels, out_channels):

		super(SkipLayer, self).__init__()
		
		self.in_channels  = in_channels
		self.out_channels = out_channels

		if in_channels != out_channels:
			self.conv = nn.Conv2d(in_channels,out_channels,1)

	def forward(self, x):
		
		if self.in_channels != self.out_channels:
			x = self.conv(x)

		return x

class Residual(nn.Module):
	'''
		The highly used Residual block
		Resolution h x w does not change across this block
	'''
	def __init__(self, in_channels, out_channels):

		super(Residual, self).__init__()

		self.convblock = ConvTripleBlock(in_channels, out_channels)
		self.skip 	   = SkipLayer(in_channels, out_channels)


	def forward(self, x):

		y = self.convblock(x)
		z = self.skip(x)
		o = y + z

		return o



class Generator(nn.Module):
	'''
		Must be fully convolutional with conv deconv architecture
	'''	

	def __init__(self, num_joints, num_stacks):
		
		super(Generator, self).__init__()
		
		self.num_stacks = num_stacks
		self.num_joints = num_joints

		## Define Layers Here ##
		self.startconv 	= ConvReluBn(3,64, kernel_size = 7, stride = 2, padding = 3)
		self.maxpool 	= nn.MaxPool2d(2)
		self.residual1	= nn.Residual(64,512)
		self.residual2	= nn.Residual(512,512)


	def forward(self, x):
	'''
		The forward pass of the network
		Format : batch size x num chan x h x w
	'''

		orig = x+0
		# N x 3 x 256 x 256
		x = self.startconv(x)
		# N x 64 x 128 x 128
		x = self.maxpool(x)
		# N x 64 x 64 x 64
		x = self.residual1(x)
		# N x 512 x 64 x 64
		x = self.residual2(x)
		# N x 512 x 64 x 64
		
		
		## Now this is input to the stacked network





		# N x (num_joints) x 64 x 64

		# N x (num_joints) x 256 x 256
		
		return x
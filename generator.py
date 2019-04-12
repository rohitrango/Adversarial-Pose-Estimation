import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import modules

class Generator(nn.Module):
	"""
		Must be fully convolutional with conv deconv architecture
	"""	

	def __init__(self, num_joints, num_stacks, hourglass_params, mid_channels=512, preprocessed_channels=64):
		
		super(Generator, self).__init__()
		
		self.num_stacks = num_stacks
		self.num_joints = num_joints

		## Define Layers Here ##
		self.start_conv = modules.ConvBnRelu(in_channels=3, out_channels=preprocessed_channels, kernel_size=7, stride=2, padding=3)
		self.max_pool = nn.MaxPool2d(kernel_size=2)
		self.residual = []
		self.residual.append(modules.Residual(in_channels=preprocessed_channels, out_channels=mid_channels))
		self.residual.append(modules.Residual(in_channels=mid_channels, out_channels=mid_channels))

		stacked_hg = []
		stacked_hg_in_channels = []
		for i in range(num_stacks):
			if (i == 0):
				stacked_hg_in_channels.append(mid_channels)
				stacked_hg.append(modules.StackedHourglass(mid_channels, hourglass_params))
			else:
				stacked_hg_in_channels.append(mid_channels + num_joints * 2)
				stacked_hg.append(modules.StackedHourglass(mid_channels + num_joints * 2, hourglass_params))
		self.stacked_hg = stacked_hg		
		
		self.dim_reduction = [[], []]
		for i in range(num_stacks):
			###################################### check if kernel_size is 1 #################
			self.dim_reduction[0].append(nn.Conv2d(in_channels=stacked_hg_in_channels[i], out_channels=num_joints, kernel_size=1, stride=1))
			self.dim_reduction[1].append(nn.Conv2d(in_channels=stacked_hg_in_channels[i], out_channels=num_joints, kernel_size=1, stride=1))

		self.final_upsample = nn.Upsample(scale_factor=mid_channels / preprocessed_channels, mode='nearest')

	def forward(self, x):
		"""
			The forward pass of the network
			Format : batch size x num chan x h x w
		"""

		orig = x
		# N x 3 x 256 x 256
		x = self.start_conv(x)
		# N x 64 x 128 x 128
		x = self.max_pool(x)
		# N x 64 x 64 x 64
		x = self.residual[0](x)
		# N x 512 x 64 x 64
		x = self.residual[1](x)
		# N x 512 x 64 x 64
		
		
		## Now this is input to the stacked network
		inp = x
		out = [None for _  in range(self.num_stacks)]
		for i in range(self.num_stacks):
			out[i] = self.stacked_hg[i](inp)
			out[i][0] = self.dim_reduction[0][i](out[i][0])
			out[i][1] = self.dim_reduction[1][i](out[i][1])
			inp = torch.cat((out[i][0], out[i][1], x), dim=1)

		for _ in range(self.num_stacks):
			out[i] = torch.cat(out[i], dim=1)
			out[i] = self.final_upsample(out[i])

		return out
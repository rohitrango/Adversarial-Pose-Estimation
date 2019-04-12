import argparse
import importlib
import random
import os
import shutil
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

from datasets.lsp import LSP
from generator import Generator
from discriminator import Discriminator
from losses import gen_single_loss, disc_single_loss

parser = argparse.ArgumentParser()
parser.add_argument('--modelName', required=True, help='name of model; name used to create folder to save model')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, testing')
parser.add_argument('--print_every', type=int, default=1000, help='frequency to print train loss, accuracy to terminal')
parser.add_argument('--save_every', type=int, default=1, help='frequency of saving the model')
parser.add_argument('--optimizer_type', default='SGD', help='type of optimizer to use (SGD or Adam)')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')

parser.add_argument('--path', \
    default='/home/rohitrango/CSE_IITB/SEM8/CS763/Adversarial-Pose-Estimation/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=0.85)
parser.add_argument('--heatmap_sigma', type=float, default=2)
parser.add_argument('--occlusion_sigma', type=float, default=5)

args = parser.parse_args()

# initialize seed to constant for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# create directory to store models if it doesn't exist
# if it exists, delete the contents in the directory
if (not os.path.exists(args.modelName)):
    os.makedirs(args.modelName)
else:
    shutil.rmtree(args.modelName)
    os.makedirs(args.modelName)

# for handling training over GPU
cpu_device = torch.device('cpu')
fast_device = torch.device('cpu')
if (args.use_gpu):
    fast_device = torch.device('cuda:0')

# config file storing hyperparameters
config = importlib.import_module(args.config).config


# Initializing the models
generator_model = Generator(config['dataset']['num_joints'], config['generator']['num_stacks'], config['generator']['hourglass_params'], config['generator']['mid_channels'], config['generator']['preprocessed_channels'])
discriminator_model = Discriminator(config['discriminator']['in_channels'], config['discriminator']['num_channels'], config['dataset']['num_joints'], config['discriminator']['num_residuals'])

# Dataset and the Dataloader
lsp_dataset = LSP(args)
train_loader = torch.utils.data.DataLoader(lsp_dataset, batch_size=args.batch_size, shuffle=True)

# Loading on GPU, if available
if (args.use_gpu):
    generator_model = generator_model.cuda()
    discriminator_model = discriminator_model.cuda()

# Cross entropy loss
criterion = nn.CrossEntropyLoss()

# Setting the optimizer
if (args.optimizer_type == 'SGD'):
    optim_gen = optim.SGD(generator_model.parameters(), lr=args.lr, momentum=args.momentum)
    optim_disc = optim.SGD(discriminator_model.parameters(), lr=args.lr, momentum=args.momentum)

elif (args.optimizer_type == 'Adam'):
    optim_gen = optim.Adam(generator_model.parameters(), lr=args.lr)
    optim_disc = optim.Adam(discriminator_model.parameters(), lr=args.lr)

else:
    raise NotImplementedError

# The main training loop
gen_losses = []
disc_losses = []

for epoch in range(args.epochs):
    print('epoch:', epoch)

    if (epoch % args.save_every == 0):
        torch.save({'generator_model': generator_model,
                    'discriminator_model': discriminator_model,
                    'criterion': criterion, 
                    'optim_gen': optim_gen, 
                    'optim_disc': optim_disc}, os.path.join(args.modelName, 'model_' + str(epoch) + '.pt'))

    epoch_gen_loss = 0.0
    epoch_disc_loss = 0.0

    for i, data in enumerate(train_loader):

        optim_gen.zero_grad()
        optim_disc.zero_grad()
        
        images = data['image']
        ground_truth = {}
        ground_truth['heatmaps'] = data['heatmaps']
        ground_truth['occlusions'] = data['occlusions']

    ################################## Check and complete the code here #######################################################

        ############# Forward pass and calculate losses here #########
        outputs = generator_model(images)

        cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, discriminator_model)
        cur_disc_loss_dic = disc_single_loss(ground_truth, outputs, discriminator_model)

        cur_gen_loss = cur_gen_loss_dic['loss']
        cur_disc_loss = cur_disc_loss_dic['loss']

        ######### Backpropagating the losses here #######

        cur_gen_loss.backward()
        cur_disc_loss.backward()

        optim_gen.step()
        optim_disc.step()

        gen_losses.append(cur_gen_loss)
        disc_losses.append(cur_disc_loss)

        epoch_gen_loss += cur_gen_loss
        epoch_disc_loss += cur_disc_loss

        if i % args.print_every == 0:
            print("Train iter: %d, generator loss : %f, discriminator loss : %f" % (i ,gen_losses[-1], disc_losses[-1]))

    epoch_gen_loss /= len(lsp_dataset)
    epoch_disc_loss /= len(lsp_dataset)

    print('Epoch train gen loss: %f' % (epoch_gen_loss))
    print('Epoch train disc loss: %f' % (epoch_disc_loss))

    ######################################################################################################################

# Saving the model and the losses
torch.save({'generator_model': generator_model,
            'discriminator_model': discriminator_model,
            'criterion': criterion, 
            'optim_gen': optim_gen, 
            'optim_disc': optim_disc}, os.path.join(args.modelName, 'model_' + str(epoch) + '.pt'))

with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
    pickle.dump((disc_losses, gen_losses), f)

# Plotting the loss function
plt.plot(loss)
plt.savefig(os.path.join(args.modelName, 'loss_graph.pdf'))
plt.clf()
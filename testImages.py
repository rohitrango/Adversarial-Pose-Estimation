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
import cv2

from datasets.lsp import LSP
from generator import Generator
from discriminator import Discriminator
from losses import gen_single_loss, disc_single_loss, get_loss_disc
import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--modelName', required=True, help='name of model; name used to create folder to save model')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, testing')
parser.add_argument('--print_every', type=int, default=1000, help='frequency to print train loss, accuracy to terminal')
parser.add_argument('--save_every', type=int, default=10, help='frequency of saving the model')
parser.add_argument('--optimizer_type', default='SGD', help='type of optimizer to use (SGD or Adam)')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')
parser.add_argument('--gpu_device', type=int, default=0, help='GPU device which needs to be used for computation')
parser.add_argument('--validation_sample_size', type=int, default=1, help='size of validation sample')
parser.add_argument('--validate_every', type=int, default=50, help='frequency of evaluating on validation set')

parser.add_argument('--path', \
    default='/home/rohitrango/CSE_IITB/SEM8/CS763/Adversarial-Pose-Estimation/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=0.95)
parser.add_argument('--heatmap_sigma', type=float, default=2)
parser.add_argument('--occlusion_sigma', type=float, default=5)
parser.add_argument('--loss', type=str, default='mse')

args = parser.parse_args()

# initialize seed to constant for reproducibility
np.random.seed(58)
torch.manual_seed(58)
random.seed(58)


with torch.no_grad():
    # for handling training over GPU
    cpu_device = torch.device('cpu')
    fast_device = torch.device('cpu')
    if (args.use_gpu):
        fast_device = torch.device('cuda:' + str(args.gpu_device))

    # config file storing hyperparameters
    config = importlib.import_module(args.config).config

    # Initializing the models
    generator_model = Generator(config['dataset']['num_joints'], config['generator']['num_stacks'], config['generator']['hourglass_params'], config['generator']['mid_channels'], config['generator']['preprocessed_channels'])
    discriminator_model = Discriminator(config['discriminator']['in_channels'], config['discriminator']['num_channels'], config['dataset']['num_joints'], config['discriminator']['num_residuals'])

    # Load
    model_data = torch.load(args.modelName)
    generator_model = model_data['generator_model']

    # Use dataparallel
    generator_model = nn.DataParallel(generator_model)
    discriminator_model = nn.DataParallel(discriminator_model)

    # Dataset and the Dataloader
    lsp_train_dataset = LSP(args)
    args.mode = 'val'
    lsp_val_dataset = LSP(args)
    train_loader = torch.utils.data.DataLoader(lsp_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_save_loader = torch.utils.data.DataLoader(lsp_val_dataset, batch_size=args.batch_size)
    val_eval_loader = torch.utils.data.DataLoader(lsp_val_dataset, batch_size=1, shuffle=True)

    pck = metrics.PCK(metrics.Options(256, 4))

    # Loading on GPU, if available
    if (args.use_gpu):
        generator_model = generator_model.to(fast_device)
        discriminator_model = discriminator_model.to(fast_device)

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

    # Save images here
    for i, data in enumerate(val_eval_loader):
        optim_gen.zero_grad()
        optim_disc.zero_grad()
        
        images = data['image']
        ground_truth = {}
        ground_truth['heatmaps'] = data['heatmaps']
        ground_truth['occlusions'] = data['occlusions']
        if (args.use_gpu):
            images = images.to(fast_device)
            ground_truth['heatmaps'] = ground_truth['heatmaps'].to(fast_device)
            ground_truth['occlusions'] = ground_truth['occlusions'].to(fast_device)

        # Get outputs
        outputs = generator_model(images + 0)
        plt.subplot(1, 3, 1)

        # print(images.max(), images.min())

        plt.imshow(images.detach().cpu().numpy()[0].transpose(1, 2, 0)/255.0)
        # plt.imshow((images.detach().cpu().numpy()[0].transpose(1, 2, 0) * 128 + 128).astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.imshow(np.sum(ground_truth['heatmaps'].detach().cpu().numpy()[0], axis=0))
        plt.subplot(1, 3, 3)
        plt.imshow(outputs[-1][0, 1, :, :].cpu().numpy())
        # plt.imshow(np.sum(outputs[-1][:, 0, :, :].detach().cpu().numpy()[0], axis=0))
        plt.show()
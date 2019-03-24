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

from datasets import lsp
import generator, discriminator

import Model

####################################
parser = argparse.ArgumentParser()
parser.add_argument('--modelName', required=True, help='name of model; name used to create folder to save model')
parser.add_argument('--data', required=True, help='path to training data (train_data.txt)')
parser.add_argument('--target', required=True, help='path to training labels (train_labels.txt)')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, testing')
parser.add_argument('--print_every', type=int, default=1000, help='frequency to print train loss, accuracy to terminal')
parser.add_argument('--save_every', type=int, default=1, help='frequency of saving the model')
parser.add_argument('--fraction_validation', type=float, default=0.1, help='fraction of data to be used for validation')
parser.add_argument('--optimizer_type', default='SGD', help='type of optimizer to use (SGD or Adam)')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')

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
####################################



(X_train, y_train), (X_val, y_val), word_to_index = data_loader.load_train_val_dataset(args.data, args.target, args.fraction_validation)
train_dataset = data_loader.ListDataset(X_train, y_train, word_to_index)
val_dataset = None
if (X_val is not None):
    val_dataset = data_loader.ListDataset(X_val, y_val, word_to_index)

pad_tensor = data_loader.one_hot_tensor(word_to_index['PAD'], len(word_to_index))
train_loader = data_loader.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_loader.PadCollate(config['dataset']['seq_max_len'], pad_tensor, config['dataset']['pad_beginning'], config['dataset']['truncate_end']))
val_loader = None
if (X_val is not None):
    val_loader = data_loader.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_loader.PadCollate(config['dataset']['seq_max_len'], pad_tensor, config['dataset']['pad_beginning'], config['dataset']['truncate_end']))


####################################
generator_model = Generator(config['dataset']['num_joints'], config['generator']['num_stacks'], config['generator']['hourglass_params'], config['generator']['mid_channels'], config['generator']['preprocessed_channels'])
discriminator_model = Discriminator(config['discriminator']['in_channels'], config['discriminator']['num_channels'], config['dataset']['num_joints'], config['discriminator']['num_residuals'])

if (args.use_gpu):
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

if (args.optimizer_type == 'SGD'):
    optim = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
elif (args.optimizer_type == 'Adam'):
    optim = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError
####################################

loss = []
acc = []
val_acc = []

def get_init_state(network_params, batch_size, fast_device):
    if (network_params['cell_type'] == 'RNN'):
        init_state = torch.zeros((network_params['num_layers'], batch_size, network_params['hidden_size']))
        init_state = init_state.to(fast_device)
    elif (network_params['cell_type'] == 'LSTM'):
        init_state = (torch.zeros((network_params['num_layers'], batch_size, network_params['hidden_size'])), 
                      torch.zeros((network_params['num_layers'], batch_size, network_params['hidden_size'])))
        init_state = (init_state[0].to(fast_device), init_state[1].to(fast_device))
    return init_state

def get_accuracy(model, data_loader, fast_device):
    acc = 0.0
    num_data = 0
    data_loader.reset_pos()
    with torch.no_grad():
        while (not data_loader.is_done_epoch()):
            batch_xs, batch_ys = data_loader.next_batch()
            batch_xs, batch_ys = batch_xs.to(fast_device), batch_ys.to(fast_device)
            
            cur_batch_size = batch_xs.size(0)
            # init_state = get_init_state(config['network'], batch_xs.size(1), fast_device)
            scores = model.forward(batch_xs)

            acc += torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item() * 1.0
            num_data += cur_batch_size

    return acc / num_data

for epoch in range(args.epochs):
    print('epoch:', epoch)
    i = 0

    if (epoch % args.save_every == 0):
        torch.save({'model': model, 
                    'criterion': criterion, 
                    'optim': optim, 
                    'word_to_index': word_to_index}, os.path.join(args.modelName, 'model_' + str(epoch) + '.pt'))

    val_acc.append(get_accuracy(model, val_loader, fast_device))
    print('Validation Accuracy: %f' % (val_acc[-1], ))

    epoch_loss, epoch_acc = 0.0, 0.0
    num_train = 0
    train_loader.reset_pos()
    while (not train_loader.is_done_epoch()):
        batch_xs, batch_ys = train_loader.next_batch()
        batch_xs, batch_ys = batch_xs.to(fast_device), batch_ys.to(fast_device)
        cur_batch_size = batch_xs.size(0)
        optim.zero_grad()
        # init_state = get_init_state(config['network'], cur_batch_size, fast_device)
        scores = model.forward(batch_xs)
        
        cur_loss = criterion.forward(scores, batch_ys)
        cur_acc = torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item() * 1.0 / cur_batch_size
        epoch_loss += cur_loss.item() * cur_batch_size
        epoch_acc += cur_acc * cur_batch_size
        num_train += cur_batch_size

        loss.append(cur_loss.item())
        acc.append(cur_acc)
        
        grad_output = criterion.backward(scores, batch_ys)
        model.backward(batch_xs, grad_output)
        optim.step()
        
        if (i % args.print_every == 0):
            print("iter: %d, Train loss : %f, Train acc : %f" % (i ,loss[-1], acc[-1]))

        i += 1

    epoch_loss /= num_train
    epoch_acc /= num_train

    print('Epoch train loss: %f, epoch train acc: %f' % (epoch_loss, epoch_acc))

torch.save({'model': model, 
            'criterion': criterion, 
            'optim': optim, 
            'word_to_index': word_to_index}, os.path.join(args.modelName, 'model_final.pt'))

with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
    pickle.dump((val_acc, loss, acc), f)

with open(os.path.join(args.modelName, 'stats.txt'), 'w') as f:
    f.write('Validation accuracy : %f' % (val_acc[-1]))

plt.plot(val_acc)
plt.savefig(os.path.join(args.modelName, 'val_acc_graph.pdf'))
plt.clf()

plt.plot(loss)
plt.savefig(os.path.join(args.modelName, 'loss_graph.pdf'))
plt.clf()
plt.plot(acc)
plt.savefig(os.path.join(args.modelName, 'acc_graph.pdf'))
plt.clf()
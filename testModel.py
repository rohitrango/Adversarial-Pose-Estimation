"""
Creates torch RNN model and trains it with the given dataset
"""

import argparse
import importlib
import random
import os
import shutil
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import data_loader
import optim
import criterion

import Model

parser = argparse.ArgumentParser()
parser.add_argument('-modelName', required=True, help='name of model; name used to create folder to save model')
parser.add_argument('-data', required=True, help='path to training data (train_data.txt)')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, testing')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')

args = parser.parse_args()

# initialize seed to constant for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# for handling training over GPU
cpu_device = torch.device('cpu')
fast_device = torch.device('cpu')
if (args.use_gpu):
    fast_device = torch.device('cuda:0')

# config file storing hyperparameters
config = importlib.import_module(args.config).config

X_test = data_loader.load_test_data(args.data)
y_test_dummy = [int(0) for _ in range(len(X_test))]
load_dict = torch.load(os.path.join(args.modelName, 'model_42.pt'), map_location=fast_device)
word_to_index = load_dict['word_to_index']
test_dataset = data_loader.ListDataset(X_test, y_test_dummy, word_to_index)

pad_tensor = data_loader.one_hot_tensor(word_to_index['PAD'], len(word_to_index))
test_loader = data_loader.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_loader.PadCollate(config['dataset']['seq_max_len'], pad_tensor, config['dataset']['pad_beginning'], config['dataset']['truncate_end']))

model = load_dict['model']
if (args.use_gpu):
    model = model.cuda()

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

def get_predictions(model, data_loader, fast_device):
    data_loader.reset_pos()
    pred = None
    with torch.no_grad():
        while (not data_loader.is_done_epoch()):
            batch_xs, batch_ys = data_loader.next_batch()
            batch_xs, batch_ys = batch_xs.to(fast_device), batch_ys.to(fast_device)
            
            scores = model.forward(batch_xs)
            cur_pred = torch.argmax(scores, dim=1)
            if (pred is None):
                pred = cur_pred
            else:
                pred = torch.cat([pred, cur_pred], dim=0)

    return pred

pred = get_predictions(model, test_loader, fast_device)
torch.save({'predictions': pred}, 'testPredictions.bin')

pred = pred.numpy()
with open(os.path.join(args.modelName, 'test_pred.txt'), 'w') as f:
	f.write('id,label\n')
	for i in range(pred.shape[0]):
		f.write(str(i))
		f.write(',')
		f.write('%d\n' % (pred[i], ))
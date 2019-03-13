'''
LSP Dataset
'''
from os.path import join
import argparse

from glob import glob
import cv2
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='/home/rohitrango/CSE_IITB/\SEM8/CS763/Adversarial-Pose-Estimation/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)

class LSP(Dataset):
    '''
    LSP dataset
    '''
    def __init__(self, cfg):
        # Path = dataset path, mode = train/val
        self.path = cfg.path
        self.mode = cfg.mode
        self.crop_size = cfg.crop_size
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        self.files = sorted(glob(join(self.path, 'images/*.jpg')))
        self.annot = loadmat(join(self.path, 'joints.mat'))['joints']

    def __len__(self):
        # Return length
        return len(self.files)

    def __getitem__(self, idx):
        # Get the i'th entry
        file_name = self.files[idx]
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        crop_image = cv2.resize(image, (self.crop_size, self.crop_size))

        # Read annotations
        annot = self.annot[:, :, idx]
        annot[:, :2] = annot[:, :2] * np.array(\
            [[self.crop_size*1.0/image.shape[1], self.crop_size*1.0/image.shape[0]]])

        return {
            # image is in CHW format
            'image': torch.Tensor(crop_image.transpose(2, 0, 1)),
            'kp_2d': torch.Tensor(annot),
            # TODO: Return heatmaps
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = LSP(args)
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        plt.clf()
        plt.imshow(data['image'].numpy().transpose(1, 2, 0)/255.0)
        plt.scatter(data['kp_2d'][:, 0].numpy(), data['kp_2d'][:, 1].numpy(), c=data['kp_2d'][:, 2])
        plt.draw()
        plt.pause(0.001)

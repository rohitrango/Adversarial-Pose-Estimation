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
# from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='/home/rohitrango/CSE_IITB/SEM8/CS763/Adversarial-Pose-Estimation/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=0.85)
parser.add_argument('--heatmap_sigma', type=float, default=2)
parser.add_argument('--occlusion_sigma', type=float, default=5)

class LSP(Dataset):
    '''
    LSP dataset
    '''
    def __init__(self, cfg):
        # Path = dataset path, mode = train/val
        self.path = cfg.path
        self.mode = cfg.mode
        self.crop_size = cfg.crop_size
        self.train_split = cfg.train_split
        self.heatmap_sigma = cfg.heatmap_sigma
        self.occlusion_sigma = cfg.occlusion_sigma

        assert self.mode in ['train', 'val'], 'invalid mode {}'.format(self.mode)
        assert cfg.train_split > 0 and cfg.train_split < 1, 'train_split should be a fraction'
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        self.files = sorted(glob(join(self.path, 'images/*.jpg')))
        self.annot = loadmat(join(self.path, 'joints.mat'))['joints']

    def __len__(self):
        # Return length
        if self.mode == 'train':
            return int(self.train_split * len(self.files))
        else:
            return len(self.files) - int(self.train_split * len(self.files))


    def __getitem__(self, idx):
        # if validation, offset index
        if self.mode == 'val':
            idx += int(self.train_split * len(self.files))

        # Get the i'th entry
        file_name = self.files[idx]
        # if (self.mode == 'val'):
            # print('reading ', file_name)
        # image = Image.open(file_name)
        # b, g, r = image.split()
        # image = image.merge('RGB', (r, g, b))
        # image = image.resize((self.crop_size, self.crop_size))
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        crop_image = cv2.resize(image, (self.crop_size, self.crop_size))

        # Read annotations
        annot = self.annot[:, :, idx] + 0.0
        # annot = K * 3
        annot[:, :2] = annot[:, :2] * np.array(\
            [[self.crop_size*1.0/image.shape[1], self.crop_size*1.0/image.shape[0]]])

        # Generate heatmaps
        x = range(self.crop_size)
        xx, yy = np.meshgrid(x, x)
        heatmaps = np.zeros((annot.shape[0], self.crop_size, self.crop_size))
        occlusions = np.zeros((annot.shape[0], self.crop_size, self.crop_size)) 
        # Annotate heatmap
        for joint_id in range(annot.shape[0]):
            x_c, y_c, vis = annot[joint_id] + 0
            heatmaps[joint_id] = np.exp(-0.5*((x_c - xx)**2 + (y_c - yy)**2)/(self.heatmap_sigma**2))
            occlusions[joint_id] = (1 - vis)*np.exp(-0.5*((x_c - xx)**2 + (y_c - yy)**2)/(self.occlusion_sigma**2))

        return {
            # image is in CHW format
            'image': torch.Tensor(crop_image.transpose(2, 0, 1)),
            'kp_2d': torch.Tensor(annot),
            'heatmaps': torch.Tensor(heatmaps),
            'occlusions': torch.Tensor(occlusions),
            # TODO: Return heatmaps
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = LSP(args)
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(data['image'].numpy().transpose(1, 2, 0)/255.0)
        plt.scatter(data['kp_2d'][:, 0].numpy(), data['kp_2d'][:, 1].numpy(), c=data['kp_2d'][:, 2])

        plt.subplot(1, 3, 2)
        plt.imshow(data['heatmaps'].numpy().sum(0), 'jet')

        plt.subplot(1, 3, 3)
        plt.imshow(data['occlusions'].numpy().sum(0), 'jet')

        plt.show()


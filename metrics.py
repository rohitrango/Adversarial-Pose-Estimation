import torch
import numpy as np
import matplotlib.pyplot as plt
eps = 1e-8

class Options(object):
    def __init__(self, outputRes, nStack):
        self.outputRes = outputRes
        self.nStack = nStack

class PCK(object):
    """docstring for PCK"""
    def __init__(self, opts):
        super(PCK, self).__init__()
        self.opts = opts

    def calc_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists

    def dist_acc(self, dists, thr=0.5):
         ''' Return percentage below threshold while ignoring values with a -1 '''
         dist_cal = np.not_equal(dists, -1)
         num_dist_cal = dist_cal.sum()
         if num_dist_cal > 0:
             return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
         else:
             return -1

    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        # print('before mask', preds)

        preds *= pred_mask
        # print('after mask', preds)
        return preds, maxvals

    def eval(self, pred, target, alpha=0.5):
        '''
        Calculate accuracy according to PCK,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        '''
        idx = list(range(14))
        norm = 1.0
        if True:
         h = self.opts.outputRes
         w = self.opts.outputRes
         norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        dists = self.calc_dists(pred, target, norm)

        acc = np.zeros((len(idx) + 1))
        avg_acc = 0.0
        cnt = 0

        for i in range(len(idx)):
         acc[i + 1] = self.dist_acc(dists[idx[i]])
         if acc[i + 1] >= 0:
             avg_acc = avg_acc + acc[i + 1]
             cnt += 1
         print('acc[%d] = %f' % (i + 1, acc[i + 1]))

        avg_acc = 1.0 * avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
         acc[0] = avg_acc
        return avg_acc,cnt

    def StackedHourGlass(self, output, target, alpha=0.5):
        predictions = self.get_max_preds(output[self.opts.nStack-1].detach().cpu().numpy())
        
        comb_pred = np.sum(output[-1].detach().cpu().numpy()[0], axis=0)
        # print(comb_pred)
        # plt.imshow(comb_pred)
        # plt.savefig('comb_hmap.png')
        # plt.clf()
        plt.imshow(np.sum(target.detach().cpu().numpy()[0], axis=0))
        plt.savefig('gt_hmap.png')
        plt.clf()

        target = self.get_max_preds(target.cpu().numpy())

        return self.eval(predictions[0], target[0], alpha)

    def PyraNet(self, output, target, alpha=0.5):
        predictions = self.get_max_preds(output[self.opts.nStack-1].detach().cpu().numpy())
        target = self.get_max_preds(target.cpu().numpy())
        return self.eval(predictions[0], target[0], alpha)

    def PoseAttention(self, output, target, alpha=0.5):
        predictions = self.get_max_preds(output[self.opts.nStack-1].detach().cpu().numpy())
        target = self.get_max_preds(target.cpu().numpy())
        return self.eval(predictions[0], target[0], alpha)

    def ChainedPredictions(self, output, target, alpha=0.5):
        predictions = self.get_max_preds(output.detach().cpu().numpy())
        target = self.get_max_preds(target.cpu().numpy())
        return self.eval(predictions[0], target[0], alpha)

    def DeepPose(self, output, target, alpha=0.5):
        predictions = (0. + (output).reshape(-1,16,2).detach().cpu().numpy())*self.opts.outputRes
        target = (0. + (target).reshape(-1,16,2).cpu().numpy())*self.opts.outputRes
        return self.eval(predictions, target, alpha)
# --------------------------------------------------------
# Copyright (c) 2019
# --------------------------------------------------------

import os, sys
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0,os.path.join(os.path.dirname(__file__), ".."))
from data_process import DataProcess, get_while_running, kill_data_processes
from data_utils import load_h5, load_csv, augment_cloud, pad_cloudN
from vis import plot_pcds


class ShapenetDataProcess(DataProcess):

    def __init__(self, data_queue, args, split='train', repeat=True):
        """Shapenet dataloader.

        Args:
            data_queue: multiprocessing queue where data is stored at.
            split: str in ('train', 'val', 'test'). Loads corresponding dataset.
            repeat: repeats epoch if true. Terminates after one epoch otherwise.
        """
        self.args=args
        self.split = split
        args.DATA_PATH = 'data/%s' % (args.dataset)
        classmap = load_csv(args.DATA_PATH + '/synsetoffset2category.txt')
        args.classmap = {}
        for i in range(classmap.shape[0]):
            args.classmap[str(classmap[i][1]).zfill(8)] = classmap[i][0]

        self.data_paths = sorted([os.path.join(args.DATA_PATH, split, 'partial', k.rstrip()+ '.h5') for k in open(args.DATA_PATH + '/%s.list' % (split)).readlines()])
        N = int(len(self.data_paths)/args.batch_size)*args.batch_size
        self.data_paths = self.data_paths[0:N]
        super().__init__(data_queue, self.data_paths, None, args.batch_size, repeat=repeat)

    def get_pair(self, args, fname, train):
        partial = load_h5(fname)
        gtpts = load_h5(fname.replace('partial', 'gt'))
        if train:
            gtpts, partial = augment_cloud([gtpts, partial], args)
        partial  = pad_cloudN(partial, args.inpts)
        return partial, gtpts

    def load_data(self, fname):
        pair = self.get_pair(self.args, fname, train=self.split == 'train')
        partial = pair[0].T
        target = pair[1]
        cloud_meta = ['{}.{:d}'.format('/'.join(fname.split('/')[-2:]),0),]
        return target[np.newaxis, ...], cloud_meta, partial[np.newaxis, ...]

    def collate(self, batch):
        targets, clouds_meta, clouds = list(zip(*batch))
        targets = np.concatenate(targets, 0)
        if len(clouds_meta[0])>0:
            clouds = np.concatenate(clouds, 0)
            clouds_meta = [item for sublist in clouds_meta for item in sublist]
        return targets, (clouds_meta, clouds)

def test_process():
    from multiprocessing import Queue
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.dataset = 'shapenet'
    args.nworkers = 4
    args.batch_size=32
    args.pc_augm_scale=0
    args.pc_augm_rot=0
    args.pc_augm_mirror_prob=0
    args.pc_augm_jitter=0
    args.inpts=2048
    data_processes = []
    data_queue = Queue(1)
    for i in range(args.nworkers):
        data_processes.append(ShapenetDataProcess(data_queue, args, split='train',
                repeat=False))
        data_processes[-1].start()

    for targets, clouds_data in get_while_running(data_processes, data_queue, 0.5):
        inp = clouds_data[1][0].squeeze().T
        targets = targets[0]
        plot_pcds(None, [inp.squeeze(), targets.squeeze()], ['partial', 'gt'], use_color=[0, 0], color=[None, None])

    kill_data_processes(data_queue, data_processes)

if __name__ == '__main__':
    test_process()

